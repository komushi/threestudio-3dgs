"""
Interior camera data module for threestudio-3dgs.

Provides outward-looking cameras positioned inside a room, for interior scene
rendering. This is the inverse topology of the default object-centric cameras.

See design/D-2c_threestudio_interior_camera.md for full specification.
"""

import math
from dataclasses import dataclass, field

import threestudio
import torch
import torch.nn.functional as F
from threestudio.data.uncond import (
    RandomCameraDataModule,
    RandomCameraDataModuleConfig,
)
from threestudio.utils.ops import get_mvp_matrix


@threestudio.register("interior-camera-datamodule")
class InteriorCameraDataModule(RandomCameraDataModule):
    @dataclass
    class Config(RandomCameraDataModuleConfig):
        # Fixed camera position inside the room (world coordinates, meters).
        # Default: typical standing eye height near room centroid.
        camera_position: tuple[float, float, float] = (0.0, 0.84, 0.0)

        # Small jitter around the fixed position, in meters.
        # Set to 0 for strictly single-position (matches our D-1 data).
        position_jitter: float = 0.0

    cfg: Config

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

        # Pre-compute unit direction vectors for the grid (same as parent)
        # These will be rotated to arbitrary outward-looking directions
        hx, hy = self.width // 2, self.height // 2
        x, y = torch.arange(-hx, hx), torch.arange(-hy, hy)
        # note: flip y axis to match the convention of image coordinates
        y = -y
        self.directions_unit_focal = torch.stack(
            [
                torch.outer(x.to(torch.float32), torch.ones_like(y)).T,
                torch.outer(torch.ones_like(x), y).T,
                torch.zeros_like(x),
            ],
            dim=-1,
        )[None, :, :, :]

    def __iter__(self):
        while True:
            yield self.__next__()

    def __next__(self):
        batch_size = self.batch_size

        # Sample elevation and azimuth for look direction (same as parent)
        elevation = (
            torch.rand(batch_size) * (self.elevation_range[1] - self.elevation_range[0])
            + self.elevation_range[0]
        ) * math.pi / 180
        azimuth = (
            torch.rand(batch_size) * (self.azimuth_range[1] - self.azimuth_range[0])
            + self.azimuth_range[0]
        ) * math.pi / 180

        # INTERIOR CAMERA: Fixed position with optional jitter
        # NOT orbiting around origin like object-centric cameras
        pos = torch.tensor(self.cfg.camera_position, dtype=torch.float32)
        camera_positions = pos.expand(batch_size, -1).clone()

        if self.cfg.position_jitter > 0:
            camera_positions += (
                torch.randn_like(camera_positions) * self.cfg.position_jitter
            )

        # INTERIOR CAMERA: Look direction is sampled OUTWARD from camera position
        # NOT looking toward origin
        lookat = torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ],
            dim=-1,
        )  # already unit length, pointing outward

        # Default camera up direction as +z
        up: torch.Tensor = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(batch_size, 1)

        # Sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: torch.Tensor = (
            torch.rand(batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb

        # Sample FOVs from a uniform distribution bounded by fov_range
        fovy_deg: torch.Tensor = (
            torch.rand(batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # Sample light distance
        light_distances: torch.Tensor = (
            torch.rand(batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        # Sample light direction (same as parent - dreamfusion strategy)
        light_direction: torch.Tensor = F.normalize(
            camera_positions
            + torch.randn(batch_size, 3) * self.cfg.light_position_perturb,
            dim=-1,
        )
        light_positions: torch.Tensor = light_direction * light_distances[:, None]

        # Compute camera basis: right, up, lookat
        # lookat is already normalized (outward direction)
        right: torch.Tensor = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)

        # Build camera-to-world matrix
        # Camera convention: X right, Y up, Z -lookat (OpenGL/COLMAP)
        c2w3x4: torch.Tensor = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: torch.Tensor = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        # Get ray directions (same as parent)
        focal_length: torch.Tensor = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: torch.Tensor = self.directions_unit_focal[None, :, :, :].repeat(
            batch_size, 1, 1, 1
        )
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Transform directions to world space
        # directions is in camera space: (B, H, W, 3) with X right, Y down, Z forward
        # We need to rotate by the camera orientation
        # camera_basis = [right, up, -lookat] as columns
        directions = (
            directions[..., 0, None] * right[:, None, None, :]
            + directions[..., 1, None] * up[:, None, None, :]
            - directions[..., 2, None] * lookat[:, None, None, :]
        )

        # rays_o is camera position, rays_d is the direction
        rays_o: torch.Tensor = camera_positions[:, None, None, :].repeat(
            1, self.height, self.width, 1
        )
        rays_d: torch.Tensor = F.normalize(directions, dim=-1)

        # Compute MVP matrix for rasterizer
        w2c: torch.Tensor = torch.linalg.inv(c2w)
        c2p: torch.Tensor = torch.linalg.inv(
            torch.eye(4, device=c2w.device).unsqueeze(0).repeat(batch_size, 1, 1)
        )
        mvp_mtx: torch.Tensor = get_mvp_matrix(w2c, c2p, fovy)

        # Build batch dictionary (same keys as parent for compatibility)
        batch = {
            "c2w": c2w,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "fovy": fovy,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "light_positions": light_positions,
            "height": self.height,
            "width": self.width,
            "index": torch.arange(batch_size),
        }

        return batch
