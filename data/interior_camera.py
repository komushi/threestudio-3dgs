"""
Interior camera data module for threestudio-3dgs (Step 1 - GATING).

Provides outward-looking cameras positioned inside a room, for interior scene
rendering. This is the inverse topology of the default object-centric cameras.

See design/D-2c_threestudio_interior_camera.md for full specification.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Tuple, List

import threestudio
import torch
import torch.nn.functional as F
from threestudio.data.uncond import (
    RandomCameraDataModule,
    RandomCameraDataModuleConfig,
    RandomCameraIterableDataset,
    RandomCameraDataset,
)
from threestudio.utils.ops import (
    get_rays,
    get_projection_matrix,
    get_mvp_matrix,
)


@dataclass
class InteriorCameraDataModuleConfig:
    # Inherit all fields from RandomCameraDataModuleConfig by repetition
    # OmegaConf doesn't handle dataclass inheritance well
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-60, 60)  # Modified for interior
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)  # Not used but required
    fovy_range: Tuple[float, float] = (40, 70)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0
    rays_d_normalize: bool = True

    # Interior camera specific fields
    camera_position: tuple[float, float, float] = (0.0, 0.0, 1.6)
    position_jitter: float = 0.0
    eval_camera_position: tuple[float, float, float] = (0.0, 0.0, 1.6)


class InteriorCameraIterableDataset(RandomCameraIterableDataset):
    """Training iterable with fixed interior position, outward-looking directions."""

    def collate(self, batch):
        # Call parent to get elevation/azimuth/FOV sampling and basic pose
        out = super().collate(batch)

        batch_size = out["c2w"].shape[0]

        # OVERRIDE: Fixed interior position (NOT orbiting around origin)
        pos = torch.as_tensor(self.cfg.camera_position, dtype=torch.float32, device=out["c2w"].device)
        camera_positions = pos.expand(batch_size, -1).clone()
        if self.cfg.position_jitter > 0:
            camera_positions = camera_positions + (
                torch.randn_like(camera_positions) * self.cfg.position_jitter
            )

        # OVERRIDE: Look direction is OUTWARD from camera (NOT toward origin)
        # Reuse elevation/azimuth from parent but interpret as outward direction
        elev = out["elevation"] * math.pi / 180
        az = out["azimuth"] * math.pi / 180
        lookat = torch.stack([
            torch.cos(elev) * torch.cos(az),
            torch.cos(elev) * torch.sin(az),
            torch.sin(elev),
        ], dim=-1)  # already unit length, pointing outward

        # Compute camera basis: up is +Z (threestudio convention)
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=out["c2w"].device)[None].expand(batch_size, 3)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)

        # Build c2w matrix
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        # Rebuild rays from new c2w
        focal_length = 0.5 * self.height / torch.tan(0.5 * out["fovy"])
        directions = self.directions_unit_focal.to(c2w)[None, :, :, :].repeat(
            batch_size, 1, 1, 1
        )
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Transform to world space
        directions = (
            directions[..., 0, None] * right[:, None, None, :]
            + directions[..., 1, None] * up[:, None, None, :]
            - directions[..., 2, None] * lookat[:, None, None, :]
        )

        rays_o = camera_positions[:, None, None, :].repeat(1, self.height, self.width, 1)
        rays_d = F.normalize(directions, dim=-1)

        # Recompute projection and MVP matrices
        proj_mtx = get_projection_matrix(out["fovy"], self.width / self.height, 0.01, 100.0)
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)

        # Update batch with interior camera pose
        out.update({
            "c2w": c2w,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "proj_mtx": proj_mtx,
            "camera_positions": camera_positions,
        })

        return out


class InteriorCameraDataset(RandomCameraDataset):
    """Eval/test dataset with fixed interior position, outward-looking directions."""

    def __init__(self, cfg, split):
        # Temporarily override config to use interior camera position
        # We'll fix the pose after parent construction
        super().__init__(cfg, split)
        # self.c2w, self.camera_positions, self.rays_o, self.rays_d, etc.
        # are already set by parent — now we override them

        batch_size = self.c2w.shape[0]
        device = self.c2w.device

        # OVERRIDE: Fixed interior position
        pos = torch.as_tensor(cfg.camera_position, dtype=torch.float32, device=device)
        camera_positions = pos.expand(batch_size, -1).clone()
        if cfg.position_jitter > 0:
            camera_positions = camera_positions + (
                torch.randn_like(camera_positions) * cfg.position_jitter
            )

        # OVERRIDE: Outward look direction from elevation/azimuth
        lookat = torch.stack([
            torch.cos(self.elevation) * torch.cos(self.azimuth),
            torch.cos(self.elevation) * torch.sin(self.azimuth),
            torch.sin(self.elevation),
        ], dim=-1)

        up = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=device)[None].expand(batch_size, 3)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)

        # Rebuild c2w
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        # Rebuild rays
        focal_length = 0.5 * cfg.eval_height / torch.tan(0.5 * self.fovy)
        directions = self.directions_unit_focal[None, :, :, :].repeat(batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        directions = (
            directions[..., 0, None] * right[:, None, None, :]
            + directions[..., 1, None] * up[:, None, None, :]
            - directions[..., 2, None] * lookat[:, None, None, :]
        )

        rays_o = camera_positions[:, None, None, :].repeat(1, cfg.eval_height, cfg.eval_width, 1)
        rays_d = F.normalize(directions, dim=-1)

        # Update all pose-dependent fields
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.mvp_mtx = get_mvp_matrix(c2w, self.proj_mtx)


@threestudio.register("interior-camera-datamodule")
class InteriorCameraDataModule:
    cfg: InteriorCameraDataModuleConfig

    def __init__(self, cfg=None):
        self.cfg = threestudio.utils.config.parse_structured(InteriorCameraDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = InteriorCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = InteriorCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = InteriorCameraDataset(self.cfg, "test")
