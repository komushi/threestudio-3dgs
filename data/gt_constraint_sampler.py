"""
GT Constraint Sampler for Step 1.5 (Approach D).

Loads equirectangular constraint maps from D-0a output and samples them at
arbitrary outbound camera directions. Used for geometric-alignment testing
between Blender GT and 3DGS renders.

Uses Y-up coordinate system to match interior_camera.py (commit 2ff886c).
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional


class GTConstraintSampler:
    """Load the 4 equirect constraint maps from a D-0a output directory and
    sample them at arbitrary outbound camera directions (Y-up convention)."""

    def __init__(self, gt_dir: str, device: str = "cuda"):
        """
        Args:
            gt_dir: Path to D-0a output directory containing:
                - depth_equirect_metric.png (16-bit linear depth)
                - normal_equirect.png (normal map as RGB)
                - semantic_equirect.png (ADE20K semantic map)
                - edge_equirect.png (binary edge map)
                - position.json (camera metadata)
            device: Device to store tensors on
        """
        gt_dir = Path(gt_dir)

        # Load all four signals as (C, H, W) tensors.
        # Blender's file_slots output appends a frame index (e.g., "0000") to
        # depth/normal filenames, so resolve via glob.
        depth_path = self._resolve(gt_dir, "depth_equirect_metric")
        # Scale depends on format:
        #   .tif  → uint16 [0, 65535] → 10.0 / 65535.0
        #   .exr  → float  [0, 1]     → 10.0  (legacy; requires cv2 EXR support)
        #   .png  → uint8  [0, 255]   → 10.0 / 255.0  (legacy 8-bit)
        _sfx = depth_path.suffix.lower()
        if _sfx in (".tif", ".tiff"):
            depth_scale = 10.0 / 65535.0
        elif _sfx == ".exr":
            depth_scale = 10.0
        else:
            depth_scale = 10.0 / 255.0
        self.depth = self._load_equirect(
            depth_path,
            channels=1,
            dtype=torch.float32,
            scale=depth_scale,
        )
        self.normal = self._load_equirect(
            self._resolve(gt_dir, "normal_equirect"),
            channels=3,
            dtype=torch.float32,
            scale=1.0 / 127.5,
            shift=-1.0,  # [0, 255] -> [-1, 1]
        )
        self.semantic = self._load_equirect(
            self._resolve(gt_dir, "semantic_equirect"),
            channels=3,
            dtype=torch.float32,
            scale=1.0 / 255.0,  # ADE20K RGB
        )
        self.edge = self._load_equirect(
            self._resolve(gt_dir, "edge_equirect"),
            channels=1,
            dtype=torch.float32,
            scale=1.0 / 255.0,  # binary
        )

        # Apply Blender -> threestudio azimuth-origin correction (once, at load).
        # +90 deg in azimuth = roll equirect by width/4
        # This converts from Blender's azimuth origin to threestudio's convention
        for name in ("depth", "normal", "semantic", "edge"):
            t = getattr(self, name)
            setattr(
                self, name, torch.roll(t, shifts=t.shape[-1] // 4, dims=-1).to(device)
            )

        # Store dimensions for validation
        self.H = self.depth.shape[1]
        self.W = self.depth.shape[2]
        self._device = device

    @staticmethod
    def _resolve(gt_dir: Path, stem: str) -> Path:
        # Preference order: .tif (16-bit, no special libs) → .exr (legacy) → .png (8-bit legacy).
        # Also handles Blender's frame-index suffix (e.g., depth_equirect_metric0000.tif).
        for ext in (".tif", ".tiff", ".exr", ".png"):
            exact = gt_dir / f"{stem}{ext}"
            if exact.exists():
                return exact
        for ext in (".tif", ".tiff", ".exr", ".png"):
            matches = sorted(gt_dir.glob(f"{stem}*{ext}"))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"No equirect file matching '{stem}.(tif|exr|png)' in {gt_dir}"
        )

    @staticmethod
    def _load_equirect(
        path: Path, channels: int, dtype: torch.dtype, scale: float = 1.0, shift: float = 0.0
    ) -> torch.Tensor:
        """Load an equirectangular image and convert to tensor.

        Args:
            path: Path to image file (.exr or .png)
            channels: Number of channels (1 for depth/edge, 3 for normal/semantic)
            dtype: Output tensor dtype
            scale: Scale factor applied after loading
            shift: Offset applied after scaling

        Returns:
            Tensor of shape (C, H, W)
        """
        if path.suffix.lower() == ".exr":
            # Use OpenEXR package for EXR loading - handles single-channel EXR correctly
            # Install on supermicro: pip install OpenEXR
            import OpenEXR
            from Imath import ChannelType
            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header
            w, h = header['dataWindow'].max.x + 1, header['dataWindow'].max.y + 1
            # Read the first channel (R for grayscale depth)
            data = exr_file.channel('R', ChannelType.FLOAT)
            img = np.frombuffer(data, dtype=np.float32).reshape(h, w)
        else:
            img = np.asarray(Image.open(path))

        if channels == 1 and img.ndim == 3:
            img = img[..., 0]

        if channels == 1:
            t = torch.from_numpy(img).to(dtype)[None]  # (1, H, W)
        else:
            t = torch.from_numpy(img).to(dtype).permute(2, 0, 1)  # (3, H, W)

        return t * scale + shift

    def sample_at(
        self, rays_d: torch.Tensor, signal: str
    ) -> torch.Tensor:
        """Sample a constraint signal at given ray directions.

        Args:
            rays_d: (B, H, W, 3) unit vectors in threestudio world frame (Y-up)
            signal: One of "depth", "normal", "semantic", "edge"

        Returns:
            (B, C, H, W) sampled values
        """
        t = getattr(self, signal)  # (C, H_eq, W_eq)
        B, H, W, _ = rays_d.shape

        # Equirect (u, v) from direction - Y-up convention
        # Matches interior_camera.py's lookat formula:
        #   lookat = [cos(el)*sin(az), sin(el), cos(el)*cos(az)]
        # Inverse: az = atan2(x, z), el = asin(y)
        theta = torch.atan2(rays_d[..., 0], rays_d[..., 2])  # azimuth in XZ plane
        phi = torch.asin(rays_d[..., 1].clamp(-1.0, 1.0))     # elevation from Y axis

        # Normalize to [0, 1]
        u = theta / (2 * torch.pi) + 0.5  # [0, 1]
        v = 0.5 - phi / torch.pi           # [0, 1]

        # grid_sample expects grid in [-1, 1]
        grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1)  # (B, H, W, 2)

        # Expand tensor for batch processing and move to same device as grid
        eq = t[None].expand(B, -1, -1, -1).to(grid.device)  # (B, C, H_eq, W_eq)

        sampled = F.grid_sample(
            eq, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )

        return sampled  # (B, C, H, W)

    def sample_all(
        self, rays_d: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Sample all constraint signals at given ray directions.

        Args:
            rays_d: (B, H, W, 3) unit vectors in threestudio world frame (Y-up)

        Returns:
            Dict with keys: "depth", "normal", "semantic", "edge"
        """
        return {
            "depth": self.sample_at(rays_d, "depth"),
            "normal": self.sample_at(rays_d, "normal"),
            "semantic": self.sample_at(rays_d, "semantic"),
            "edge": self.sample_at(rays_d, "edge"),
        }
