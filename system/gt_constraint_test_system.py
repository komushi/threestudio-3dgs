"""
GT Constraint Test System for Step 1.5 (Approach D).

Minimal system that renders 3DGS and samples GT constraint maps at the same
camera poses, saving 4-panel mosaics for visual alignment verification.
"""

import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *
from pathlib import Path
import numpy as np

from ..data.gt_constraint_sampler import GTConstraintSampler


@threestudio.register("gt-constraint-test-system")
class GTConstraintTestSystem(BaseLift3DSystem):
    """Minimal system: render, sample GT, save mosaic. No SDS, no prompt, no guidance."""

    def configure(self):
        super().configure()
        # Initialize GT constraint sampler
        self.sampler = GTConstraintSampler(
            self.cfg.gt_dir, device=self._device
        )

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Render 3DGS and sample GT at the same pose, save mosaic with all 4 GT signals."""
        # Render 3DGS
        render_out = self.renderer(batch)
        rgb = render_out["comp_rgb"]  # (B, H, W, 3)

        # Sample GT constraint maps
        gt_depth = self.sampler.sample_at(batch["rays_d"], "depth")  # (B, 1, H, W)
        gt_normal = self.sampler.sample_at(batch["rays_d"], "normal")  # (B, 3, H, W)
        gt_semantic = self.sampler.sample_at(
            batch["rays_d"], "semantic"
        )  # (B, 3, H, W)
        gt_edge = self.sampler.sample_at(batch["rays_d"], "edge")  # (B, 1, H, W)

        # Save mosaic
        self.save_mosaic(batch_idx, rgb, gt_depth, gt_normal, gt_semantic, gt_edge)

        return {}

    def save_mosaic(
        self,
        batch_idx: int,
        rgb: torch.Tensor,
        gt_depth: torch.Tensor,
        gt_normal: torch.Tensor,
        gt_semantic: torch.Tensor,
        gt_edge: torch.Tensor,
    ):
        """Save mosaic with all 5 panels: RGB render + 4 GT constraint signals.

        Layout:
        +-------------+-------------+-------------+
        |  3DGS RGB   |  GT Depth   | GT Normal   |
        |             |  (colormap) |  (as RGB)   |
        +-------------+-------------+-------------+
        | GT Semantic |  GT Edge    |             |
        |   (as RGB)  | (binary)    |             |
        +-------------+-------------+-------------+
        """
        B = rgb.shape[0]

        for b in range(B):
            # Get image data (H, W, C) format
            rgb_img = (rgb[b].detach().cpu().numpy() * 255).astype(np.uint8)

            # Depth: apply colormap
            depth_val = gt_depth[b, 0].detach().cpu().numpy()  # (H, W)
            depth_img = self._depth_to_colormap(depth_val)

            # Normal: already in [-1, 1], convert to [0, 255] RGB
            normal_img = (
                ((gt_normal[b].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255)
                .astype(np.uint8)
            )

            # Semantic: already in [0, 1], convert to [0, 255] RGB
            semantic_img = (
                gt_semantic[b].permute(1, 2, 0).detach().cpu().numpy() * 255
            ).astype(np.uint8)

            # Edge: binary, convert to 3-channel for display
            edge_val = gt_edge[b, 0].detach().cpu().numpy()  # (H, W)
            edge_img = self._edge_to_display(edge_val)

            # Build 2x3 mosaic (3 columns x 2 rows)
            H, W = rgb_img.shape[:2]

            top_row = np.concatenate([rgb_img, depth_img, normal_img], axis=1)  # (H, 3W, 3)
            bottom_row = np.concatenate([semantic_img, edge_img, edge_img], axis=1)  # (H, 3W, 3)
            mosaic = np.concatenate([top_row, bottom_row], axis=0)  # (2H, 3W, 3)

            # Save
            save_dir = Path(self.get_save_path("it0-test"))
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"mosaic_{batch_idx:03d}.png"
            from PIL import Image

            Image.fromarray(mosaic).save(save_path)
            threestudio.info(f"Saved mosaic: {save_path}")

    @staticmethod
    def _depth_to_colormap(depth: np.ndarray) -> np.ndarray:
        """Convert depth values to RGB colormap.

        Uses a simple blue (near) to red (far) gradient.
        """
        H, W = depth.shape

        # Normalize to [0, 1] for colormap (ignore NaN/inf)
        depth_valid = np.clip(depth, 0, 10) / 10.0  # Assume max 10m range
        depth_valid = np.nan_to_num(depth_valid, nan=0.0, posinf=1.0, neginf=0.0)

        # Simple blue-to-red gradient
        r = depth_valid
        g = 1.0 - 2.0 * np.abs(depth_valid - 0.5)
        b = 1.0 - depth_valid

        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

        return rgb

    @staticmethod
    def _edge_to_display(edge: np.ndarray) -> np.ndarray:
        """Convert edge map to 3-channel display format.

        Edge is binary (0 or 1), convert to white lines on black background.
        """
        H, W = edge.shape

        # Invert: edges are 1 (white in source), make them white on black
        # For better visibility, use white (255) for edges, black (0) for non-edges
        edge_display = (edge * 255).astype(np.uint8)  # (H, W), 0 or 255

        # Expand to 3 channels
        edge_img = np.stack([edge_display, edge_display, edge_display], axis=-1)

        return edge_img
