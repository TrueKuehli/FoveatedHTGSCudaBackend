from enum import Enum
from typing import NamedTuple

import torch

from FoveatedHTGSCudaBackend import _C, _C_benchmarking


class PeripheryInterpolationMode(Enum):
    NEAREST = 0
    LINEAR = 1


class RasterizerSettings(NamedTuple):
    M: torch.Tensor  # affine transformation from model/world space to camera/view space
    VPM: torch.Tensor  # homogeneous transformation from model/world space to screen space
    cam_position: torch.Tensor  # camera position in world space
    gaze_position: torch.Tensor  # gaze position in screen space
    render_mask: torch.Tensor  # precomputed mask for culling invisible tiles
    render_mask_area_table: torch.Tensor  # precomputed table for culling invisible tiles
    fovea_mask_area_table: torch.Tensor  # precomputed table for the shape of the sharp foveated area
    periphery_interpolation_mode: PeripheryInterpolationMode
    K: int  # size of the core for hybrid transparency
    active_sh_bases: int  # number of spherical harmonics bases to use for color computation
    width: int
    height: int
    focal_x: float
    focal_y: float
    near_plane: float
    far_plane: float
    scale_modifier: float  # scaling factor to be applied to each Gaussian

    def as_tuple(self) -> tuple:
        return (
            self.M,
            self.VPM,
            self.cam_position,
            self.gaze_position,
            self.render_mask,
            self.render_mask_area_table,
            self.fovea_mask_area_table,
            self.periphery_interpolation_mode.value,
            self.K,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.near_plane,
            self.far_plane,
            self.scale_modifier,
        )


class FoveatedHTGSRasterizer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def render(
            self,
            positions: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            opacities: torch.Tensor,
            sh_0: torch.Tensor,
            sh_rest: torch.Tensor,
            settings: RasterizerSettings,
            to_chw: bool,
            use_median_depth: bool,
            blur_periphery: bool,
            anti_aliasing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, depth = _C.render(
            positions,
            scales,
            rotations,
            opacities,
            sh_0,
            sh_rest,
            *settings.as_tuple(),
            to_chw,
            use_median_depth,
            blur_periphery,
            anti_aliasing,
        )
        depth = depth.unsqueeze(0) if to_chw else depth.unsqueeze(-1)
        return image, depth

    def benchmark(
            self,
            positions: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            opacities: torch.Tensor,
            sh_0: torch.Tensor,
            sh_rest: torch.Tensor,
            settings: RasterizerSettings,
            to_chw: bool,
            blur_periphery: bool,
            anti_aliasing: bool,
    ) -> torch.Tensor:
        image = _C_benchmarking.benchmark(
            positions,
            scales,
            rotations,
            opacities,
            sh_0,
            sh_rest,
            *settings.as_tuple(),
            to_chw,
            blur_periphery,
            anti_aliasing,
        )
        return image
