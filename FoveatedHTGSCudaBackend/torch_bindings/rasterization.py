from typing import NamedTuple

import torch

from FoveatedHTGSCudaBackend import _C, _C_benchmarking


class RasterizerSettings(NamedTuple):
    M: torch.Tensor  # affine transformation from model/world space to camera/view space
    VPM: torch.Tensor  # homogeneous transformation from model/world space to screen space
    cam_position: torch.Tensor  # camera position in world space
    gaze_position: torch.Tensor  # gaze position in screen space
    K: int  # only used for HYBRID_BLEND and ALPHA_BLEND_FIRST_K
    active_sh_bases: int  # number of spherical harmonics bases to use for color computation
    width: int
    height: int
    near_plane: float
    far_plane: float
    scale_modifier: float  # scaling factor to be applied to each Gaussian

    def as_tuple(self) -> tuple:
        return (
            self.M,
            self.VPM,
            self.cam_position,
            self.gaze_position,
            self.K,
            self.active_sh_bases,
            self.width,
            self.height,
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
        )
        return image
