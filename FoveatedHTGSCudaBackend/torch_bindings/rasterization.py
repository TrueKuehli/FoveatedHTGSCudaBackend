from enum import Enum
from typing import NamedTuple

import torch

from FoveatedHTGSCudaBackend import _C, _C_benchmarking, _C_stereo, _C_visualization


class BackgroundModel(Enum):
    NONE = 0
    SH = 1
    TEXTURE = 2


class GazeVisualizationType(Enum):
    CIRCLE = 0
    SQUARE = 1
    CROSS = 2


class RasterizerSettings(NamedTuple):
    M: torch.Tensor  # affine transformation from model/world space to camera/view space
    VPM: torch.Tensor  # homogeneous transformation from model/world space to screen space
    VPR_inv: torch.Tensor  # homogeneous transformation from screen space to model/world space, ignoring translation
    cam_position: torch.Tensor  # camera position in world space
    gaze_position: torch.Tensor  # gaze position in screen space
    visibility_mask: torch.Tensor  # precomputed mask for culling invisible tiles
    visibility_mask_area_table: torch.Tensor  # precomputed table for culling invisible tiles
    fovea_mask_area_table: torch.Tensor  # precomputed table for the shape of the sharp foveated area
    background_model_data: torch.Tensor  # background model specific data
    background_model: BackgroundModel
    K: int  # size of the core for hybrid transparency
    active_sh_bases: int  # number of spherical harmonics bases to use for color computation
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    near_plane: float
    far_plane: float
    scale_modifier: float  # scaling factor to be applied to each Gaussian

    def as_tuple(self) -> tuple:
        return (
            self.M,
            self.VPM,
            self.VPR_inv,
            self.cam_position,
            self.gaze_position,
            self.visibility_mask,
            self.visibility_mask_area_table,
            self.fovea_mask_area_table,
            self.background_model_data,
            self.background_model.value,
            self.K,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.scale_modifier,
        )


class FoveatedHTGSRasterizer(torch.nn.Module):

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
            blur_periphery: bool,
            aaa_mode: bool,
    ) -> torch.Tensor:
        image = _C.render(
            positions,
            scales,
            rotations,
            opacities,
            sh_0,
            sh_rest,
            *settings.as_tuple(),
            to_chw,
            blur_periphery,
            aaa_mode,
        )
        return image

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
            aaa_mode: bool,
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
            aaa_mode,
        )
        return image
    
    def render_stereo(
            self,
            positions: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            opacities: torch.Tensor,
            sh_0: torch.Tensor,
            sh_rest: torch.Tensor,
            settings_left: RasterizerSettings,
            settings_right: RasterizerSettings,
            to_chw: bool,
            blur_periphery: bool,
            aaa_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_left, image_right = _C_stereo.render(
            positions,
            scales,
            rotations,
            opacities,
            sh_0,
            sh_rest,
            *settings_left.as_tuple(),
            *settings_right.as_tuple(),
            to_chw,
            blur_periphery,
            aaa_mode,
        )
        return image_left, image_right


class FoveatedHTGSVisualizer(torch.nn.Module):

    def visualize_gaze(
            self,
            image: torch.Tensor,
            gaze_position: torch.Tensor,
            gaze_color: torch.Tensor,
            visualization_size: int,
            visualization_type: GazeVisualizationType,
            to_chw: bool,
    ) -> torch.Tensor:
        image = _C_visualization.visualize_gaze(
            image,
            gaze_position,
            gaze_color,
            visualization_type.value,
            visualization_size,
            to_chw,
        )
        return image

    def visualize_tile_boundaries(
            self,
            image: torch.Tensor,
            visibility_mask: torch.Tensor,
            gaze_position: torch.Tensor,
            to_chw: bool,
    ) -> torch.Tensor:
        image = _C_visualization.visualize_tile_boundaries(
            image,
            visibility_mask,
            gaze_position,
            to_chw,
        )
        return image
