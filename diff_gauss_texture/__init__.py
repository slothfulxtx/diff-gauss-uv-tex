from typing import NamedTuple
import torch.nn as nn
import torch
import math
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    opacities,
    scales,
    rotations,
    uvs,
    gradient_uvs, # duv/dxy
    texture,
    extra_attrs,
    raster_settings,
):
    color, depth, norm, alpha, radii, extra = _RasterizeGaussians.apply(
        means3D,
        means2D,
        opacities,
        scales,
        rotations,
        uvs,
        gradient_uvs,
        texture,
        extra_attrs,
        raster_settings,
    )
    
    norm = torch.nn.functional.normalize(norm, p=2, dim=0)
    # 3, H, W
    return color, depth, norm, alpha, radii, extra

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        opacities,
        scales,
        rotations,
        uvs,
        gradient_uvs, # duv/dxyz
        texture,
        extra_attrs,
        raster_settings,
    ):
        num_gaussians = means3D.shape[0]
        assert means3D.shape == (num_gaussians, 3)
        assert opacities.shape == (num_gaussians, 1)
        assert scales.shape == (num_gaussians, 3)
        assert rotations.shape == (num_gaussians, 4)
        assert uvs.shape == (num_gaussians, 3)
        assert gradient_uvs.shape == (num_gaussians, 3*3)
        tex_res = texture.shape[1]
        assert texture.shape == (6, tex_res, tex_res, 3)
        
        # restrict the length of extra attr values to avoid dynamically sized shared memory allocation
        assert extra_attrs.shape[0] == 0 or extra_attrs.shape[1] <= 34
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            uvs,
            gradient_uvs,
            texture,
            extra_attrs,
            extra_attrs.shape[1] if extra_attrs.shape[0] != 0 else 0,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, norm, alpha, radii, extra, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, norm, alpha, radii, extra, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, uvs, gradient_uvs, texture, radii, extra_attrs, geomBuffer, binningBuffer, imgBuffer, alpha)
        return color, depth, norm, alpha, radii, extra

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_norm, grad_out_alpha, _, grad_out_extra):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        means3D, scales, rotations, uvs, gradient_uvs, texture, radii, extra_attrs, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                scales, 
                rotations, 
                extra_attrs,
                raster_settings.scale_modifier, 
                uvs,
                gradient_uvs,
                texture,
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_depth,
                grad_out_norm,
                grad_out_alpha,
                grad_out_extra,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_opacities, grad_means3D, grad_scales, grad_rotations, grad_uvs, grad_texture, grad_extra_attrs = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_opacities, grad_means3D, grad_scales, grad_rotations, grad_uvs, grad_texture, grad_extra_attrs = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_uvs,
            None,
            grad_texture,
            grad_extra_attrs,
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, scales, rotations, uvs, gradient_uvs, texture, extra_attrs=None):
        
        raster_settings = self.raster_settings
        
        if extra_attrs is None:
            extra_attrs = torch.Tensor([])
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            opacities,
            scales, 
            rotations,
            uvs,
            gradient_uvs,
            texture,
            extra_attrs,
            raster_settings, 
        )

