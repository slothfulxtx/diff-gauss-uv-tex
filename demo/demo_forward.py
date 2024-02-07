import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
from diff_gauss_texture import GaussianRasterizationSettings, GaussianRasterizer


class Gaussian3D:

    def setup_functions(self):        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = F.normalize

    def __init__(self):
        self.create_single_gaussian()
        self.setup_functions()

    def create_single_gaussian(self):
        fused_point_cloud = torch.tensor(np.array([[-0.3414, -1.3538, -0.2377]])).float().cuda()
        texture = torch.tensor([
            [1, 0, 0.5],
            [0, 0.5, 0.5],
            [0, 0, 1.0],
            [1, 0, 1.0],
            [1.0, 1.0, 0],
            [1.0, 1.0, 1.0],
        ]).reshape(6,1,1,3).repeat(1, 2, 2, 4).float().cuda()

        scales = torch.tensor([[0.0, -10.0, 0.0]]).float().cuda()
        rots = torch.tensor([[1.4030, 0.0968, 0.0413, 0.0354]]).float().cuda()

        opacities = torch.tensor([[11.8296]]).float().cuda()
        uvs = torch.tensor([[0.0, 0.0, -1.0]]).float().cuda()
        gradient_uvs = torch.zeros((1, 9)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._textures = nn.Parameter(texture.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._uvs = nn.Parameter(uvs.requires_grad_(True))
        self._grad_uvs = nn.Parameter(gradient_uvs.requires_grad_(True))
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_textures(self):
        return self._textures
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_uv(self):
        return self._uvs
    
    @property
    def get_grad_uv(self):
        return self._grad_uvs

class Camera:
    def __init__(self):
        self.image_height = 800
        self.image_width = 800
        self.world_view_transform = torch.tensor(
            [[-9.9990e-01, -4.1922e-03,  1.3346e-02,  0.0000e+00],
            [-1.3989e-02,  2.9966e-01, -9.5394e-01,  0.0000e+00],
            [-1.1926e-10, -9.5404e-01, -2.9969e-01,  0.0000e+00],
            [ 6.3307e-10, -5.1536e-08,  4.0311e+00,  1.0000e+00]]).float().cuda()
        self.full_proj_transform = torch.tensor(
            [[-3.1247e+00, -1.3101e-02,  1.3347e-02,  1.3346e-02],
            [-4.3715e-02,  9.3643e-01, -9.5404e-01, -9.5394e-01],
            [-3.7270e-10, -2.9814e+00, -2.9972e-01, -2.9969e-01],
            [ 1.9783e-09, -1.6105e-07,  4.0215e+00,  4.0311e+00]]).float().cuda()
        self.camera_center = torch.tensor([-0.0538,  3.8455,  1.2081]).float().cuda()

        self.FoVx = self.FoVy = 0.6194058656692505

def render(viewpoint_camera: Camera, gaussians: Gaussian3D, bg_color, scaling_modifier = 1.0, extra_attrs=None, debug=False):
    
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity

    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation

    textures = gaussians.get_textures
    uvs = gaussians.get_uv
    grad_uvs = gaussians.get_grad_uv
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        uvs = uvs,
        gradient_uvs=grad_uvs,
        texture=textures,
        extra_attrs=extra_attrs)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth,
            "norm": rendered_norm,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "extra": extra,
            "radii": radii}

if __name__ == '__main__':
    gaussians = Gaussian3D()
    viewpoint = Camera()
    render_pkg = render(viewpoint, gaussians,  torch.tensor([0,0,0]).float().cuda(), debug=True)
    rgb = render_pkg['render']
    # 3, H, W
    rgb = (rgb.permute(1, 2, 0) * 255).detach().cpu().int().numpy()
    cv2.imwrite('demo_tex.png', rgb[:, :, ::-1])