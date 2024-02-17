#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
  // Perform initial steps for each Gaussian prior to rasterization.
  void preprocess(
    const int P,
    const int D,
    const int M,
    const float* orig_points,
    const float* shs,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W,
    const int H,
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    int* radii,
    float2* points_xy_image,
    float3* rgbs,
    float* depths,
    float* clamp_radii,
    float* cov3Ds,
    float* norm3Ds,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered);

  // Main rasterization method.
  void render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    const int W,
    const int H,
    const int ED,
    const int TR,
    const float2* points_xy_image,
    const float* orig_points,
    const float3* rgbs,
    const float* norms,
    const float* depths,
    const float* clamp_radii,
    const float* uvs,
    const float* gradient_uvs,
    const float* texture,
    const float* extras,
    const float4* conic_opacity,
    const float tan_fovx,
    const float tan_fovy,
    const float* viewmatrix,
    const float* viewmatrix_inv,
    float* out_alpha,
    uint32_t* n_contrib,
    const float* bg_color,
    const glm::vec3* cam_pos,
    float* out_color,
    float* out_depth,
    float* out_norm,
    float* out_extra);
}


#endif