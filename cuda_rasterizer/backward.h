#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
  void render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    const int W,
    const int H, 
    const int ED,
    const int TR,
    const float* bg_color,
    const glm::vec3* campos,
    const float2* means2D,
    const float3* means3D,
    const float3* rgbs,
    const float4* conic_opacity,
    const float tan_fovx,
    const float tan_fovy,
    const float* viewmatrix,
    const float* viewmatrix_inv,
    const float* depths,
    const float* norms,
    const float* uvs,
    const float* gradient_uvs,
    const float* texture,
    const float* extras,
    const float* accum_alphas,
    const uint32_t* n_contrib,
    const float* dL_dpixels,
    const float* dL_dpixel_depths,
    const float* dL_dpixel_norms,
    const float* dL_dpixel_alphas,
    const float* dL_dpixel_extras,
    float3* dL_dmean2D,
    float4* dL_dconic2D,
    float* dL_dopacity,
    float3* dL_drgbs,
    float* dL_duvs,
    float* dL_dtexture,
    float* dL_ddepths,
    float* dL_dnorm3Ds,
    float* dL_dextras);

  void preprocess(
    const int P,
    const int D,
    const int M,
    const float3* means,
    const int* radii,
    const float* shs,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const glm::vec3* norm3Ds,
    const float* view,
    const float* proj,
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    const glm::vec3* campos,
    const float3* dL_dmean2D,
    const float3* dL_drgbs,
    const float* dL_dconics,
    glm::vec3* dL_dmeans,
    float* dL_dshs,
    float* dL_ddepth,
    float* dL_dcov3D,
    glm::vec3* dL_dnorm3D,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot);
}

#endif