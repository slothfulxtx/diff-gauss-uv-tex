#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
  class Rasterizer
  {
  public:

    static void markVisible(
      int P,
      float* means3D,
      float* viewmatrix,
      float* projmatrix,
      bool* present);

    static int forward(
      std::function<char* (size_t)> geometryBuffer,
      std::function<char* (size_t)> binningBuffer,
      std::function<char* (size_t)> imageBuffer,
      const int P,
      const int D, 
      const int M, 
      const int ED,
      const int TR,
      const float* background,
      const int width,
      const int height,
      const float* means3D,
      const float* opacities,
      const float* scales,
      const float scale_modifier,
      const float* rotations,
      const float* uvs,
      const float* gradient_uvs,
      const float* texture,
      const float* extra_attrs,
      const float* viewmatrix,
      const float* viewmatrix_inv,
      const float* projmatrix,
      const float* cam_pos,
      const float tan_fovx,
      const float tan_fovy,
      const bool prefiltered,
      float* out_color,
      float* out_depth,
      float* out_norm,
      float* out_alpha,
      float* out_extra,
      int* radii,
      bool debug = false);

    static void backward(
      const int P,
      const int D, 
      const int M, 
      const int R,
      const int ED,
      const int TR,
      const float* background,
      const int width,
      const int height,
      const float* means3D,
      const float* scales,
      const float scale_modifier,
      const float* rotations,
      const float* uvs,
      const float* gradient_uvs,
      const float* texture,
      const float* extra_attrs,
      const float* viewmatrix,
      const float* viewmatrix_inv,
      const float* projmatrix,
      const float* campos,
      const float tan_fovx,
      const float tan_fovy,
      const int* radii,
      char* geom_buffer,
      char* binning_buffer,
      char* image_buffer,
      const float* accum_alphas,
      const float* dL_dpix,
      const float* dL_dpix_depth,
      const float* dL_dpix_norm,
      const float* dL_dpix_dalpha,
      const float* dL_dpix_dextra,
      float* dL_dmean2D,
      float* dL_dconic,
      float* dL_dopacity,
      float* dL_ddepth,
      float* dL_dmean3D,
      float* dL_dcov3D,
      float* dL_dnorm3D,
      float* dL_dscale,
      float* dL_drot,
      float* dL_duvs,
      float* dL_dtexture,
      float* dL_dextra,
      bool debug);
  };
};

#endif