#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
      return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
  const torch::Tensor& shs,
  const torch::Tensor& opacity,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& uvs,
  const torch::Tensor& gradient_uvs,
  const torch::Tensor& texture,
  const torch::Tensor& extra_attrs,
  const int attr_degree,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& viewmatrix_inv,
  const torch::Tensor& projmatrix,
  const float tan_fovx, 
  const float tan_fovy,
  const int image_height,
  const int image_width,
  const int degree,
  const torch::Tensor& campos,
  const bool prefiltered,
  const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int TR = texture.size(1);
  const int F = attr_degree;
  const int M = shs.size(0) == 0 ? 0 : shs.size(1);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_norm = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_extra;
  if(F > 0)
    out_extra = torch::full({F, H, W}, 0.0, float_opts);
  else
    out_extra = torch::empty({0}, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {

    rendered = CudaRasterizer::Rasterizer::forward(
      geomFunc,
      binningFunc,
      imgFunc,
      P, degree, M, F, TR,
      background.contiguous().data<float>(),
      W, H,
      means3D.contiguous().data<float>(),
      shs.contiguous().data_ptr<float>(),
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      uvs.contiguous().data<float>(),
      gradient_uvs.contiguous().data<float>(),
      texture.contiguous().data<float>(), 
      extra_attrs.contiguous().data<float>(), 
      viewmatrix.contiguous().data<float>(),  
      viewmatrix_inv.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_color.contiguous().data<float>(),
      out_depth.contiguous().data<float>(),
      out_norm.contiguous().data<float>(),
      out_alpha.contiguous().data<float>(),
      out_extra.contiguous().data<float>(),
      radii.contiguous().data<int>(),
      debug);
  }
  return std::make_tuple(rendered, out_color, out_depth, out_norm, out_alpha, radii, out_extra, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
  const torch::Tensor& radii,
  const torch::Tensor& shs,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const torch::Tensor& extra_attrs,
  const float scale_modifier,
  const torch::Tensor& uvs,
  const torch::Tensor& gradient_uvs,
  const torch::Tensor& texture,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& viewmatrix_inv,
  const torch::Tensor& projmatrix,
  const float tan_fovx,
  const float tan_fovy,
  const torch::Tensor& dL_dout_color,
  const torch::Tensor& dL_dout_depth,
  const torch::Tensor& dL_dout_norm,
  const torch::Tensor& dL_dout_alpha,
  const torch::Tensor& dL_dout_extra,
  const int degree,
  const torch::Tensor& campos,
  const torch::Tensor& geomBuffer,
  const int R,
  const torch::Tensor& binningBuffer,
  const torch::Tensor& imageBuffer,
  const torch::Tensor& out_alpha,
  const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  const int F = (extra_attrs.size(0) != 0 ? extra_attrs.size(1) : 0);
  const int TR = texture.size(1);
  const int M = shs.size(0) == 0 ? 0 : shs.size(1);

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dshs = torch::zeros({P, M, 3}, means3D.options());
  // just for storing intermediate results
  torch::Tensor dL_drgbs = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dnorm3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_duvs = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dtexture = torch::zeros({6, TR, TR, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dextra_attrs;
  if(F > 0)
    dL_dextra_attrs = torch::zeros({P, F}, means3D.options());
  else
    dL_dextra_attrs = torch::empty({0}, means3D.options());
  if(P != 0)
  {  
    CudaRasterizer::Rasterizer::backward(
      P, degree, M, R, F, TR,
      background.contiguous().data<float>(),
      W, H, 
      means3D.contiguous().data<float>(),
      shs.contiguous().data<float>(),
      scales.data_ptr<float>(),
      scale_modifier,
      rotations.data_ptr<float>(),
      uvs.contiguous().data<float>(),
      gradient_uvs.contiguous().data<float>(),
      texture.contiguous().data<float>(),
      extra_attrs.contiguous().data<float>(),
      viewmatrix.contiguous().data<float>(),
      viewmatrix_inv.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      radii.contiguous().data<int>(),
      reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
      out_alpha.contiguous().data<float>(),
      dL_dout_color.contiguous().data<float>(),
      dL_dout_depth.contiguous().data<float>(),
      dL_dout_norm.contiguous().data<float>(),
      dL_dout_alpha.contiguous().data<float>(),
      dL_dout_extra.contiguous().data<float>(),
      dL_dmeans2D.contiguous().data<float>(),
      dL_dconic.contiguous().data<float>(), 
      dL_dopacity.contiguous().data<float>(),
      dL_ddepths.contiguous().data<float>(),
      dL_dmeans3D.contiguous().data<float>(),
      dL_dcov3D.contiguous().data<float>(),
      dL_dnorm3D.contiguous().data<float>(),
      dL_dscales.contiguous().data<float>(),
      dL_drotations.contiguous().data<float>(),
      dL_drgbs.contiguous().data<float>(),
      dL_dshs.contiguous().data<float>(),
      dL_duvs.contiguous().data<float>(),
      dL_dtexture.contiguous().data<float>(),
      dL_dextra_attrs.contiguous().data<float>(),
      debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dopacity, dL_dmeans3D, dL_dshs, dL_dscales, dL_drotations, dL_duvs, dL_dtexture, dL_dextra_attrs);
}

torch::Tensor markVisible(
  torch::Tensor& means3D,
  torch::Tensor& viewmatrix,
  torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
  CudaRasterizer::Rasterizer::markVisible(
    P,
    means3D.contiguous().data<float>(),
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    present.contiguous().data<bool>());
  }
  
  return present;
}