/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
#define MAX_EXTRA_DIMS 34

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
  1.0925484305920792f,
  -1.0925484305920792f,
  0.31539156525252005f,
  -1.0925484305920792f,
  0.5462742152960396f
};
__device__ const float SH_C3[] = {
  -0.5900435899266435f,
  2.890611442640554f,
  -0.4570457994644658f,
  0.3731763325901154f,
  -0.4570457994644658f,
  1.445305721320277f,
  -0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
  return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float pix2Ndc(float p, int S)
{
  return (p * 2 + 1.0) / S - 1.0;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
  rect_min = {
    min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
    min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
  };
  rect_max = {
    min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
    min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
  };
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
  float3 transformed = {
    matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
    matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
    matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
  };
  return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
  float4 transformed = {
    matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
    matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
    matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
  };
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
  float3 transformed = {
    matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
    matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
    matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
  float3 transformed = {
    matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
    matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
    matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float3 Vec3x3(const float* matrix, const float3& p)
{
  float3 ret = {
    matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
    matrix[3] * p.x + matrix[4] * p.y + matrix[5] * p.z,
    matrix[6] * p.x + matrix[7] * p.y + matrix[8] * p.z,
  };
  return ret;
}



__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
  float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
  return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float3 dnormvdv;
  dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
  dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
  dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
  float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
  float4 dnormvdv;
  dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
  dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
  dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
  dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
  const float* orig_points,
  const float* viewmatrix,
  const float* projmatrix,
  bool prefiltered,
  float3& p_view)
{
  float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

  // Bring points to screen space
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
  p_view = transformPoint4x3(p_orig, viewmatrix);

  if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
  {
    if (prefiltered)
    {
      printf("Point is filtered although prefiltered is set. This shouldn't happen!");
      __trap();
    }
    return false;
  }
  return true;
}


static __device__ __forceinline__ int indexCubeMap(float& x, float& y, float z)
{
  float ax = fabsf(x);
  float ay = fabsf(y);
  float az = fabsf(z);
  int idx;
  float c;
  if (az > fmaxf(ax, ay)) { idx = 4; c = z; }
  else if (ay > ax)       { idx = 2; c = y; y = z; }
  else                    { idx = 0; c = x; x = z; }
  if (c < 0.f) idx += 1;
  float m = __frcp_rz(fabsf(c)) * .5;
  float m0 = __uint_as_float(__float_as_uint(m) ^ ((0x21u >> idx) << 31));
  float m1 = (idx != 2) ? -m : m;
  x = x * m0 + .5;
  y = y * m1 + .5;
  if (!isfinite(x) || !isfinite(y))
    return -1; // Invalid uv.
  x = fminf(fmaxf(x, 0.f), 1.f);
  y = fminf(fmaxf(y, 0.f), 1.f);
  return idx;
}

static __constant__ uint32_t c_cubeWrapMask1[48] =
{
    0x1530a440, 0x1133a550, 0x6103a110, 0x1515aa44, 0x6161aa11, 0x40154a04, 0x44115a05, 0x04611a01,
    0x2630a440, 0x2233a550, 0x5203a110, 0x2626aa44, 0x5252aa11, 0x40264a04, 0x44225a05, 0x04521a01,
    0x32608064, 0x3366a055, 0x13062091, 0x32328866, 0x13132299, 0x50320846, 0x55330a55, 0x05130219,
    0x42508064, 0x4455a055, 0x14052091, 0x42428866, 0x14142299, 0x60420846, 0x66440a55, 0x06140219,
    0x5230a044, 0x5533a055, 0x1503a011, 0x5252aa44, 0x1515aa11, 0x40520a44, 0x44550a55, 0x04150a11,
    0x6130a044, 0x6633a055, 0x2603a011, 0x6161aa44, 0x2626aa11, 0x40610a44, 0x44660a55, 0x04260a11,
};

static __constant__ uint8_t c_cubeWrapMask2[48] =
{
    0x26, 0x33, 0x11, 0x05, 0x00, 0x09, 0x0c, 0x04, 0x04, 0x00, 0x00, 0x05, 0x00, 0x81, 0xc0, 0x40,
    0x02, 0x03, 0x09, 0x00, 0x0a, 0x00, 0x00, 0x02, 0x64, 0x30, 0x90, 0x55, 0xa0, 0x99, 0xcc, 0x64,
    0x24, 0x30, 0x10, 0x05, 0x00, 0x01, 0x00, 0x00, 0x06, 0x03, 0x01, 0x05, 0x00, 0x89, 0xcc, 0x44,
};

static __device__ __forceinline__ int4 wrapCubeMap(int face, int ix0, int ix1, int iy0, int iy1, int w)
{
  // Calculate case number.
  int cx = (ix0 < 0) ? 0 : (ix1 >= w) ? 2 : 1;
  int cy = (iy0 < 0) ? 0 : (iy1 >= w) ? 6 : 3;
  int c = cx + cy;
  if (c >= 5)
      c--;
  c = (face << 3) + c;

  // Compute coordinates and faces.
  unsigned int m = c_cubeWrapMask1[c];
  int x0 = (m >>  0) & 3; x0 = (x0 == 0) ? 0 : (x0 == 1) ? ix0 : iy0;
  int x1 = (m >>  2) & 3; x1 = (x1 == 0) ? 0 : (x1 == 1) ? ix1 : iy0;
  int x2 = (m >>  4) & 3; x2 = (x2 == 0) ? 0 : (x2 == 1) ? ix0 : iy1;
  int x3 = (m >>  6) & 3; x3 = (x3 == 0) ? 0 : (x3 == 1) ? ix1 : iy1;
  int y0 = (m >>  8) & 3; y0 = (y0 == 0) ? 0 : (y0 == 1) ? ix0 : iy0;
  int y1 = (m >> 10) & 3; y1 = (y1 == 0) ? 0 : (y1 == 1) ? ix1 : iy0;
  int y2 = (m >> 12) & 3; y2 = (y2 == 0) ? 0 : (y2 == 1) ? ix0 : iy1;
  int y3 = (m >> 14) & 3; y3 = (y3 == 0) ? 0 : (y3 == 1) ? ix1 : iy1;
  int f0 = ((m >> 16) & 15) - 1;
  int f1 = ((m >> 20) & 15) - 1;
  int f2 = ((m >> 24) & 15) - 1;
  int f3 = ((m >> 28)     ) - 1;

  // Flips.
  unsigned int f = c_cubeWrapMask2[c];
  int w1 = w - 1;
  if (f & 0x01) x0 = w1 - x0;
  if (f & 0x02) x1 = w1 - x1;
  if (f & 0x04) x2 = w1 - x2;
  if (f & 0x08) x3 = w1 - x3;
  if (f & 0x10) y0 = w1 - y0;
  if (f & 0x20) y1 = w1 - y1;
  if (f & 0x40) y2 = w1 - y2;
  if (f & 0x80) y3 = w1 - y3;

  // Done.
  int4 tcOut;
  tcOut.x = x0 + (y0 + f0 * w) * w;
  tcOut.y = x1 + (y1 + f1 * w) * w;
  tcOut.z = x2 + (y2 + f2 * w) * w;
  tcOut.w = x3 + (y3 + f3 * w) * w;
  return tcOut;
}

static __device__ __forceinline__ float2 indexTextureLinear(float3 uv, int4& tcOut, const int TR)
{
  int w = TR, h = TR;

  // Compute texture-space u, v.
  float u = uv.x;
  float v = uv.y;

  // Cube map indexing.
  int face = indexCubeMap(u, v, uv.z); // Rewrites u, v.
  if (face < 0)
  {
      tcOut.x = tcOut.y = tcOut.z = tcOut.w = -1; // Invalid uv.
      return make_float2(0.f, 0.f);
  }
  u = u * (float)w - 0.5f;
  v = v * (float)h - 0.5f;


  // Compute texel coordinates and weights.
  int iu0 = __float2int_rd(u);
  int iv0 = __float2int_rd(v);
  int iu1 = iu0 + 1; // Ensure zero u/v gradients with clamped.
  int iv1 = iv0 + 1;
  u -= (float)iu0;
  v -= (float)iv0;

  // Cube map wrapping.
  bool cubeWrap = (iu0 < 0 || iv0 < 0 || iu1 >= w || iv1 >= h);
  if (cubeWrap)
  {
      tcOut = wrapCubeMap(face, iu0, iu1, iv0, iv1, w);
      return make_float2(u, v); // Done.
  }

  tcOut.x = face * w * h + w * iv0 + iu0;
  tcOut.y = face * w * h + w * iv0 + iu1;
  tcOut.z = face * w * h + w * iv1 + iu0;
  tcOut.w = face * w * h + w * iv1 + iu1;

  // All done.
  return make_float2(u, v);
}

static __device__ __forceinline__ void fetchQuad(float& a00, float& a10, float& a01, float& a11, const float* pIn, int4 tc, bool corner)
{
  // For invalid cube map uv, tc will be all negative, and all texel values will be zero.
  if (corner)
  {
    float avg = 0.0;
    if (tc.x >= 0) avg += (a00 = pIn[tc.x]);
    if (tc.y >= 0) avg += (a10 = pIn[tc.y]);
    if (tc.z >= 0) avg += (a01 = pIn[tc.z]);
    if (tc.w >= 0) avg += (a11 = pIn[tc.w]);
    avg *= 0.33333333f;
    if (tc.x < 0) a00 = avg;
    if (tc.y < 0) a10 = avg;
    if (tc.z < 0) a01 = avg;
    if (tc.w < 0) a11 = avg;
  }
  else
  {
    a00 = (tc.x >= 0) ? pIn[tc.x] : 0.0;
    a10 = (tc.y >= 0) ? pIn[tc.y] : 0.0;
    a01 = (tc.z >= 0) ? pIn[tc.z] : 0.0;
    a11 = (tc.w >= 0) ? pIn[tc.w] : 0.0;
  }
}
template<class T> static __device__ __forceinline__ T lerp  (const T& a, const T& b, float c) { return a + c * (b - a); }
template<class T> static __device__ __forceinline__ T bilerp(const T& a, const T& b, const T& c, const T& d, const float2& e) { return lerp(lerp(a, b, e.x), lerp(c, d, e.x), e.y); }

__forceinline__ __device__ float3 cube_texture_fetch(const float3& uv, const float* texture, const int TR)
{
  int4 tc0 = {0, 0, 0, 0};
  float2 uv0 = indexTextureLinear(uv, tc0, TR);
  bool corner0 = ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
  tc0 = {tc0.x * 3, tc0.y * 3, tc0.z * 3, tc0.w * 3};

  float pOut[3];
  // Interpolate.
  for (int i=0; i < 3; i++)
  {
    float a00, a10, a01, a11;
    fetchQuad(a00, a10, a01, a11, texture, {tc0.x+i, tc0.y+i, tc0.z+i, tc0.w+i}, corner0);
    pOut[i] = bilerp(a00, a10, a01, a11, uv0);
  }
  return {pOut[0], pOut[1], pOut[2]}; // Exit.
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif