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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
  const float3* means,
  const int* radii,
  const float* cov3Ds,
  const float h_x, float h_y,
  const float tan_fovx, float tan_fovy,
  const float* view_matrix,
  const float* dL_dconics,
  float3* dL_dmeans,
  float* dL_dcov)
{
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0))
    return;

  // Reading location of 3D covariance for this Gaussian
  const float* cov3D = cov3Ds + 6 * idx;

  // Fetch gradients, recompute 2D covariance and relevant 
  // intermediate forward results needed in the backward.
  float3 mean = means[idx];
  float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
  float3 t = transformPoint4x3(mean, view_matrix);
  
  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;
  
  const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
  const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

  glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
    0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
    0, 0, 0);

  glm::mat3 W = glm::mat3(
    view_matrix[0], view_matrix[4], view_matrix[8],
    view_matrix[1], view_matrix[5], view_matrix[9],
    view_matrix[2], view_matrix[6], view_matrix[10]);

  glm::mat3 Vrk = glm::mat3(
    cov3D[0], cov3D[1], cov3D[2],
    cov3D[1], cov3D[3], cov3D[4],
    cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 T = W * J;

  glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Use helper variables for 2D covariance entries. More compact.
  float a = cov2D[0][0] += 0.3f;
  float b = cov2D[0][1];
  float c = cov2D[1][1] += 0.3f;

  float denom = a * c - b * b;
  float dL_da = 0, dL_db = 0, dL_dc = 0;
  float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

  if (denom2inv != 0)
  {
    // Gradients of loss w.r.t. entries of 2D covariance matrix,
    // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
    // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
    dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
    dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
    dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
    // given gradients w.r.t. 2D covariance matrix (diagonal).
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
    dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
    dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
    // Off-diagonal elements appear twice --> double the gradient.
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
    dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
    dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
  }
  else
  {
    for (int i = 0; i < 6; i++)
      dL_dcov[6 * idx + i] = 0;
  }

  // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
  // cov2D = transpose(T) * transpose(Vrk) * T;
  float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
    (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
  float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
    (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
  float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
    (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
  float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
    (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
  float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
    (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
  float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
    (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

  // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
  // T = W * J
  float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
  float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
  float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
  float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

  float tz = 1.f / t.z;
  float tz2 = tz * tz;
  float tz3 = tz2 * tz;

  // Gradients of loss w.r.t. transformed Gaussian mean t
  float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
  float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
  float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

  // Account for transformation of mean to t
  // t = transformPoint4x3(mean, view_matrix);
  float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

  // Gradients of loss w.r.t. Gaussian means, but only the portion 
  // that is caused because the mean affects the covariance matrix.
  // Additional mean gradient is accumulated in BACKWARD::preprocess.
  dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
  // Recompute (intermediate) results for the 3D covariance computation.
  glm::vec4 q = rot;// / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  glm::mat3 R = glm::mat3(
    1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
    2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
    2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
  );

  glm::mat3 S = glm::mat3(1.0f);

  glm::vec3 s = mod * scale;
  S[0][0] = s.x;
  S[1][1] = s.y;
  S[2][2] = s.z;

  glm::mat3 M = S * R;

  const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

  glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
  glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

  // Convert per-element covariance loss gradients to matrix form
  glm::mat3 dL_dSigma = glm::mat3(
    dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
    0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
    0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
  );

  // Compute loss gradient w.r.t. matrix M
  // dSigma_dM = 2 * M
  glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

  glm::mat3 Rt = glm::transpose(R);
  glm::mat3 dL_dMt = glm::transpose(dL_dM);

  // Gradients of loss w.r.t. scale
  glm::vec3* dL_dscale = dL_dscales + idx;
  dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
  dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
  dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

  dL_dMt[0] *= s.x;
  dL_dMt[1] *= s.y;
  dL_dMt[2] *= s.z;

  // Gradients of loss w.r.t. normalized quaternion
  glm::vec4 dL_dq;
  dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
  dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
  dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
  dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

  // Gradients of loss w.r.t. unnormalized quaternion
  float4* dL_drot = (float4*)(dL_drots + idx);
  *dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}


__device__ void computeNorm3D(int idx, const glm::vec3 scale, const glm::vec4 rot, const glm::vec3 norm3D, const glm::vec3 dL_dnorm3D, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
  // Recompute (intermediate) results for the 3D covariance computation.
  glm::vec4 q = rot;// / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  glm::mat3 R = glm::mat3(
    1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
    2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
    2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
  );

  glm::vec3 norm; 
  if(scale.x > scale.z && scale.y > scale.z)
  {
    norm = glm::vec3(0.0, 0.0, 1.0);
  }
  else if(scale.x > scale.y && scale.z > scale.y)
  {
    norm = glm::vec3(0.0, 1.0, 0.0);
  }
  else
  {
    norm = glm::vec3(1.0, 0.0, 0.0);
  }

  if(glm::dot(R * norm, norm3D) < 0)
    norm = -norm;
  
  glm::mat3 dL_dR = glm::mat3(
    dL_dnorm3D.x * norm.x, dL_dnorm3D.x * norm.y, dL_dnorm3D.x * norm.z,
    dL_dnorm3D.y * norm.x, dL_dnorm3D.y * norm.y, dL_dnorm3D.y * norm.z,
    dL_dnorm3D.z * norm.x, dL_dnorm3D.z * norm.y, dL_dnorm3D.z * norm.z
  );
  glm::mat3 dL_dRt = glm::transpose(dL_dR);

  // Gradients of loss w.r.t. normalized quaternion
  glm::vec4 dL_dq;
  dL_dq.x = 2 * z * (dL_dRt[0][1] - dL_dRt[1][0]) + 2 * y * (dL_dRt[2][0] - dL_dRt[0][2]) + 2 * x * (dL_dRt[1][2] - dL_dRt[2][1]);
  dL_dq.y = 2 * y * (dL_dRt[1][0] + dL_dRt[0][1]) + 2 * z * (dL_dRt[2][0] + dL_dRt[0][2]) + 2 * r * (dL_dRt[1][2] - dL_dRt[2][1]) - 4 * x * (dL_dRt[2][2] + dL_dRt[1][1]);
  dL_dq.z = 2 * x * (dL_dRt[1][0] + dL_dRt[0][1]) + 2 * r * (dL_dRt[2][0] - dL_dRt[0][2]) + 2 * z * (dL_dRt[1][2] + dL_dRt[2][1]) - 4 * y * (dL_dRt[2][2] + dL_dRt[0][0]);
  dL_dq.w = 2 * r * (dL_dRt[0][1] - dL_dRt[1][0]) + 2 * x * (dL_dRt[2][0] + dL_dRt[0][2]) + 2 * y * (dL_dRt[1][2] + dL_dRt[2][1]) - 4 * z * (dL_dRt[1][1] + dL_dRt[0][0]);

  // Gradients of loss w.r.t. unnormalized quaternion
  dL_drots[idx] += dL_dq;
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
__global__ void preprocessCUDA(
  const int P,
  const float3* means,
  const int* radii,
  const glm::vec3* norm3Ds,
  const glm::vec3* scales,
  const glm::vec4* rotations,
  const float scale_modifier,
  const float* view,
  const float* proj,
  const glm::vec3* campos,
  const float3* dL_dmean2D,
  glm::vec3* dL_dmeans,
  float* dL_ddepth,
  float* dL_dcov3D,
  glm::vec3* dL_dnorm3D,
  glm::vec3* dL_dscale,
  glm::vec4* dL_drot)
{
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0))
    return;

  float3 m = means[idx];

  // Taking care of gradients from the screenspace points
  float4 m_hom = transformPoint4x4(m, proj);
  float m_w = 1.0f / (m_hom.w + 0.0000001f);

  // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
  // from rendering procedure
  glm::vec3 dL_dmean;
  float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
  float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
  dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
  dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
  dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

  // That's the second part of the mean gradient. Previous computation
  // of cov2D and following SH conversion also affects it.
  dL_dmeans[idx] += dL_dmean;

  // the w must be equal to 1 for view^T * [x,y,z,1]
  float3 m_view = transformPoint4x3(m, view);
  
  // Compute loss gradient w.r.t. 3D means due to gradients of depth
  // from rendering procedure
  glm::vec3 dL_dmean2;
  float mul3 = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
  dL_dmean2.x = (view[2] - view[3] * mul3) * dL_ddepth[idx];
  dL_dmean2.y = (view[6] - view[7] * mul3) * dL_ddepth[idx];
  dL_dmean2.z = (view[10] - view[11] * mul3) * dL_ddepth[idx];
  
  // That's the third part of the mean gradient.
  dL_dmeans[idx] += dL_dmean2;

  
  // Compute gradient updates due to computing covariance from scale/rotation
  if (scales)
    computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
    
  computeNorm3D(idx, scales[idx], rotations[idx], norm3Ds[idx], dL_dnorm3D[idx], dL_dscale, dL_drot);
  
}

static __device__ __forceinline__ bool isfinite_vec3(const float3& a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z); }

// Based on dA/d{s,t}, compute dA/d{x,y,z} at a given 3D lookup vector.
static __device__ __forceinline__ float3 indexCubeMapGrad(float3 uv, float gu, float gv)
{
    float ax = fabsf(uv.x);
    float ay = fabsf(uv.y);
    float az = fabsf(uv.z);
    int idx;
    float c;
    float c0 = gu;
    float c1 = gv;
    if (az > fmaxf(ax, ay)) { idx = 0x10; c = uv.z; c0 *= uv.x; c1 *= uv.y; }
    else if (ay > ax)       { idx = 0x04; c = uv.y; c0 *= uv.x; c1 *= uv.z; }
    else                    { idx = 0x01; c = uv.x; c0 *= uv.z; c1 *= uv.y; }
    if (c < 0.f) idx += idx;
    float m = __frcp_rz(fabsf(c));
    c0 = (idx & 0x34) ? -c0 : c0;
    c1 = (idx & 0x2e) ? -c1 : c1;
    float gl = (c0 + c1) * m;
    float gx = (idx & 0x03) ? gl : (idx & 0x20) ? -gu : gu;
    float gy = (idx & 0x0c) ? gl : -gv;
    float gz = (idx & 0x30) ? gl : (idx & 0x03) ? gu : gv;
    gz = (idx & 0x09) ? -gz : gz;
    float3 res = make_float3(gx * (m * .5f), gy * (m * .5f), gz * (m * .5f));
    if (!isfinite_vec3(res))
        return make_float3(0.f, 0.f, 0.f); // Invalid uv.
    return res;
}

static __device__ __forceinline__ void accumQuad(float4 c, glm::vec3 dL_dsh, float* pOut, int4 tc, bool corner)
{
    // For invalid cube map uv, tc will be all negative, and no accumulation will take place.
    if (corner)
    {
        glm::vec3 cb;
        if (tc.x < 0) cb = c.x * dL_dsh;
        if (tc.y < 0) cb = c.y * dL_dsh;
        if (tc.z < 0) cb = c.z * dL_dsh;
        if (tc.w < 0) cb = c.w * dL_dsh;
        cb *= 0.33333333f;
        if (tc.x >= 0) {
          atomicAdd(&(pOut[tc.x * 3 + 0]), c.x * dL_dsh.x + cb.x);
          atomicAdd(&(pOut[tc.x * 3 + 1]), c.x * dL_dsh.y + cb.y);
          atomicAdd(&(pOut[tc.x * 3 + 2]), c.x * dL_dsh.z + cb.z); 
        }
        if (tc.y >= 0) {
          atomicAdd(&(pOut[tc.y * 3 + 0]), c.y * dL_dsh.x + cb.x);
          atomicAdd(&(pOut[tc.y * 3 + 1]), c.y * dL_dsh.y + cb.y);
          atomicAdd(&(pOut[tc.y * 3 + 2]), c.y * dL_dsh.z + cb.z); 
        }
        if (tc.z >= 0) {
          atomicAdd(&(pOut[tc.z * 3 + 0]), c.z * dL_dsh.x + cb.x);
          atomicAdd(&(pOut[tc.z * 3 + 1]), c.z * dL_dsh.y + cb.y);
          atomicAdd(&(pOut[tc.z * 3 + 2]), c.z * dL_dsh.z + cb.z); 
        }
        if (tc.w >= 0) {
          atomicAdd(&(pOut[tc.w * 3 + 0]), c.w * dL_dsh.x + cb.x);
          atomicAdd(&(pOut[tc.w * 3 + 1]), c.w * dL_dsh.y + cb.y);
          atomicAdd(&(pOut[tc.w * 3 + 2]), c.w * dL_dsh.z + cb.z); 
        } 
    }
    else
    {
        if (tc.x >= 0) {
          atomicAdd(&(pOut[tc.x * 3 + 0]), c.x * dL_dsh.x);
          atomicAdd(&(pOut[tc.x * 3 + 1]), c.x * dL_dsh.y);
          atomicAdd(&(pOut[tc.x * 3 + 2]), c.x * dL_dsh.z); 
        } 
        if (tc.y >= 0) {
          atomicAdd(&(pOut[tc.y * 3 + 0]), c.y * dL_dsh.x);
          atomicAdd(&(pOut[tc.y * 3 + 1]), c.y * dL_dsh.y);
          atomicAdd(&(pOut[tc.y * 3 + 2]), c.y * dL_dsh.z); 
        }
        if (tc.z >= 0) {
          atomicAdd(&(pOut[tc.z * 3 + 0]), c.z * dL_dsh.x);
          atomicAdd(&(pOut[tc.z * 3 + 1]), c.z * dL_dsh.y);
          atomicAdd(&(pOut[tc.z * 3 + 2]), c.z * dL_dsh.z); 
        } 
        if (tc.w >= 0)  {
          atomicAdd(&(pOut[tc.w * 3 + 0]), c.w * dL_dsh.x);
          atomicAdd(&(pOut[tc.w * 3 + 1]), c.w * dL_dsh.y);
          atomicAdd(&(pOut[tc.w * 3 + 2]), c.w * dL_dsh.z); 
        }
    }
}


__forceinline__ __device__ float3 cube_texture_fetch_forward(const float3& uv, const float* texture, const float3 &dir, const int TR, const int D, const int M, int3 &clamped)
{
  int4 tc0 = {0, 0, 0, 0};
  float2 uv0 = indexTextureLinear(uv, tc0, TR);
  bool corner0 = ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
  tc0 = {tc0.x * M * M, tc0.y * M * M, tc0.z * M * M, tc0.w * M * M};

  glm::vec3 sh[16];
  // Interpolate.
  for (int i=0; i < (D+1)*(D+1); i++)
  {
    glm::vec3 a00, a10, a01, a11;
    fetchQuad(a00, a10, a01, a11, (glm::vec3*)texture, {tc0.x+i, tc0.y+i, tc0.z+i, tc0.w+i}, corner0);
    sh[i] = bilerp(a00, a10, a01, a11, uv0);
  }
  glm::vec3 result = SH_C0 * sh[0];

  if (D > 0)
  {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (D > 1)
    {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result +
        SH_C2[0] * xy * sh[4] +
        SH_C2[1] * yz * sh[5] +
        SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
        SH_C2[3] * xz * sh[7] +
        SH_C2[4] * (xx - yy) * sh[8];

      if (D > 2)
      {
        result = result +
          SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
          SH_C3[1] * xy * z * sh[10] +
          SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
          SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
          SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
          SH_C3[5] * z * (xx - yy) * sh[14] +
          SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }
  result += 0.5f;
  clamped = make_int3(result.x < 0.0f, result.y < 0.0f, result.z < 0.0f);
  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  
  return {max(result.x, 0.0f), max(result.y, 0.0f), max(result.z, 0.0f)}; // Exit.
}

__forceinline__ __device__ void cube_texture_fetch_backward(const float3& uv, const float* texture, const float3 dir, const int TR, const int D, const int M, int3 &clamped, float3 grad_color, float3& grad_uv, float* grad_tex)
{
  // UV gradient accumulators.
  float gu = 0.f;
  float gv = 0.f;

  int4 tc0 = {0, 0, 0, 0};
  float2 uv0 = indexTextureLinear(uv, tc0, TR);
  bool corner0 = ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
  tc0 = {tc0.x * M * M, tc0.y * M * M, tc0.z * M * M, tc0.w * M * M};  
  // Texel weights.
  float uv011 = uv0.x * uv0.y;
  float uv010 = uv0.x - uv011;
  float uv001 = uv0.y - uv011;
  float uv000 = 1.f - uv0.x - uv001;
  float4 tw0 = make_float4(uv000, uv010, uv001, uv011);

  glm::vec3 dL_dRGB = {grad_color.x * (1 - clamped.x), grad_color.y * (1 - clamped.y), grad_color.z * (1 - clamped.z)};
  glm::vec3 dL_dsh[16];
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;

  float dRGBdsh0 = SH_C0;
  dL_dsh[0] = dRGBdsh0 * dL_dRGB;
  if (D > 0)
  {
    float dRGBdsh1 = -SH_C1 * y;
    float dRGBdsh2 = SH_C1 * z;
    float dRGBdsh3 = -SH_C1 * x;
    dL_dsh[1] = dRGBdsh1 * dL_dRGB;
    dL_dsh[2] = dRGBdsh2 * dL_dRGB;
    dL_dsh[3] = dRGBdsh3 * dL_dRGB;

    if (D > 1)
    {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;

      float dRGBdsh4 = SH_C2[0] * xy;
      float dRGBdsh5 = SH_C2[1] * yz;
      float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
      float dRGBdsh7 = SH_C2[3] * xz;
      float dRGBdsh8 = SH_C2[4] * (xx - yy);
      dL_dsh[4] = dRGBdsh4 * dL_dRGB;
      dL_dsh[5] = dRGBdsh5 * dL_dRGB;
      dL_dsh[6] = dRGBdsh6 * dL_dRGB;
      dL_dsh[7] = dRGBdsh7 * dL_dRGB;
      dL_dsh[8] = dRGBdsh8 * dL_dRGB;

      if (D > 2)
      {
        float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
        float dRGBdsh10 = SH_C3[1] * xy * z;
        float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
        float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
        dL_dsh[9] = dRGBdsh9 * dL_dRGB;
        dL_dsh[10] = dRGBdsh10 * dL_dRGB;
        dL_dsh[11] = dRGBdsh11 * dL_dRGB;
        dL_dsh[12] = dRGBdsh12 * dL_dRGB;
        dL_dsh[13] = dRGBdsh13 * dL_dRGB;
        dL_dsh[14] = dRGBdsh14 * dL_dRGB;
        dL_dsh[15] = dRGBdsh15 * dL_dRGB;
      }
    }
  }

  for (int i=0; i < (D+1)*(D+1); i++)
  {
      accumQuad(tw0, dL_dsh[i], grad_tex, {tc0.x+i, tc0.y+i, tc0.z+i, tc0.w+i}, corner0);
      glm::vec3 a00, a10, a01, a11;
      fetchQuad(a00, a10, a01, a11, (glm::vec3*) texture, {tc0.x+i, tc0.y+i, tc0.z+i, tc0.w+i}, corner0);
      glm::vec3 ad = (a11 + a00 - a10 - a01);
      glm::vec3 tmp;
      tmp = dL_dsh[i] * ((a10 - a00) + uv0.y * ad) * float(TR);
      gu += tmp.x + tmp.y + tmp.z;
      tmp = dL_dsh[i] * ((a01 - a00) + uv0.x * ad) * float(TR);
      gv += tmp.x + tmp.y + tmp.z; 
  }

  // Store UV gradients and exit.
  grad_uv = indexCubeMapGrad(uv, gu, gv);
}

// Backward version of the rendering procedure.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
  const uint2* __restrict__ ranges,
  const uint32_t* __restrict__ point_list,
  const int W,
  const int H,
  const int D, 
  const int M,
  const int ED,
  const int TR,
  const float* __restrict__ bg_color,
  const glm::vec3* cam_pos,
  const float2* __restrict__ points_xy_image,
  const float3* __restrict__ orig_points,
  const float4* __restrict__ conic_opacity,
  const float tan_fovx,
  const float tan_fovy,
  const float* viewmatrix,
  const float* viewmatrix_inv,
  const float* __restrict__ depths,
  const float* __restrict__ norms,
  const float* __restrict__ uvs,
  const float* __restrict__ gradient_uvs,
  const float* __restrict__ texture,
  const float* __restrict__ extras,
  const float* __restrict__ accum_alphas,
  const uint32_t* __restrict__ n_contrib,
  const float* __restrict__ dL_dpixels,
  const float* __restrict__ dL_dpixel_depths,
  const float* __restrict__ dL_dpixel_norms,
  const float* __restrict__ dL_dpixel_alphas,
  const float* __restrict__ dL_dpixel_extras,
  float3* __restrict__ dL_dmean2D,
  float4* __restrict__ dL_dconic2D,
  float* __restrict__ dL_dopacity,
  float* __restrict__ dL_duvs,
  float* __restrict__ dL_dtexture,
  float* __restrict__ dL_ddepths,
  float* __restrict__ dL_dnorm3Ds,
  float* __restrict__ dL_dextras)
{
  // We rasterize again. Compute necessary block info.
  auto block = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
  const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
  const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
  const uint32_t pix_id = W * pix.y + pix.x;
  const float2 pixf = { (float)pix.x, (float)pix.y };
  float3 pix_dir = {pix2Ndc(pix.x, W) * tan_fovx, pix2Ndc(pix.y, H) * tan_fovy, 1.0};
  pix_dir = transformVec4x3(pix_dir, viewmatrix_inv);
  float3 cam_p = {cam_pos->x, cam_pos->y, cam_pos->z};
  
  const bool inside = pix.x < W&& pix.y < H;
  const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  bool done = !inside;
  int toDo = range.y - range.x;

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
  __shared__ float collected_depths[BLOCK_SIZE];
  __shared__ float collected_norms[3 * BLOCK_SIZE];
  __shared__ float collected_extras[MAX_EXTRA_DIMS * BLOCK_SIZE];

  // In the forward, we stored the final value for T, the
  // product of all (1 - alpha) factors. 
  const float T_final = inside ? (1 - accum_alphas[pix_id]) : 0;
  float T = T_final;

  // We start from the back. The ID of the last contributing
  // Gaussian is known from each pixel from the forward.
  uint32_t contributor = toDo;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  float accum_rec[3] = { 0 };
  float accum_red = 0;
  float accum_ren[3] = { 0 };
  float accum_rea = 0;
  float accum_ree[MAX_EXTRA_DIMS] = { 0 };
  float dL_dpixel[3];
  float dL_dpixel_depth;
  float dL_dpixel_norm[3];
  float dL_dpixel_alpha;
  float dL_dpixel_extra[MAX_EXTRA_DIMS];
  if (inside) 
  {
    for (int i = 0; i < 3; i++)
      dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
    dL_dpixel_depth = dL_dpixel_depths[pix_id];
    for (int i = 0; i < 3; i++)
      dL_dpixel_norm[i] = dL_dpixel_norms[i * H * W + pix_id];
    dL_dpixel_alpha = dL_dpixel_alphas[pix_id];
    for (int i = 0; i < ED; i++)
      dL_dpixel_extra[i] = dL_dpixel_extras[i * H * W + pix_id];
  }
  float last_alpha = 0;
  float last_color[3] = { 0 };
  float last_depth = 0;
  float last_norm[3] = { 0 };
  float last_extra[MAX_EXTRA_DIMS] = { 0 };
  // Gradient of pixel coordinate w.r.t. normalized 
  // screen-space viewport corrdinates (-1 to 1)
  const float ddelx_dx = 0.5 * W;
  const float ddely_dy = 0.5 * H;

  // Traverse all Gaussians
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
  {
    // Load auxiliary data into shared memory, start in the BACK
    // and load them in revers order.
    block.sync();
    const int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y)
    {
      const int coll_id = point_list[range.y - progress - 1];
      collected_id[block.thread_rank()] = coll_id;
      collected_xy[block.thread_rank()] = points_xy_image[coll_id];
      collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
      for (int i = 0; i < 3; i++)
        collected_norms[i * BLOCK_SIZE + block.thread_rank()] = norms[coll_id * 3 + i];
      collected_depths[block.thread_rank()] = depths[coll_id];
      for (int i = 0; i < ED; i++)
        collected_extras[i * BLOCK_SIZE + block.thread_rank()] = extras[coll_id * ED + i];
    }
    block.sync();

    // Iterate over Gaussians
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
    {
      // Keep track of current Gaussian ID. Skip, if this one
      // is behind the last contributor for this pixel.
      contributor--;
      if (contributor >= last_contributor)
        continue;

      // Compute blending values, as before.
      const float2 xy = collected_xy[j];
      const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
      const float4 con_o = collected_conic_opacity[j];
      const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;

      const float G = exp(power);
      const float alpha = min(0.99f, con_o.w * G);
      if (alpha < 1.0f / 255.0f)
        continue;

      T = T / (1.f - alpha);
      const float weight = alpha * T;
      // const float dchannel_dcolor = alpha * T;
      // const float dpixel_depth_ddepth = alpha * T;
      // const float dpixel_norm_dnorm = alpha * T;
      // const float dpixel_extra_dextra = alpha * T;

      // Propagate gradients to per-Gaussian colors and keep
      // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
      // pair).
      float dL_dalpha = 0.0f;
      const int global_id = collected_id[j];

      float3 orig_point = orig_points[global_id];
      float3 norm = {collected_norms[0 * BLOCK_SIZE + j], collected_norms[1 * BLOCK_SIZE + j], collected_norms[2 * BLOCK_SIZE + j]};
      // (cam_p - orig_point) * norm
      float bias = (cam_p.x - orig_point.x)*norm.x + (cam_p.y - orig_point.y)*norm.y + (cam_p.z - orig_point.z)*norm.z;
      float denom = pix_dir.x * norm.x + pix_dir.y * norm.y + pix_dir.z * norm.z;
      float t;
      float3 delta_xyz;
      if(fabs(denom) > 1e-6){
        t = -bias / denom;
        delta_xyz = {cam_p.x + t*pix_dir.x - orig_point.x, cam_p.y + t*pix_dir.y  - orig_point.y, cam_p.z + t*pix_dir.z - orig_point.z};
      }else{
        delta_xyz = {0, 0, 0};
      }
      float3 delta_uv = Vec3x3(gradient_uvs + global_id*9, delta_xyz);
      float3 uv = {uvs[global_id*3] + delta_uv.x, uvs[global_id*3+1] + delta_uv.y, uvs[global_id*3+2] + delta_uv.z};
      denom = sqrt(uv.x * uv.x + uv.y * uv.y + uv.z * uv.z);
      float3 norm_uv;
      if(denom < 1e-6){
        norm_uv = {uvs[global_id*3], uvs[global_id*3+1], uvs[global_id*3+2]};
        uv = norm_uv;
      }
      else
        norm_uv = {uv.x / denom, uv.y / denom, uv.z / denom};
      
      int3 clamped;
      float3 color = cube_texture_fetch_forward(norm_uv, texture, pix_dir, TR, D, M, clamped);
      float3 dL_dcolor = {0, 0, 0};
      for (int ch = 0; ch < 3; ch++)
      {
        const float c = (ch == 0) ? color.x : ((ch == 1) ? color.y : color.z);
        // Update last color (to be used in the next iteration)
        accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
        last_color[ch] = c;

        const float dL_dchannel = dL_dpixel[ch];
        dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
        // Update the gradients w.r.t. color of the Gaussian. 
        // Atomic, since this pixel is just one of potentially
        // many that were affected by this Gaussian.
        if(ch==0)
          dL_dcolor.x = weight * dL_dchannel;
        else if(ch==1)
          dL_dcolor.y = weight * dL_dchannel;
        else if(ch==2)
          dL_dcolor.z = weight * dL_dchannel;
      }
      float3 dL_dnorm_uv;
      cube_texture_fetch_backward(norm_uv, texture, pix_dir, TR, D, M, clamped, dL_dcolor, dL_dnorm_uv, dL_dtexture);
      float3 dL_duv = dnormvdv(uv, dL_dnorm_uv);
      atomicAdd(&(dL_duvs[global_id * 3]), dL_duv.x);
      atomicAdd(&(dL_duvs[global_id * 3+1]), dL_duv.y);
      atomicAdd(&(dL_duvs[global_id * 3+2]), dL_duv.z);

      const float dep = collected_depths[j];
      accum_red = last_alpha * last_depth + (1.f - last_alpha) * accum_red;
      last_depth = dep;
      dL_dalpha += (dep-accum_red) * dL_dpixel_depth;
      atomicAdd(&(dL_ddepths[global_id]), weight * dL_dpixel_depth);
      
      for (int ch = 0; ch < 3; ch++)
      {
        const float n = collected_norms[ch * BLOCK_SIZE + j];
        // Update last norm (to be used in the next iteration)
        accum_ren[ch] = last_alpha * last_norm[ch] + (1.f - last_alpha) * accum_ren[ch];
        last_norm[ch] = n;

        const float dL_dnormch = dL_dpixel_norm[ch];
        dL_dalpha += (n - accum_ren[ch]) * dL_dnormch;
        // Update the gradients w.r.t. norm of the Gaussian. 
        // Atomic, since this pixel is just one of potentially
        // many that were affected by this Gaussian.
        atomicAdd(&(dL_dnorm3Ds[global_id * 3 + ch]), weight * dL_dnormch);
      }

      for (int ch = 0; ch < ED; ch++)
      {
        const float e = collected_extras[ch * BLOCK_SIZE + j];
        // Update last norm (to be used in the next iteration)
        accum_ree[ch] = last_alpha * last_extra[ch] + (1.f - last_alpha) * accum_ree[ch];
        last_extra[ch] = e;

        const float dL_dextrach = dL_dpixel_extra[ch];
        dL_dalpha += (e - accum_ree[ch]) * dL_dextrach;
        // Update the gradients w.r.t. norm of the Gaussian. 
        // Atomic, since this pixel is just one of potentially
        // many that were affected by this Gaussian.
        atomicAdd(&(dL_dextras[global_id * ED + ch]), weight * dL_dextrach);
      }

      accum_rea = last_alpha + (1.f - last_alpha) * accum_rea;
      dL_dalpha += (1 - accum_rea) * dL_dpixel_alpha;


      dL_dalpha *= T;
      // Update last alpha (to be used in the next iteration)
      last_alpha = alpha;

      // Account for fact that alpha also influences how much of
      // the background color is added if nothing left to blend
      float bg_dot_dpixel = 0;
      for (int i = 0; i < 3; i++)
        bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
      dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

      // Set background depth value == 0, thus no contribution for
      // dL_dalpha

      // Helpful reusable temporary variables
      const float dL_dG = con_o.w * dL_dalpha;
      const float gdx = G * d.x;
      const float gdy = G * d.y;
      const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
      const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

      // Update gradients w.r.t. 2D mean position of the Gaussian
      atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
      atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

      // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
      atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
      atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
      atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

      // Update gradients w.r.t. opacity of the Gaussian
      atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
    }
  }
}

void BACKWARD::preprocess(
  const int P,
  const float3* means3D,
  const int* radii,
  const glm::vec3* scales,
  const glm::vec4* rotations,
  const float scale_modifier,
  const float* cov3Ds,
  const glm::vec3* norm3Ds,
  const float* viewmatrix,
  const float* projmatrix,
  const float focal_x,
  const float focal_y,
  const float tan_fovx,
  const float tan_fovy,
  const glm::vec3* campos,
  const float3* dL_dmean2D,
  const float* dL_dconic,
  glm::vec3* dL_dmean3D,
  float* dL_ddepth,
  float* dL_dcov3D,
  glm::vec3* dL_dnorm3D,
  glm::vec3* dL_dscale,
  glm::vec4* dL_drot)
{
  // Propagate gradients for the path of 2D conic matrix computation. 
  // Somewhat long, thus it is its own kernel rather than being part of 
  // "preprocess". When done, loss gradient w.r.t. 3D means has been
  // modified and gradient w.r.t. 3D covariance matrix has been computed.  
  computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
    P,
    means3D,
    radii,
    cov3Ds,
    focal_x,
    focal_y,
    tan_fovx,
    tan_fovy,
    viewmatrix,
    dL_dconic,
    (float3*)dL_dmean3D,
    dL_dcov3D);

  // Propagate gradients for remaining steps: finish 3D mean gradients,
  // propagate color gradients to SH (if desireD), propagate 3D covariance
  // matrix gradients to scale and rotation.
  preprocessCUDA << < (P + 255) / 256, 256 >> > (
    P,
    (float3*)means3D,
    radii,
    (glm::vec3*)norm3Ds,
    (glm::vec3*)scales,
    (glm::vec4*)rotations,
    scale_modifier,
    viewmatrix,
    projmatrix,
    campos,
    (float3*)dL_dmean2D,
    (glm::vec3*)dL_dmean3D,
    dL_ddepth,
    dL_dcov3D,
    (glm::vec3*)dL_dnorm3D,
    dL_dscale,
    dL_drot);
}

void BACKWARD::render(
  const dim3 grid, const dim3 block,
  const uint2* ranges,
  const uint32_t* point_list,
  const int W,
  const int H,
  const int D, 
  const int M,
  const int ED,
  const int TR,
  const float* bg_color,
  const glm::vec3* campos,
  const float2* means2D,
  const float3* means3D,
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
  float* dL_duvs,
  float* dL_dtexture,
  float* dL_ddepths,
  float* dL_dnorm3Ds,
  float* dL_dextras)
{
  renderCUDA << <grid, block >> >(
    ranges,
    point_list,
    W, H, D, M, ED, TR,
    bg_color,
    campos,
    means2D,
    means3D,
    conic_opacity,
    tan_fovx, tan_fovy,
    viewmatrix,
    viewmatrix_inv,
    depths,
    norms,
    uvs,
    gradient_uvs,
    texture,
    extras,
    accum_alphas,
    n_contrib,
    dL_dpixels,
    dL_dpixel_depths,
    dL_dpixel_norms,
    dL_dpixel_alphas,
    dL_dpixel_extras,
    dL_dmean2D,
    dL_dconic2D,
    dL_dopacity,
    dL_duvs,
    dL_dtexture,
    dL_ddepths,
    dL_dnorm3Ds,
    dL_dextras);
}