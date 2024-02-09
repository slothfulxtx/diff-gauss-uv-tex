#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ float3 computeSpecularColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs)
{
  // The implementation is loosely based on code for 
  // "Differentiable Point-Based Radiance Fields for 
  // Efficient View Synthesis" by Zhang et al. (2022)
  glm::vec3 pos = means[idx];
  glm::vec3 dir = pos - campos;
  dir = dir / glm::length(dir);

  glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
  glm::vec3 result(0.0, 0.0, 0.0);

  if (deg > 0)
  {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

    if (deg > 1)
    {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result +
        SH_C2[0] * xy * sh[3] +
        SH_C2[1] * yz * sh[4] +
        SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
        SH_C2[3] * xz * sh[6] +
        SH_C2[4] * (xx - yy) * sh[7];

      if (deg > 2)
      {
        result = result +
          SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
          SH_C3[1] * xy * z * sh[9] +
          SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
          SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
          SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
          SH_C3[5] * z * (xx - yy) * sh[13] +
          SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
      }
    }
  }
  return make_float3(result.x, result.y, result.z);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
  // The following models the steps outlined by equations 29
  // and 31 in "EWA Splatting" (Zwicker et al., 2002). 
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  float3 t = transformPoint4x3(mean, viewmatrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J = glm::mat3(
    focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
    0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
    0, 0, 0);

  glm::mat3 W = glm::mat3(
    viewmatrix[0], viewmatrix[4], viewmatrix[8],
    viewmatrix[1], viewmatrix[5], viewmatrix[9],
    viewmatrix[2], viewmatrix[6], viewmatrix[10]);

  glm::mat3 T = W * J;

  glm::mat3 Vrk = glm::mat3(
    cov3D[0], cov3D[1], cov3D[2],
    cov3D[1], cov3D[3], cov3D[4],
    cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.
  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
  // Create scaling matrix
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;

  // Normalize quaternion to get valid rotation
  glm::vec4 q = rot;// / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(
    1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
    2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
    2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
  );

  glm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  glm::mat3 Sigma = glm::transpose(M) * M;

  // Covariance is symmetric, only store upper right
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D norm vector in world space.
__device__ void computeNorm3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* norm3D, int idx, const glm::vec3* means, glm::vec3 campos)
{
  // Create scaling matrix
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;

  // Normalize quaternion to get valid rotation
  glm::vec4 q = rot;// / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix from quaternion
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
  norm = glm::transpose(R) * norm;

  glm::vec3 raydir = means[idx] - campos;
  if(glm::dot(raydir, norm) > 0)
    norm = -norm;

  norm3D[0] = norm.x;
  norm3D[1] = norm.y;
  norm3D[2] = norm.z;
}

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
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
  float* cov3Ds,
  float* norm3Ds,
  float4* conic_opacity,
  const dim3 grid,
  uint32_t* tiles_touched,
  bool prefiltered)
{
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx] = 0;
  tiles_touched[idx] = 0;

  // Perform near culling, quit if outside.
  float3 p_view;
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
    return;

  // Transform point by projecting
  float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

  // If 3D covariance matrix is precomputed, use it, otherwise compute
  // from scaling and rotation parameters. 
  computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
  const float* cov3D = cov3Ds + idx * 6;
  
  computeNorm3D(scales[idx], scale_modifier, rotations[idx], norm3Ds + idx * 3, idx, (glm::vec3*)orig_points, *cam_pos);
  const float*  norm3D = norm3Ds + idx * 3;

  // Compute 2D screen-space covariance matrix
  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

  // Invert covariance (EWA algorithm)
  float det = (cov.x * cov.z - cov.y * cov.y);
  if (det == 0.0f)
    return;
  float det_inv = 1.f / det;
  float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

  // Compute extent in screen space (by finding eigenvalues of
  // 2D covariance matrix). Use extent to compute a bounding rectangle
  // of screen-space tiles that this Gaussian overlaps with. Quit if
  // rectangle covers 0 tiles. 
  float mid = 0.5f * (cov.x + cov.z);
  float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
  float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
  uint2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid);
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
    return;

  if (D > 0)
  {
    rgbs[idx] = computeSpecularColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs);
  }
  // Store some useful helper data for the next steps.
  depths[idx] = p_view.z;
  radii[idx] = my_radius;
  points_xy_image[idx] = point_image;
  // Inverse 2D covariance and opacity neatly pack into one float4
  conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
  const uint2* __restrict__ ranges,
  const uint32_t* __restrict__ point_list,
  const int W, 
  const int H,
  const int ED,
  const int TR,
  const float2* __restrict__ points_xy_image,
  const float* __restrict__ orig_points,
  const float3* __restrict__ rgbs,
  const float* __restrict__ norms,
  const float* __restrict__ depths,
  const float* __restrict__ uvs,
  const float* __restrict__ gradient_uvs,
  const float* __restrict__ texture,
  const float* __restrict__ extras,
  const float4* __restrict__ conic_opacity,
  const float tan_fovx,
  const float tan_fovy,
  const float* viewmatrix,
  const float* viewmatrix_inv,
  float* __restrict__ out_alpha,
  uint32_t* __restrict__ n_contrib,
  const float* __restrict__ bg_color,
  const glm::vec3* cam_pos,
  float* __restrict__ out_color,
  float* __restrict__ out_depth,
  float* __restrict__ out_norm,
  float* __restrict__ out_extra)
{
  // Identify current tile and associated min/max pixel range.
  auto block = cg::this_thread_block();
  uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
  uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
  uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
  uint32_t pix_id = W * pix.y + pix.x;
  float2 pixf = { (float)pix.x, (float)pix.y };
  float3 pix_dir = {pix2Ndc(pix.x, W) * tan_fovx, pix2Ndc(pix.y, H) * tan_fovy, 1.0};
  pix_dir = transformVec4x3(pix_dir, viewmatrix_inv);
  float3 cam_p = {cam_pos->x, cam_pos->y, cam_pos->z};
  // Check if this thread is associated with a valid pixel or outside.
  bool inside = pix.x < W&& pix.y < H;
  // Done threads can help with fetching, but don't rasterize
  bool done = !inside;

  // Load start/end range of IDs to process in bit sorted list.
  uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo = range.y - range.x;

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  // Initialize helper variables
  float T = 1.0f;
  uint32_t contributor = 0;
  uint32_t last_contributor = 0;
  float C[3] = { 0 };
  float Dp = 0;
  float N[3] = {0};
  float E[MAX_EXTRA_DIMS] = {0};
  // We assure the extra feature dim ED <= 8

  // Iterate over batches until all done or range is complete
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
  {
    // End if entire block votes that it is done rasterizing
    int num_done = __syncthreads_count(done);
    if (num_done == BLOCK_SIZE)
      break;

    // Collectively fetch per-Gaussian data from global to shared
    int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y)
    {
      int coll_id = point_list[range.x + progress];
      collected_id[block.thread_rank()] = coll_id;
      collected_xy[block.thread_rank()] = points_xy_image[coll_id];
      collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
    }
    block.sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
    {
      // Keep track of current position in range
      contributor++;

      // Resample using conic matrix (cf. "Surface 
      // Splatting" by Zwicker et al., 2001)
      float2 xy = collected_xy[j];
      float2 d = { xy.x - pixf.x, xy.y - pixf.y };
      float4 con_o = collected_conic_opacity[j];
      float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;

      // Eq. (2) from 3D Gaussian splatting paper.
      // Obtain alpha by multiplying with Gaussian opacity
      // and its exponential falloff from mean.
      // Avoid numerical instabilities (see paper appendix). 
      float alpha = min(0.99f, con_o.w * exp(power));
      if (alpha < 1.0f / 255.0f)
        continue;
      float test_T = T * (1 - alpha);
      if (test_T < 0.0001f)
      {
        done = true;
        continue;
      }

      int g_idx = collected_id[j];

      // solve Eq : [(cam_p + t * pix_dir) - orig_points] * norm = 0 

      float3 orig_point = {orig_points[g_idx * 3], orig_points[g_idx * 3+1], orig_points[g_idx * 3+2]};
      float3 norm = {norms[g_idx * 3], norms[g_idx * 3 + 1], norms[g_idx * 3 + 2]};
      // (cam_p - orig_point) * norm
      float bias = (cam_p.x - orig_point.x)*norm.x + (cam_p.y - orig_point.y)*norm.y + (cam_p.z - orig_point.z)*norm.z;
      float denom = pix_dir.x * norm.x + pix_dir.y * norm.y + pix_dir.z * norm.z;
      float t;
      float3 delta_xyz;
      if(fabs(denom) > 1e-6) {
        t = -bias / denom;
        delta_xyz = {cam_p.x + t*pix_dir.x - orig_point.x, cam_p.y + t*pix_dir.y  - orig_point.y, cam_p.z + t*pix_dir.z - orig_point.z};
      }else{
        delta_xyz = {0, 0, 0};
      }
      float3 delta_uv = Vec3x3(gradient_uvs + g_idx*9, delta_xyz);
      float3 uv = {uvs[g_idx*3] + delta_uv.x, uvs[g_idx*3+1] + delta_uv.y, uvs[g_idx*3+2] + delta_uv.z};
      denom = sqrt(uv.x * uv.x + uv.y * uv.y + uv.z * uv.z);
      if(denom < 1e-6)
        uv = {uvs[g_idx*3], uvs[g_idx*3+1], uvs[g_idx*3+2]};
      else
        uv = {uv.x / denom, uv.y / denom, uv.z / denom};
      
      float3 color = cube_texture_fetch(uv, texture, TR, rgbs[g_idx]);

      // Eq. (3) from 3D Gaussian splatting paper.
      C[0] += color.x * alpha * T;
      C[1] += color.y * alpha * T;
      C[2] += color.z * alpha * T;
      Dp += depths[g_idx] * alpha * T;
      for (int ch = 0; ch < 3; ch++)
        N[ch] += norms[g_idx * 3 + ch] * alpha * T;
      for(int ch = 0; ch < ED; ch++)
        E[ch] += extras[g_idx * ED + ch] * alpha * T;
      T = test_T;

      // Keep track of last range entry to update this
      // pixel.
      last_contributor = contributor;
    }
  }

  // All threads that treat valid pixel write out their final
  // rendering data to the frame and auxiliary buffers.
  if (inside)
  {
    out_alpha[pix_id] = 1 - T;
    n_contrib[pix_id] = last_contributor;
    for (int ch = 0; ch < 3; ch++)
      out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    out_depth[pix_id] = Dp;
    // float len = sqrt(N[0]*N[0] + N[1]*N[1] + N[2]*N[2]) + 1e-6;
    for (int ch = 0; ch < 3; ch++)
      out_norm[ch * H * W + pix_id] = N[ch];
    for (int ch = 0; ch < ED; ch++)
      out_extra[ch * H * W + pix_id] = E[ch];
    
  }
}

void FORWARD::render(
  const dim3 grid, dim3 block,
  const uint2* ranges,
  const uint32_t* point_list,
  const int W,
  const int H,
  const int ED,
  const int TR,
  const float2* means2D,
  const float* means3D,
  const float3* rgbs,
  const float* norms,
  const float* depths,
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
  float* out_extra)
{
  renderCUDA << <grid, block >> > (
    ranges,
    point_list,
    W, H, ED, TR,
    means2D,
    means3D,
    rgbs,
    norms,
    depths,
    uvs,
    gradient_uvs,
    texture,
    extras,
    conic_opacity,
    tan_fovx,
    tan_fovy,
    viewmatrix,
    viewmatrix_inv,
    out_alpha,
    n_contrib,
    bg_color,
    cam_pos,
    out_color,
    out_depth,
    out_norm,
    out_extra);
}

void FORWARD::preprocess(
  const int P,
  const int D,
  const int M,
  const float* means3D,
  const float* shs,
  const glm::vec3* scales,
  const float scale_modifier,
  const glm::vec4* rotations,
  const float* opacities,
  const float* viewmatrix,
  const float* projmatrix,
  const glm::vec3* cam_pos,
  const int W, int H,
  const float focal_x,
  const float focal_y,
  const float tan_fovx,
  const float tan_fovy,
  int* radii,
  float2* means2D,
  float3* rgbs,
  float* depths,
  float* cov3Ds,
  float* norm3Ds,
  float4* conic_opacity,
  const dim3 grid,
  uint32_t* tiles_touched,
  bool prefiltered)
{
  preprocessCUDA << <(P + 255) / 256, 256 >> > (
    P, D, M,
    means3D,
    shs,
    scales,
    scale_modifier,
    rotations,
    opacities,
    viewmatrix, 
    projmatrix,
    cam_pos,
    W, H,
    focal_x, focal_y,
    tan_fovx, tan_fovy,
    radii,
    means2D,
    rgbs,
    depths,
    cov3Ds,
    norm3Ds,
    conic_opacity,
    grid,
    tiles_touched,
    prefiltered);
}