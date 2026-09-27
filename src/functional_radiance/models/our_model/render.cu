#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>
#include "helper_math.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// some handy operator overloading & helpers
__device__ inline float sign(float a) {
    return a > 0.0 ? 1.0 : a < 0.0 ? -1.0 : 0.0;
}
__device__ inline float3 sign(float3 a) {
    return make_float3(sign(a.x), sign(a.y), sign(a.z));
}

__device__ float ray_aabb_intersection(float3 ray_origin, float3 ray_direction,
                                        float3 bbox_start, float3 bbox_end,
                                        float t_start) {
    float3 t1 = (bbox_start - ray_origin) / ray_direction;
    float3 t2 = (bbox_end - ray_origin) / ray_direction;
    float3 dmin = fminf(t1, t2);
    float3 dmax = fmaxf(t1, t2);

    float tmin = max(max(max(dmin.x, dmin.y), dmin.z), t_start);
    float tmax = min(min(min(dmax.x, dmax.y), dmax.z), INFINITY);

    return tmax >= tmin ? tmin : NAN;
}

__device__ inline int gridpos_encode(int3 ijk, int32_t grid_size) {
    return ijk.x * grid_size * grid_size + ijk.y * grid_size + ijk.z;
}

__device__ inline float3 sh_coeff(const float* coeffs, int coeff, int sh_deg) {
    int sh_deg_dim = (sh_deg + 1) * (sh_deg + 1);
    return make_float3(coeffs[0*sh_deg_dim + coeff], coeffs[1*sh_deg_dim + coeff], coeffs[2*sh_deg_dim + coeff]);
}
__device__ inline float3 eval_sh(const float* coeffs, float3 direction, int sh_deg) {
    constexpr float C0 = 0.28209479177387814;
    constexpr float C1 = 0.4886025119029199;
    constexpr float C2[] = {
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    };
    constexpr float C3[] = {
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    };
    constexpr float C4[] = {
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    };

    float3 result = make_float3(0.0, 0.0, 0.0);
    result += C0 * sh_coeff(coeffs, 0, sh_deg);
    if (sh_deg > 0) {
        float x = direction.x;
        float y = direction.y;
        float z = direction.z;
        result += -C1 * y * sh_coeff(coeffs, 1, sh_deg) + C1 * z * sh_coeff(coeffs, 2, sh_deg) - C1 * x * sh_coeff(coeffs, 3, sh_deg);

        if (sh_deg > 1) {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;
            result += C2[0] * xy * sh_coeff(coeffs, 4, sh_deg) + C2[1] * yz * sh_coeff(coeffs, 5, sh_deg) + C2[2] * (2.0 * zz - xx - yy) * sh_coeff(coeffs, 6, sh_deg) + C2[3] * xz * sh_coeff(coeffs, 7, sh_deg) + C2[4] * (xx - yy) * sh_coeff(coeffs, 8, sh_deg);

            if (sh_deg > 2) {
                result += (
                    C3[0] * y * (3 * xx - yy) * sh_coeff(coeffs, 9, sh_deg)
                    + C3[1] * xy * z * sh_coeff(coeffs, 10, sh_deg)
                    + C3[2] * y * (4 * zz - xx - yy) * sh_coeff(coeffs, 11, sh_deg)
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeff(coeffs, 12, sh_deg)
                    + C3[4] * x * (4 * zz - xx - yy) * sh_coeff(coeffs, 13, sh_deg)
                    + C3[5] * z * (xx - yy) * sh_coeff(coeffs, 14, sh_deg)
                    + C3[6] * x * (xx - 3 * yy) * sh_coeff(coeffs, 15, sh_deg)
                );

                if (sh_deg > 3) {
                    result += (
                        C4[0] * xy * (xx - yy) * sh_coeff(coeffs, 16, sh_deg)
                        + C4[1] * yz * (3 * xx - yy) * sh_coeff(coeffs, 17, sh_deg)
                        + C4[2] * xy * (7 * zz - 1) * sh_coeff(coeffs, 18, sh_deg)
                        + C4[3] * yz * (7 * zz - 3) * sh_coeff(coeffs, 19, sh_deg)
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh_coeff(coeffs, 20, sh_deg)
                        + C4[5] * xz * (7 * zz - 3) * sh_coeff(coeffs, 21, sh_deg)
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh_coeff(coeffs, 22, sh_deg)
                        + C4[7] * xz * (xx - 3 * yy) * sh_coeff(coeffs, 23, sh_deg)
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh_coeff(coeffs, 24, sh_deg)
                    );
                }
            }
        }
    }

    return result;
}

__device__ inline int interp_to_grid_on_axis(float x, float inf, float sup, int32_t grid_size) {
    float t = (x - inf) / (sup - inf);
    int index = floor(t * grid_size);
    index = clamp(index, 0, grid_size - 1);
    return index;
}

__device__ inline float3 get_voxel_offset(float3 x, int3 voxel_i, float3 inf, float3 sup, int32_t grid_size) {
    float3 t = (x - inf) / (sup - inf);
    return (float)grid_size * t - make_float3(voxel_i);
}

__device__ inline float trilinear_interp(const float* vertices, float3 offset) {
    float v000 = vertices[0];
    float v001 = vertices[1];
    float v010 = vertices[2];
    float v011 = vertices[3];
    float v100 = vertices[4];
    float v101 = vertices[5];
    float v110 = vertices[6];
    float v111 = vertices[7];

    float v00 = lerp(v000, v001, offset.z);
    float v01 = lerp(v010, v011, offset.z);
    float v10 = lerp(v100, v101, offset.z);
    float v11 = lerp(v110, v111, offset.z);

    float v0 = lerp(v00, v01, offset.y);
    float v1 = lerp(v10, v11, offset.y);

    return lerp(v0, v1, offset.x);
}

template <bool use_trilinear_interp>
__device__ void render_ray_kernel(float3 ray_origin,
                                  float3 ray_direction,
                                  float t_start,
                                  float t_end,
                                  const float* density_grid,
                                  const float* color_grid,
                                  float* out_color,
                                  float* out_transmittance,
                                  float3 bbox_start, float3 bbox_end,
                                  int32_t grid_size,
                                  int32_t sh_deg) {
    float entry_t = ray_aabb_intersection(ray_origin, ray_direction, bbox_start, bbox_end, t_start);

    float3 entry_pos = ray_origin + entry_t * ray_direction;
    entry_pos = clamp(entry_pos, bbox_start, bbox_end);
    int3 ijk = make_int3(interp_to_grid_on_axis(entry_pos.x, bbox_start.x, bbox_end.x, grid_size),
                         interp_to_grid_on_axis(entry_pos.y, bbox_start.y, bbox_end.y, grid_size),
                         interp_to_grid_on_axis(entry_pos.z, bbox_start.z, bbox_end.z, grid_size));
    float3 voxel_center = (make_float3(ijk.x + 0.5, ijk.y + 0.5, ijk.z + 0.5) / (float)grid_size) * (bbox_end - bbox_start) + bbox_start;
    float3 first_edge_coords = voxel_center + 0.5 * ((bbox_end - bbox_start) / (float)grid_size) * sign(ray_direction);

    int3 step = make_int3(sign(ray_direction));
    float3 delta = fabs(((bbox_end - bbox_start) / (float)grid_size) / ray_direction);
    float3 tmax = (first_edge_coords - ray_origin) / ray_direction;
    tmax.x = tmax.x == -INFINITY ? INFINITY : tmax.x;
    tmax.y = tmax.y == -INFINITY ? INFINITY : tmax.y;
    tmax.z = tmax.z == -INFINITY ? INFINITY : tmax.z;
    float current_t = entry_t;

    float3 rendered_color = make_float3(0.0, 0.0, 0.0);
    float density_integral = 0.0;
    bool is_inside = !isnan(entry_t) && entry_t < t_end;

    // while (is_inside) {
    for (int k = 0; k < 3 * grid_size - 2 && is_inside; ++k) {
        // figure out in which direction to walk in
        int3 mask;
        float new_t;
        if (tmax.y < tmax.z) {
            if (tmax.x < tmax.y) {
                new_t = tmax.x;
                mask = make_int3(1, 0, 0);
            } else {
                new_t = tmax.y;
                mask = make_int3(0, 1, 0);
            }
        } else { // tmax.z <= tmax.y
            if (tmax.x < tmax.z) {
                new_t = tmax.x;
                mask = make_int3(1, 0, 0);
            } else {
                new_t = tmax.z;
                mask = make_int3(0, 0, 1);
            }
        }
        float next_t = min(new_t, t_end);
        float delta_t = next_t - current_t;

        float density_integral_step;
        if constexpr (use_trilinear_interp) {
            // https://gemini.google.com/share/3151df37634f
            float mid_t = (current_t + new_t) * 0.5f;
            float3 offset_a = get_voxel_offset(ray_origin + current_t * ray_direction, ijk, bbox_start, bbox_end, grid_size);
            float3 offset_m = get_voxel_offset(ray_origin + mid_t * ray_direction, ijk, bbox_start, bbox_end, grid_size);
            float3 offset_b = get_voxel_offset(ray_origin + new_t * ray_direction, ijk, bbox_start, bbox_end, grid_size);

            float density_a = trilinear_interp(density_grid + gridpos_encode(ijk, grid_size) * 8, offset_a);
            float density_m = trilinear_interp(density_grid + gridpos_encode(ijk, grid_size) * 8, offset_m);
            float density_b = trilinear_interp(density_grid + gridpos_encode(ijk, grid_size) * 8, offset_b);

            density_integral_step = (delta_t / 6.0f) * (density_a + 4.0f * density_m + density_b);
        } else {
            float density = density_grid[gridpos_encode(ijk, grid_size)];
            density_integral_step = density * delta_t;
        }
        float3 color = eval_sh(color_grid + 3 * ((sh_deg + 1) * (sh_deg + 1)) * gridpos_encode(ijk, grid_size), ray_direction, sh_deg);
        rendered_color += exp(-density_integral) * (-expm1(-density_integral_step)) * color;
        density_integral += density_integral_step;

        ijk += mask * step;
        current_t = new_t;
        tmax += make_float3(mask) * delta;
        is_inside &=
            0 <= ijk.x && ijk.x < grid_size
            && 0 <= ijk.y && ijk.y < grid_size
            && 0 <= ijk.z && ijk.z < grid_size
            && current_t < t_end;
    }

    out_color[0] = rendered_color.x;
    out_color[1] = rendered_color.y;
    out_color[2] = rendered_color.z;
    *out_transmittance = exp(-density_integral);
}

template <bool use_trilinear_interp>
__global__ void render_kernel(int32_t n_batch, const float* ray_data,
                              const float* density_grid, const float* color_grid,
                              float* out_colors, float* out_transmittances,
                              float3 bbox_start, float3 bbox_end,
                              int32_t grid_size, int32_t sh_deg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // assert(i >= 0)
    if (i >= n_batch) return;

    float3 ray_origin = make_float3(ray_data[8*i + 0], ray_data[8*i + 1], ray_data[8*i + 2]);
    float3 ray_direction = make_float3(ray_data[8*i + 3], ray_data[8*i + 4], ray_data[8*i + 5]);
    float t_start = ray_data[8*i + 6];
    float t_end = ray_data[8*i + 7];
    float* out_color = out_colors + i * 3;
    float* out_transmittance = out_transmittances + i;

    render_ray_kernel<use_trilinear_interp>(ray_origin, ray_direction,
                                            t_start, t_end,
                                            density_grid, color_grid,
                                            out_color, out_transmittance,
                                            bbox_start, bbox_end,
                                            grid_size, sh_deg);
}

template <bool use_trilinear_interp>
ffi::Error render_within_interval_impl_cuda(cudaStream_t stream,
                                            size_t grid_size,
                                            size_t sh_deg,
                                            float bbox_start_x,
                                            float bbox_start_y,
                                            float bbox_start_z,
                                            float bbox_end_x,
                                            float bbox_end_y,
                                            float bbox_end_z,
                                            ffi::Buffer<ffi::F32> ray_data,
                                            ffi::Buffer<ffi::F32> density_grid,
                                            ffi::Buffer<ffi::F32> color_grid,
                                            ffi::ResultBuffer<ffi::F32> out_colors,
                                            ffi::ResultBuffer<ffi::F32> out_transmittances) {
    auto ray_data_dims = ray_data.dimensions();
    auto density_grid_dims = density_grid.dimensions();
    auto color_grid_dims = color_grid.dimensions();

    if constexpr (use_trilinear_interp) {
        assert(density_grid.element_count() == grid_size * grid_size * grid_size * 8);
    } else {
        assert(density_grid.element_count() == grid_size * grid_size * grid_size);
    }
    assert(color_grid.element_count() == grid_size * grid_size * grid_size * 3 * ((sh_deg + 1) * (sh_deg + 1)));

    assert(ray_data_dims.back() == 8);
    auto n_batch = ray_data.element_count() / 8;

    float3 bbox_start = make_float3(bbox_start_x, bbox_start_y, bbox_start_z);
    float3 bbox_end = make_float3(bbox_end_x, bbox_end_y, bbox_end_z);

    auto block_size = 256;
    auto n_blocks = (n_batch + block_size - 1) / block_size;
    render_kernel<use_trilinear_interp><<<n_blocks,block_size,0,stream>>>(n_batch,
                                                                          ray_data.typed_data(),
                                                                          density_grid.typed_data(), color_grid.typed_data(),
                                                                          out_colors->typed_data(), out_transmittances->typed_data(),
                                                                          bbox_start, bbox_end,
                                                                          grid_size, sh_deg);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    render_within_interval_cuda, render_within_interval_impl_cuda<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("grid_size")
        .Attr<int64_t>("sh_deg")
        .Attr<double>("bbox_start_x")
        .Attr<double>("bbox_start_y")
        .Attr<double>("bbox_start_z")
        .Attr<double>("bbox_end_x")
        .Attr<double>("bbox_end_y")
        .Attr<double>("bbox_end_z")
        .Arg<ffi::Buffer<ffi::F32>>() // ray_data
        .Arg<ffi::Buffer<ffi::F32>>() // density_grid
        .Arg<ffi::Buffer<ffi::F32>>() // color_grid
        .Ret<ffi::Buffer<ffi::F32>>() // out_colors
        .Ret<ffi::Buffer<ffi::F32>>() // out_transmittances
);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    render_within_interval_trilerp_cuda, render_within_interval_impl_cuda<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("grid_size")
        .Attr<int64_t>("sh_deg")
        .Attr<double>("bbox_start_x")
        .Attr<double>("bbox_start_y")
        .Attr<double>("bbox_start_z")
        .Attr<double>("bbox_end_x")
        .Attr<double>("bbox_end_y")
        .Attr<double>("bbox_end_z")
        .Arg<ffi::Buffer<ffi::F32>>() // ray_data
        .Arg<ffi::Buffer<ffi::F32>>() // density_grid
        .Arg<ffi::Buffer<ffi::F32>>() // color_grid
        .Ret<ffi::Buffer<ffi::F32>>() // out_colors
        .Ret<ffi::Buffer<ffi::F32>>() // out_transmittances
);
