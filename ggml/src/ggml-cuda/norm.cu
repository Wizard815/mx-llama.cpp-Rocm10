#include "norm.cuh"
#include <cstdint>

template <int block_size, bool clamped = false>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float2 s_sum2[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z;; sample += gridDim.z) {
        for (int channel = blockIdx.y;; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float2 mean_var = make_float2(0.0f, 0.0f);

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                mean_var.x += xi;
                mean_var.y += xi * xi;
            }

            // sum up partial sums
            mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

            const float mean = mean_var.x / ncols;
            const float var = mean_var.y / ncols - mean * mean;
            const float inv_std = rsqrtf(var + eps);

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = (xc[col] - mean) * inv_std;
            }

            // unclamped grid: exactly one channel/sample per block, the loops fold away
            if constexpr (!clamped) {
                break;
            } else {
                if (channel + (int) gridDim.y >= nchannels) {
                    break;
                }
                if constexpr (block_size > WARP_SIZE) {
                    // sync is needed as we reuse s_sum2 across block_reduce invocations, see #26385
                    __syncthreads();
                }
            }
        }
        if constexpr (!clamped) {
            break;
        } else {
            if (sample + (int) gridDim.z >= nsamples) {
                break;
            }
            if constexpr (block_size > WARP_SIZE) {
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// n_resid > 0: x is the first operand of a residual ADD chain, the sum x + resid0 (+ resid1) is computed while loading,
// written to resid_sum and normalized; every element is read and written by the same thread, so outputs may alias inputs exactly.
// do_pair: blockIdx.z >= nsamples runs a second same-shape norm on x1/dst1 (the GDN q and k l2 norms in one launch).
template <int block_size, int vpt, bool do_vec, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool clamped = false,
          int n_resid = 0, bool do_pair = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int     nchannels,
                                    const int     nsamples,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f,
                                    const float * resid0               = nullptr,
                                    const float * resid1               = nullptr,
                                    float *       resid_sum            = nullptr,
                                    const float * x1                   = nullptr,
                                    float *       dst1                 = nullptr) {
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");
    static_assert(n_resid >= 0 && n_resid <= 2, "at most two residual addends");
    static_assert(!do_pair || (!clamped && n_resid == 0), "the norm pair needs an unclamped grid");

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    int sample0 = blockIdx.z;
    if constexpr (do_pair) {
        if (sample0 >= nsamples) {
            x        = x1;
            dst      = dst1;
            sample0 -= nsamples;
        }
    }
    GGML_UNUSED(x1);
    GGML_UNUSED(dst1);

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = sample0;; sample += gridDim.z) {
        for (int channel = blockIdx.y;; channel += gridDim.y) {
            const int64_t x_off = sample*stride_sample + channel*stride_channel + row*stride_row;
            const float * xc   = x   + x_off;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            // residual operands and the sum share x's layout (contiguous, checked on the host)
            [[maybe_unused]] const float * r0c = nullptr;
            [[maybe_unused]] const float * r1c = nullptr;
            [[maybe_unused]] float *       sc  = nullptr;
            if constexpr (n_resid > 0) {
                r0c = resid0    + x_off;
                sc  = resid_sum + x_off;
            }
            if constexpr (n_resid > 1) {
                r1c = resid1 + x_off;
            }

            [[maybe_unused]] const float * mulc = nullptr;
            if constexpr (do_multiply) {
                const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
                const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
                const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
                mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
            }

            [[maybe_unused]] const float * addc = nullptr;
            if constexpr (do_add) {
                const int add_row     = fastmodulo(row, add_nrows_packed);
                const int add_channel = fastmodulo(channel, add_nchannels_packed);
                const int add_sample  = fastmodulo(sample, add_nsamples_packed);
                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
            }

            if constexpr (do_vec) {
                const int      ncols4 = ncols / 4;
                const float4 * x4     = (const float4 *) xc;
                const float4 * mul4   = (const float4 *) mulc;
                const float4 * add4   = (const float4 *) addc;
                float4 *       dst4   = (float4 *) dstc;

                float4 xs[vpt];
                float  tmp = 0.0f;

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        xs[k] = x4[i];
                        if constexpr (n_resid > 0) {
                            const float4 r = ((const float4 *) r0c)[i];
                            xs[k].x += r.x;
                            xs[k].y += r.y;
                            xs[k].z += r.z;
                            xs[k].w += r.w;
                        }
                        if constexpr (n_resid > 1) {
                            const float4 r = ((const float4 *) r1c)[i];
                            xs[k].x += r.x;
                            xs[k].y += r.y;
                            xs[k].z += r.z;
                            xs[k].w += r.w;
                        }
                        if constexpr (n_resid > 0) {
                            ((float4 *) sc)[i] = xs[k];
                        }
                        tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
                    }
                }
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
                        if constexpr (do_multiply) {
                            const float4 m = mul4[i];
                            r.x *= m.x;
                            r.y *= m.y;
                            r.z *= m.z;
                            r.w *= m.w;
                        }
                        if constexpr (do_add) {
                            const float4 a = add4[i];
                            r.x += a.x;
                            r.y += a.y;
                            r.z += a.z;
                            r.w += a.w;
                        }
                        if constexpr (do_scale) {
                            r.x *= scale_out;
                            r.y *= scale_out;
                            r.z *= scale_out;
                            r.w *= scale_out;
                        }
                        dst4[i] = r;
                    }
                }
            } else {
                float tmp = 0.0f; // partial sum for thread in warp

                for (int col = tid; col < ncols; col += block_size) {
                    float xi = xc[col];
                    if constexpr (n_resid > 0) {
                        xi += r0c[col];
                    }
                    if constexpr (n_resid > 1) {
                        xi += r1c[col];
                    }
                    if constexpr (n_resid > 0) {
                        sc[col] = xi;
                    }
                    tmp += xi * xi;
                }

                // sum up partial sums
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

                const float mean = tmp / ncols;
                const float scale = rsqrtf(mean + eps);

                // with a residual sum, normalize the sum this thread just wrote rather than x
                if constexpr (n_resid > 0) {
                    xc = sc;
                }

                for (int col = tid; col < ncols; col += block_size) {
                    if constexpr (do_multiply && do_add) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        const int add_col = fastmodulo(col, add_ncols_packed);
                        dstc[col]         = scale * xc[col] * mulc[mul_col] + addc[add_col];
                    } else if constexpr (do_multiply) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        dstc[col]         = scale * xc[col] * mulc[mul_col];
                    } else if constexpr (do_scale) {
                        dstc[col] = scale_out * (scale * xc[col]);
                    } else {
                        dstc[col] = scale * xc[col];
                    }
                }
            }

            // unclamped grid: exactly one channel/sample per block, the loops fold away
            if constexpr (!clamped) {
                break;
            } else {
                if (channel + (int) gridDim.y >= nchannels) {
                    break;
                }
                if constexpr (block_size > WARP_SIZE) {
                    // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                    __syncthreads();
                }
            }
        }
        if constexpr (!clamped) {
            break;
        } else {
            if (sample + (int) gridDim.z >= nsamples) {
                break;
            }
            if constexpr (block_size > WARP_SIZE) {
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size, bool clamped = false>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z;; sample += gridDim.z) {
        for (int channel = blockIdx.y;; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float tmp = 0.0f; // partial sum for thread in warp

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            // sum up partial sums
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
            const float scale = rsqrtf(fmaxf(tmp, eps * eps));

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = scale * xc[col];
            }

            // unclamped grid: exactly one channel/sample per block, the loops fold away
            if constexpr (!clamped) {
                break;
            } else {
                if (channel + (int) gridDim.y >= nchannels) {
                    break;
                }
                if constexpr (block_size > WARP_SIZE) {
                    // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                    __syncthreads();
                }
            }
        }
        if constexpr (!clamped) {
            break;
        } else {
            if (sample + (int) gridDim.z >= nsamples) {
                break;
            }
            if constexpr (block_size > WARP_SIZE) {
                __syncthreads();
            }
        }
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    // clamped grids need the channel/sample loops in the kernel, the common case must not pay for them
    const bool clamped_grid = nchannels > UINT16_MAX || nsamples > UINT16_MAX;
    auto launch = [&](auto clamped_tag) {
    constexpr bool clamped = decltype(clamped_tag)::value;
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE, clamped><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024, clamped><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
    };
    if (clamped_grid) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

#define RMS_NORM_LAUNCH(BLOCK_SIZE, VPT, VEC) \
    do { \
        const dim3 block_dims(BLOCK_SIZE, 1, 1); \
        const ggml_cuda_kernel_launch_params launch_params{ blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream }; \
        if (clamped_grid) { \
            ggml_cuda_kernel_launch(rms_norm_f32<BLOCK_SIZE, VPT, VEC, do_multiply, do_add, do_scale, true, n_resid, false>, launch_params, \
                x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, \
                mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \
                add, add_stride_row, add_stride_channel, add_stride_sample, \
                add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out, \
                resid0, resid1, resid_sum, x1, dst1); \
        } else { \
            ggml_cuda_kernel_launch(rms_norm_f32<BLOCK_SIZE, VPT, VEC, do_multiply, do_add, do_scale, false, n_resid, do_pair>, launch_params, \
                x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, \
                mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \
                add, add_stride_row, add_stride_channel, add_stride_sample, \
                add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out, \
                resid0, resid1, resid_sum, x1, dst1); \
        } \
    } while (0)

template <bool do_multiply, bool do_add, bool do_scale, int n_resid = 0, bool do_pair = false>
static void rms_norm_f32_dispatch(
        const dim3     blocks_num,
        const ggml_tensor * x_t,   float * dst,   const int ncols, const int nchannels, const int nsamples,
        const int64_t  stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const ggml_tensor * mul_t, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3    mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const ggml_tensor * add_t, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3    add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
        cudaStream_t   stream, const float scale_out = 1.0f,
        const ggml_tensor * resid0_t = nullptr, const ggml_tensor * resid1_t = nullptr, const ggml_tensor * resid_sum_t = nullptr,
        const ggml_tensor * x1_t = nullptr, float * dst1 = nullptr) {
    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    const float * x   = (const float *) x_t->data;
    const float * mul = mul_t ? (const float *) mul_t->data : nullptr;
    const float * add = add_t ? (const float *) add_t->data : nullptr;

    const float * resid0    = resid0_t    ? (const float *) resid0_t->data    : nullptr;
    const float * resid1    = resid1_t    ? (const float *) resid1_t->data    : nullptr;
    float *       resid_sum = resid_sum_t ? (float *)       resid_sum_t->data : nullptr;
    const float * x1        = x1_t        ? (const float *) x1_t->data        : nullptr;

    // clamped grids need the channel/sample loops in the kernel, the common case must not pay for them
    const bool clamped_grid = nchannels > UINT16_MAX || nsamples > UINT16_MAX;
    GGML_ASSERT(!do_pair || !clamped_grid);

    // dst is written at row * ncols, i.e. always at a multiple of ncols, so only its base needs checking here
    bool vec = ncols % 4 == 0 && ncols > 0 && ((uintptr_t) dst & 15) == 0;
    vec = vec && ggml_cuda_is_aligned(x_t, 16);
    if constexpr (do_multiply) {
        vec = vec && mul_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(mul_t, 16);
    }
    if constexpr (do_add) {
        vec = vec && add_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(add_t, 16);
    }
    if constexpr (n_resid > 0) {
        vec = vec && ggml_cuda_is_aligned(resid0_t, 16) && ggml_cuda_is_aligned(resid_sum_t, 16);
    }
    if constexpr (n_resid > 1) {
        vec = vec && ggml_cuda_is_aligned(resid1_t, 16);
    }
    if constexpr (do_pair) {
        vec = vec && ggml_cuda_is_aligned(x1_t, 16) && ((uintptr_t) dst1 & 15) == 0;
    }

    const int n4 = ncols / 4;
    if (vec) {
        if (n4 <= 32) {
            RMS_NORM_LAUNCH(32,   1, true);
        } else if (n4 <= 128) {
            RMS_NORM_LAUNCH(128,  1, true);
        } else if (n4 <= 256) {
            RMS_NORM_LAUNCH(256,  1, true);
        } else if (n4 <= 512) {
            RMS_NORM_LAUNCH(512,  1, true);
        } else if (n4 <= 1024) {
            RMS_NORM_LAUNCH(1024, 1, true);
        } else if (n4 <= 4096) {
            RMS_NORM_LAUNCH(1024, 4, true);
        } else {
            RMS_NORM_LAUNCH(1024, 1, false);
        }
    } else if (ncols < 1024) {
        RMS_NORM_LAUNCH(256,  1, false);
    } else {
        RMS_NORM_LAUNCH(1024, 1, false);
    }
}
#undef RMS_NORM_LAUNCH

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const ggml_tensor * x_t, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    rms_norm_f32_dispatch</*do_multiply*/false, /*do_add*/false, do_scale>(
        blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        stream, scale_out);
}

static void rms_norm_mul_f32_cuda(const ggml_tensor * x_t,
                                  const ggml_tensor * mul_t,
                                  const ggml_tensor * add_t,
                                  float             * dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (mul_t == nullptr) {
        rms_norm_f32_cuda(x_t, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
    const uint3 zero_packed          = make_uint3(0, 0, 0);

    if (add_t == nullptr) {
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream);
    } else {
        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/true, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            add_t, add_stride_row, add_stride_channel, add_stride_sample,
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, stream);
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    // clamped grids need the channel/sample loops in the kernel, the common case must not pay for them
    const bool clamped_grid = nchannels > UINT16_MAX || nsamples > UINT16_MAX;
    auto launch = [&](auto clamped_tag) {
    constexpr bool clamped = decltype(clamped_tag)::value;
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE, clamped>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024, clamped>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
    };
    if (clamped_grid) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src,add_src,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

// [data, data + nbytes) of a and b intersect
static bool rms_norm_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// same memory with the same (contiguous) layout, or no common byte
static bool rms_norm_exact_or_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    if (!rms_norm_ranges_overlap(a, b)) {
        return true;
    }
    return a->data == b->data && ggml_nbytes(a) == ggml_nbytes(b) && ggml_is_contiguous(a) && ggml_is_contiguous(b);
}

bool ggml_cuda_op_add_rms_norm_mul_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add0, ggml_tensor * add1,
                                         ggml_tensor * norm, ggml_tensor * mul_tensor) {
    ggml_tensor * sum = add1 ? add1 : add0;

    const ggml_tensor * a  = add0->src[0];
    const ggml_tensor * b0 = add0->src[1];
    const ggml_tensor * b1 = nullptr;
    if (add1) {
        if (add1->src[0] == add0) {
            b1 = add1->src[1];
        } else if (add1->src[1] == add0) {
            b1 = add1->src[0];
        } else {
            return false;
        }
    }
    if (norm->src[0] != sum) {
        return false;
    }

    const ggml_tensor * w = nullptr;
    if (mul_tensor->src[0] == norm) {
        w = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == norm) {
        w = mul_tensor->src[0];
    } else {
        return false;
    }

    // every addend, the intermediate sum and the final sum share one contiguous F32 layout
    const ggml_tensor * same[] = { a, b0, add0, sum, norm, mul_tensor };
    for (const ggml_tensor * t : same) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || !ggml_are_same_shape(t, sum)) {
            return false;
        }
    }
    if (b1 && (b1->type != GGML_TYPE_F32 || !ggml_is_contiguous(b1) || !ggml_are_same_shape(b1, sum))) {
        return false;
    }
    if (w->type != GGML_TYPE_F32 || w->nb[0] != sizeof(float)) {
        return false;
    }

    // the sum and the normed output are written, everything else is read in the same launch
    const ggml_tensor * outs[] = { sum, mul_tensor };
    for (const ggml_tensor * o : outs) {
        if (!rms_norm_exact_or_disjoint(o, a) || !rms_norm_exact_or_disjoint(o, b0) ||
                (b1 && !rms_norm_exact_or_disjoint(o, b1)) || rms_norm_ranges_overlap(o, w)) {
            return false;
        }
    }
    if (rms_norm_ranges_overlap(sum, mul_tensor)) {
        return false;
    }
    // the unfused RMS_NORM picks its vectorized path from the sum's alignment; take the same path or leave it alone
    const bool sum_aligned = ggml_cuda_is_aligned(sum, 16);
    if (ggml_cuda_is_aligned(a, 16) != sum_aligned || ggml_cuda_is_aligned(b0, 16) != sum_aligned ||
            (b1 && ggml_cuda_is_aligned(b1, 16) != sum_aligned)) {
        return false;
    }

    float eps = 0.0f;
    memcpy(&eps, norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = sum->ne[0];
    const int64_t ne01 = sum->ne[1];
    const int64_t ne02 = sum->ne[2];
    const int64_t ne03 = sum->ne[3];

    const int64_t s01 = a->nb[1] / sizeof(float);
    const int64_t s02 = a->nb[2] / sizeof(float);
    const int64_t s03 = a->nb[3] / sizeof(float);

    const int64_t mul_s01 = w->nb[1] / sizeof(float);
    const int64_t mul_s02 = w->nb[2] / sizeof(float);
    const int64_t mul_s03 = w->nb[3] / sizeof(float);

    const uint3 zero_packed = make_uint3(0, 0, 0);
    const dim3  blocks_num(ne01, MIN(ne02, UINT16_MAX), MIN(ne03, UINT16_MAX));

#define ADD_RMS_NORM_DISPATCH(N_RESID)                                                                              \
    rms_norm_f32_dispatch</*do_multiply*/ true, /*do_add*/ false, /*do_scale*/ false, N_RESID>(                    \
        blocks_num, a, (float *) mul_tensor->data, ne00, ne02, ne03, s01, s02, s03, eps,                            \
        w, mul_s01, mul_s02, mul_s03,                                                                               \
        init_fastdiv_values(w->ne[0]), init_fastdiv_values(w->ne[1]),                                               \
        init_fastdiv_values(w->ne[2]), init_fastdiv_values(w->ne[3]),                                               \
        nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, ctx.stream(), 1.0f,                  \
        b0, b1, sum)

    if (b1) {
        ADD_RMS_NORM_DISPATCH(2);
    } else {
        ADD_RMS_NORM_DISPATCH(1);
    }
#undef ADD_RMS_NORM_DISPATCH

    return true;
}

bool ggml_cuda_op_rms_norm_scale_pair_fused(ggml_backend_cuda_context & ctx, ggml_tensor * norm0, ggml_tensor * scale0,
                                            ggml_tensor * norm1, ggml_tensor * scale1) {
    const ggml_tensor * x0 = norm0->src[0];
    const ggml_tensor * x1 = norm1->src[0];

    if (scale0->src[0] != norm0 || scale1->src[0] != norm1) {
        return false;
    }
    if (x0->type != GGML_TYPE_F32 || x1->type != GGML_TYPE_F32 || norm0->type != GGML_TYPE_F32 || norm1->type != GGML_TYPE_F32 ||
            scale0->type != GGML_TYPE_F32 || scale1->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(x0, x1) || !ggml_are_same_shape(x0, scale0) || !ggml_are_same_shape(x0, scale1) ||
            !ggml_is_contiguous(scale0) || !ggml_is_contiguous(scale1)) {
        return false;
    }
    if (x0->nb[0] != sizeof(float) || x1->nb[0] != sizeof(float) ||
            x0->nb[1] != x1->nb[1] || x0->nb[2] != x1->nb[2] || x0->nb[3] != x1->nb[3]) {
        return false;
    }
    // same eps, same scale and bias
    if (memcmp(norm0->op_params, norm1->op_params, sizeof(float)) != 0 ||
            memcmp(scale0->op_params, scale1->op_params, 2*sizeof(float)) != 0) {
        return false;
    }
    if (x0->ne[2] > UINT16_MAX || 2*x0->ne[3] > UINT16_MAX) {
        return false;
    }
    // the unfused norms pick their vectorized path per tensor; both must take the same one
    if (ggml_cuda_is_aligned(x0, 16) != ggml_cuda_is_aligned(x1, 16) ||
            ggml_cuda_is_aligned(scale0, 16) != ggml_cuda_is_aligned(scale1, 16)) {
        return false;
    }
    // both pairs run at once: no output may touch the other output or either input
    if (rms_norm_ranges_overlap(scale0, scale1) ||
            rms_norm_ranges_overlap(scale0, x0) || rms_norm_ranges_overlap(scale0, x1) ||
            rms_norm_ranges_overlap(scale1, x0) || rms_norm_ranges_overlap(scale1, x1)) {
        return false;
    }

    float eps = 0.0f;
    memcpy(&eps, norm0->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale = 1.0f;
    memcpy(&scale, (const float *) scale0->op_params + 0, sizeof(float));

    const int64_t s01 = x0->nb[1] / sizeof(float);
    const int64_t s02 = x0->nb[2] / sizeof(float);
    const int64_t s03 = x0->nb[3] / sizeof(float);

    const uint3 zero_packed = make_uint3(0, 0, 0);
    const dim3  blocks_num(x0->ne[1], x0->ne[2], 2*x0->ne[3]);

    rms_norm_f32_dispatch</*do_multiply*/ false, /*do_add*/ false, /*do_scale*/ true, /*n_resid*/ 0, /*do_pair*/ true>(
        blocks_num, x0, (float *) scale0->data, x0->ne[0], x0->ne[2], x0->ne[3], s01, s02, s03, eps,
        nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed,
        nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, ctx.stream(), scale,
        nullptr, nullptr, nullptr, x1, (float *) scale1->data);

    return true;
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
