// TurboQuant turbo5 / turbo6 KV cache: device codebooks and decode helpers.
// Ported from TheTom/llama-cpp-turboquant PR #381. Mirrors ggml-turbo-quant.c.
#pragma once

#include "common.cuh"

// 64 Lloyd-Max centroids for N(0, 1/128), sorted ascending.
static __constant__ float TURBO6_CENTROIDS[64] = {
    -0.330935f, -0.286417f, -0.257865f, -0.236198f,
    -0.218435f, -0.203203f, -0.189753f, -0.177626f,
    -0.166522f, -0.156230f, -0.146600f, -0.137517f,
    -0.128895f, -0.120663f, -0.112765f, -0.105157f,
    -0.097801f, -0.090663f, -0.083717f, -0.076940f,
    -0.070310f, -0.063809f, -0.057422f, -0.051133f,
    -0.044929f, -0.038798f, -0.032729f, -0.026710f,
    -0.020733f, -0.014787f, -0.008863f, -0.002953f,
     0.002953f,  0.008863f,  0.014787f,  0.020733f,
     0.026710f,  0.032729f,  0.038798f,  0.044929f,
     0.051133f,  0.057422f,  0.063809f,  0.070310f,
     0.076940f,  0.083717f,  0.090663f,  0.097801f,
     0.105157f,  0.112765f,  0.120663f,  0.128895f,
     0.137517f,  0.146600f,  0.156230f,  0.166522f,
     0.177626f,  0.189753f,  0.203203f,  0.218435f,
     0.236198f,  0.257865f,  0.286417f,  0.330935f
};

// 32 antisymmetric Lloyd-Max centroids for N(0, 1/128), sorted ascending (c[31-i] == -c[i]).
static __constant__ float TURBO5_CENTROIDS[32] = {
    -0.288236f, -0.237892f, -0.204892f, -0.179348f,
    -0.158003f, -0.139353f, -0.122569f, -0.107141f,
    -0.092730f, -0.079097f, -0.066064f, -0.053491f,
    -0.041269f, -0.029304f, -0.017515f, -0.005827f,
     0.005827f,  0.017515f,  0.029304f,  0.041269f,
     0.053491f,  0.066064f,  0.079097f,  0.092730f,
     0.107141f,  0.122569f,  0.139353f,  0.158003f,
     0.179348f,  0.204892f,  0.237892f,  0.288236f
};

// turbo5 storage is sign-magnitude: magnitude index m and sign bit -> centroid code (neg ? 15-m : 16+m)
static __device__ __forceinline__ int turbo5_sm_to_code(const int mag, const int neg) {
    return neg ? 15 - mag : 16 + mag;
}

// 6-bit code of value j (0..127) of a turbo6 block
static __device__ __forceinline__ int turbo6_code(const block_turbo6_0 * x, const int j) {
    const int lo = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    const int hi = (x->qh[j / 4] >> ((j % 4) * 2)) & 0x3;
    return lo | (hi << 4);
}

// 5-bit code of value j (0..127) of a turbo5 block
static __device__ __forceinline__ int turbo5_code(const block_turbo5_0 * x, const int j) {
    const int mag = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    const int neg = (x->qh[j / 8] >> (j % 8)) & 0x1;
    return turbo5_sm_to_code(mag, neg);
}
