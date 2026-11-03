/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit), and
 * GGML_TYPE_TURBO4_0 (4-bit) for use as --cache-type-k turbo2|turbo3|turbo4
 * --cache-type-v turbo2|turbo3|turbo4 in llama-server.
 *
 * These are the CPU reference quantize/dequantize implementations. The GPU
 * fast paths (with the O(d log d) Walsh-Hadamard rotation applied via the
 * GGML_OP_TURBO_WHT graph op) live in ggml/src/ggml-cuda/turbo-quant.cu.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

/* ---------- TURBO3_0: 2-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TURBO3;

        // Compute L2 norm of the block
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) {
            norm += xb[j] * xb[j];
        }
        norm = sqrtf(norm);
        y[i].norm = GGML_FP32_TO_FP16(norm);

        memset(y[i].qs, 0, sizeof(y[i].qs));
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm < 1e-10f) continue;

        const float inv_norm = 1.0f / norm;

        for (int j = 0; j < QK_TURBO3; j++) {
            const float val = xb[j] * inv_norm;

            // Find nearest centroid (exhaustive search over 8 centroids)
            int best_idx = 0;
            float best_dist = fabsf(val - CENTROIDS_3BIT[0]);
            for (int c = 1; c < 8; c++) {
                float dist = fabsf(val - CENTROIDS_3BIT[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = c;
                }
            }

            // Pack: low 2 bits into qs, high bit into signs
            uint8_t low2 = best_idx & 0x3;
            uint8_t hi1  = (best_idx >> 2) & 0x1;

            y[i].qs[j / 4]    |= (low2 << ((j % 4) * 2));
            y[i].signs[j / 8] |= (hi1  << (j % 8));
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — GPU kernel handles dequant on device; this CPU path only needs to be correct.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Rotate */
        float rotated[TURBO_D];
        matvec(turbo_rotation, normalized, rotated, d);

        /* Step 3: 3-bit quantization */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }


        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);

        /* Pack */
        y[block].norm  = GGML_FP32_TO_FP16(norm);

        /* Pack 3-bit indices: 8 indices per 3 bytes */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            /* Write up to 2 bytes (3 bits might span a byte boundary) */
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }

        /* Pack 1-bit QJL signs */
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 3-bit indices */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        /* Unpack signs */
        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        /* PolarQuant dequant */
        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        /* QJL dequant */
        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        /* Combine */
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_0: 2-bit PolarQuant (no QJL) ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO2; j++) norm += x[i*QK_TURBO2 + j] * x[i*QK_TURBO2 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_TURBO2 / 4);
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO5_0 / TURBO6_0: 5/6-bit PolarQuant (no QJL) ----------
 *
 * Codebooks: Lloyd-Max centroids for N(0, 1/128), from TheTom/llama-cpp-turboquant PR #381.
 * Mirrored in ggml-cuda/turbo56.cuh. Like turbo2/turbo3 here, this CPU reference is the codec
 * only: it does not apply the WHT. On the GPU the rotation happens in SET_ROWS
 * (ggml-cuda/turbo-quant.cu), so a dequantized row is in the rotated domain there as well.
 * Each 128-value block stores the norm corrected by the reconstruction norm, so a dequantized
 * block has the L2 norm of the input block.
 */

static const float TURBO6_CENTROIDS[64] = {
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

static const float TURBO5_CENTROIDS[32] = {
    -0.288236f, -0.237892f, -0.204892f, -0.179348f,
    -0.158003f, -0.139353f, -0.122569f, -0.107141f,
    -0.092730f, -0.079097f, -0.066064f, -0.053491f,
    -0.041269f, -0.029304f, -0.017515f, -0.005827f,
     0.005827f,  0.017515f,  0.029304f,  0.041269f,
     0.053491f,  0.066064f,  0.079097f,  0.092730f,
     0.107141f,  0.122569f,  0.139353f,  0.158003f,
     0.179348f,  0.204892f,  0.237892f,  0.288236f
};

/* nearest centroid: binary search on the midpoints of a sorted codebook */
static int turbo_nearest_centroid_sorted(const float * c, int n, float val) {
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (val < 0.5f*(c[mid] + c[mid + 1])) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

/* quantize one 128-value block to codebook indices, returns the corrected norm */
static float turbo56_quantize_block(const float * src, const float * c, int n_c, uint8_t * idx) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    const float norm     = sqrtf(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        idx[j] = (uint8_t) turbo_nearest_centroid_sorted(c, n_c, src[j] * inv_norm);
        recon_sq += c[idx[j]] * c[idx[j]];
    }
    const float recon_norm = sqrtf(recon_sq);
    return recon_norm > 1e-10f ? norm / recon_norm : norm;
}

void quantize_row_turbo6_0_ref(const float * GGML_RESTRICT x, block_turbo6_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO6 == 0);
    const int64_t nb = k / QK_TURBO6;
    for (int64_t ib = 0; ib < nb; ib++) {
        uint8_t idx[QK_TURBO6];
        y[ib].norm = GGML_FP32_TO_FP16(turbo56_quantize_block(x + ib*QK_TURBO6, TURBO6_CENTROIDS, 64, idx));
        memset(y[ib].qs, 0, sizeof(y[ib].qs));
        memset(y[ib].qh, 0, sizeof(y[ib].qh));
        for (int j = 0; j < QK_TURBO6; j++) {
            y[ib].qs[j/2] |= (uint8_t) ((idx[j] & 0xF) << ((j % 2) * 4));
            y[ib].qh[j/4] |= (uint8_t) (((idx[j] >> 4) & 0x3) << ((j % 4) * 2));
        }
    }
}

void dequantize_row_turbo6_0(const block_turbo6_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO6 == 0);
    const int64_t nb = k / QK_TURBO6;
    for (int64_t ib = 0; ib < nb; ib++) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int j = 0; j < QK_TURBO6; j++) {
            const int lo = (x[ib].qs[j/2] >> ((j % 2) * 4)) & 0xF;
            const int hi = (x[ib].qh[j/4] >> ((j % 4) * 2)) & 0x3;
            y[ib*QK_TURBO6 + j] = TURBO6_CENTROIDS[lo | (hi << 4)] * norm;
        }
    }
}

size_t quantize_turbo6_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO6 == 0);

    const size_t row_size = (n_per_row / QK_TURBO6) * sizeof(block_turbo6_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo6_0_ref(src + row * n_per_row, (block_turbo6_0 *) ((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

void quantize_row_turbo5_0_ref(const float * GGML_RESTRICT x, block_turbo5_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO5 == 0);
    const int64_t nb = k / QK_TURBO5;
    for (int64_t ib = 0; ib < nb; ib++) {
        uint8_t idx[QK_TURBO5];
        y[ib].norm = GGML_FP32_TO_FP16(turbo56_quantize_block(x + ib*QK_TURBO5, TURBO5_CENTROIDS, 32, idx));
        memset(y[ib].qs, 0, sizeof(y[ib].qs));
        memset(y[ib].qh, 0, sizeof(y[ib].qh));
        for (int j = 0; j < QK_TURBO5; j++) {
            const uint8_t neg = idx[j] < 16;
            const uint8_t mag = neg ? (uint8_t) (15 - idx[j]) : (uint8_t) (idx[j] - 16);
            y[ib].qs[j/2] |= (uint8_t) (mag << ((j % 2) * 4));
            y[ib].qh[j/8] |= (uint8_t) (neg << (j % 8));
        }
    }
}

void dequantize_row_turbo5_0(const block_turbo5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO5 == 0);
    const int64_t nb = k / QK_TURBO5;
    for (int64_t ib = 0; ib < nb; ib++) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int j = 0; j < QK_TURBO5; j++) {
            const int mag = (x[ib].qs[j/2] >> ((j % 2) * 4)) & 0xF;
            const int neg = (x[ib].qh[j/8] >> (j % 8)) & 0x1;
            y[ib*QK_TURBO5 + j] = TURBO5_CENTROIDS[neg ? 15 - mag : 16 + mag] * norm;
        }
    }
}

size_t quantize_turbo5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO5 == 0);

    const size_t row_size = (n_per_row / QK_TURBO5) * sizeof(block_turbo5_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo5_0_ref(src + row * n_per_row, (block_turbo5_0 *) ((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}
