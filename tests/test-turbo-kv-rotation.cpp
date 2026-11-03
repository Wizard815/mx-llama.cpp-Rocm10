// TurboQuant KV cache rotation check (CPU reference).
//
// A turbo K cache holds R_K k and a turbo V cache holds R_V v (per 128-value group, with K and V
// using different FWHT sign vectors, see ggml-cuda/turbo-quant.cu). For attention to be exact:
//   - Q must be rotated with R_K iff K is turbo:   <R_K q, R_K k> = <q, k>
//   - the output must be rotated with R_V^T iff V is turbo:   sum_i p_i R_V v_i = R_V o
// This builds the same graph as llama-graph.cpp (GGML_OP_TURBO_WHT + FLASH_ATTN_EXT) for each
// K/V type pair, with the cache contents rotated the way the GPU SET_ROWS kernels store them,
// and compares against an f16 K/V reference. It also checks that gating the output rotation on
// the K type (the previous rule) breaks the mixed pairs.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const int D     = 128; // head size, one rotation group
static const int N_KV  = 256;
static const int N_Q   = 4;
static const int N_H   = 4;   // query heads
static const int N_HKV = 2;   // kv heads (GQA 2)

static bool is_turbo(ggml_type t) {
    return t == GGML_TYPE_TURBO5_0 || t == GGML_TYPE_TURBO6_0;
}

// run y = turbo_wht(x, dir) for n rows of 128 on the CPU backend
static std::vector<float> run_wht(const std::vector<float> & x, int dir) {
    ggml_init_params ip = { 64*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, (int64_t) x.size()/128);
    memcpy(a->data, x.data(), x.size()*sizeof(float));
    ggml_tensor * y = ggml_turbo_wht(ctx, a, dir);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 2);
    std::vector<float> out(x.size());
    memcpy(out.data(), y->data, x.size()*sizeof(float));
    ggml_free(ctx);
    return out;
}

// fill a tensor of type t from f32 data (rows of ne0)
static void set_data(ggml_tensor * t, const std::vector<float> & x) {
    if (t->type == GGML_TYPE_F32) {
        memcpy(t->data, x.data(), x.size()*sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(x.data(), (ggml_fp16_t *) t->data, (int64_t) x.size());
    } else {
        ggml_quantize_chunk(t->type, x.data(), t->data, 0, (int64_t) x.size()/t->ne[0], t->ne[0], nullptr);
    }
}

// attention output [D, N_H, N_Q] for the given cache types and rotation gating
static std::vector<float> attn(const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
                               ggml_type tk, ggml_type tv, bool rot_q, bool inv_out) {
    ggml_init_params ip = { 256*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_Q, N_H);
    ggml_tensor * K = ggml_new_tensor_3d(ctx, tk, D, N_KV, N_HKV);
    ggml_tensor * V = ggml_new_tensor_3d(ctx, tv, D, N_KV, N_HKV);
    ggml_tensor * M = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N_KV, N_Q);
    memcpy(Q->data, q.data(), q.size()*sizeof(float));
    set_data(K, k);
    set_data(V, v);
    std::vector<float> zeros(ggml_nelements(M), 0.0f);
    ggml_fp32_to_fp16_row(zeros.data(), (ggml_fp16_t *) M->data, ggml_nelements(M));

    ggml_tensor * q_in = rot_q ? ggml_turbo_wht(ctx, Q, 0) : Q;
    ggml_tensor * out  = ggml_flash_attn_ext(ctx, q_in, K, V, M, 1.0f/sqrtf((float) D), 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    if (inv_out) {
        out = ggml_turbo_wht(ctx, out, 1);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    std::vector<float> res(ggml_nelements(out));
    memcpy(res.data(), out->data, res.size()*sizeof(float));
    ggml_free(ctx);
    return res;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - ref[i]) * (a[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return num / den;
}

int main() {
    ggml_cpu_init();

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Build R_K and R_V from the op itself: dir 0 maps e_i to column i of R_K,
    // dir 1 maps e_i to column i of R_V^T, i.e. row i of R_V.
    std::vector<float> eye(128*128, 0.0f);
    for (int i = 0; i < 128; ++i) eye[i*128 + i] = 1.0f;
    const std::vector<float> cols_rk  = run_wht(eye, 0); // cols_rk[i*128 + r]  = R_K[r][i]
    const std::vector<float> rows_rv  = run_wht(eye, 1); // rows_rv[i*128 + r]  = R_V[i][r]

    int n_fail = 0;

    // sanity: both are orthogonal, and K and V really use different rotations
    {
        double err_k = 0.0, err_v = 0.0, diff = 0.0;
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                double dk = 0.0, dv = 0.0;
                for (int r = 0; r < 128; ++r) {
                    dk += cols_rk[i*128 + r] * cols_rk[j*128 + r];
                    dv += rows_rv[i*128 + r] * rows_rv[j*128 + r];
                }
                err_k = std::max(err_k, std::fabs(dk - (i == j)));
                err_v = std::max(err_v, std::fabs(dv - (i == j)));
                diff  = std::max(diff, (double) std::fabs(cols_rk[j*128 + i] - rows_rv[i*128 + j]));
            }
        }
        printf("orthogonality: max |R_K^T R_K - I| = %.2e, max |R_V R_V^T - I| = %.2e, max |R_K - R_V| = %.2f\n", err_k, err_v, diff);
        if (err_k > 1e-5 || err_v > 1e-5 || diff < 0.1) { // entries are +-1/sqrt(128) = 0.088
            printf("FAIL: rotation matrices\n");
            n_fail++;
        }
    }

    // rotate each 128-group of x like the GPU SET_ROWS does for K (R_K) or V (R_V)
    auto rotate = [&](const std::vector<float> & x, bool is_v) {
        std::vector<float> y(x.size());
        for (size_t g = 0; g < x.size()/128; ++g) {
            for (int i = 0; i < 128; ++i) {
                double acc = 0.0;
                for (int r = 0; r < 128; ++r) {
                    const float m = is_v ? rows_rv[i*128 + r] : cols_rk[r*128 + i];
                    acc += m * x[g*128 + r];
                }
                y[g*128 + i] = (float) acc;
            }
        }
        return y;
    };

    std::vector<float> q(D*N_Q*N_H), k(D*N_KV*N_HKV), v(D*N_KV*N_HKV);
    for (auto & x : q) x = nd(rng);
    for (auto & x : k) x = nd(rng);
    for (auto & x : v) x = nd(rng);

    const std::vector<float> k_rot = rotate(k, false);
    const std::vector<float> v_rot = rotate(v, true);

    const std::vector<float> ref = attn(q, k, v, GGML_TYPE_F16, GGML_TYPE_F16, false, false);
    printf("q8_0/q8_0 baseline: nmse = %.3e\n", nmse(attn(q, k, v, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, false), ref));

    struct pair { ggml_type k, v; double tol; };
    const pair pairs[] = {
        { GGML_TYPE_Q8_0,     GGML_TYPE_TURBO6_0, 2e-3 },
        { GGML_TYPE_Q8_0,     GGML_TYPE_TURBO5_0, 5e-3 },
        { GGML_TYPE_F16,      GGML_TYPE_TURBO6_0, 2e-3 },
        { GGML_TYPE_F16,      GGML_TYPE_TURBO5_0, 5e-3 },
        { GGML_TYPE_TURBO6_0, GGML_TYPE_Q8_0,     2e-3 },
        { GGML_TYPE_TURBO5_0, GGML_TYPE_F16,      5e-3 },
        { GGML_TYPE_TURBO6_0, GGML_TYPE_TURBO6_0, 4e-3 },
        { GGML_TYPE_TURBO5_0, GGML_TYPE_TURBO5_0, 1e-2 },
    };

    for (const pair & p : pairs) {
        const bool tk = is_turbo(p.k);
        const bool tv = is_turbo(p.v);
        const std::vector<float> & kd = tk ? k_rot : k;
        const std::vector<float> & vd = tv ? v_rot : v;

        // current rule: rotate Q iff K is turbo, inverse-rotate the output iff V is turbo
        const double e_new = nmse(attn(q, kd, vd, p.k, p.v, tk, tv), ref);
        // previous rule: output rotation gated on K
        const double e_old = nmse(attn(q, kd, vd, p.k, p.v, tk, tk), ref);

        const bool ok = e_new < p.tol;
        printf("K=%-6s V=%-6s: nmse = %.3e (tol %.0e) %s | output rotation gated on K: nmse = %.3e\n",
            ggml_type_name(p.k), ggml_type_name(p.v), e_new, p.tol, ok ? "OK" : "FAIL", e_old);
        if (!ok) {
            n_fail++;
        }
        if (tk != tv && e_old < 0.5) {
            printf("FAIL: K-gated output rotation unexpectedly matches for a mixed pair\n");
            n_fail++;
        }
    }

    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
