#include "softcap.cuh"

static __global__ void softcap_f32(const float * x, float * dst, const float scale, const float softcap, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = tanhf(scale * x[i]) * softcap;
}

static void softcap_f32_cuda(const float * x, float * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(softcap_f32, launch_params, x, dst, scale, softcap, k);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    softcap_f32_cuda(src0_d, dst_d, scale, softcap, ggml_nelements(src0), stream);
}

// fused GGML_OP_MUL (scalar) + GGML_OP_ADD (per-column bias) + GGML_UNARY_OP_SIGMOID + GGML_OP_SCALE
// The DeepSeek-V4 hyper-connection gates are sigmoid(x * s + b[i0]) * scale + bias on a [n, n_tokens] view.
// Each step is kept as its own rounded operation so the result matches the four separate kernels bit for bit.
static __global__ void mul_add_sigmoid_scale_f32(
        const float * x, const float * s, const float * b, float * dst,
        const float scale, const float bias,
        const int ne0, const int64_t s0, const int64_t s1, const int64_t k) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t i1 = i / ne0;
    const int64_t i0 = i - i1*ne0;

    ggml_cuda_pdl_sync();
    float v = __fmul_rn(x[i1*s1 + i0*s0], s[0]);
    v = __fadd_rn(v, b[i0]);
    v = 1.0f / (1.0f + expf(-v));
    dst[i] = scale * v + bias;
}

static int gate_dbg() {
    static const int v = getenv("GGML_CUDA_GATE_DEBUG") ? atoi(getenv("GGML_CUDA_GATE_DEBUG")) : 0;
    return v;
}

bool ggml_cuda_should_fuse_mul_add_sigmoid_scale(
        const ggml_tensor * mul, const ggml_tensor * add, const ggml_tensor * sig, const ggml_tensor * scale) {
    if (add->src[0] != mul || sig->src[0] != add || scale->src[0] != sig) {
        return false;
    }
    const ggml_tensor * x = mul->src[0];
    const ggml_tensor * s = mul->src[1];
    const ggml_tensor * b = add->src[1];
    if (x->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || sig->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
        return false;
    }
    if (x->ne[2] != 1 || x->ne[3] != 1 || x->nb[0] != sizeof(float)) {
        return false;
    }
    if (ggml_nelements(s) != 1 || ggml_nelements(b) != x->ne[0] || b->nb[0] != sizeof(float)) {
        return false;
    }
    if (!ggml_is_contiguous(scale) || !ggml_are_same_shape(x, scale)) {
        return false;
    }
    if (gate_dbg()) {
        fprintf(stderr, "[gate] fuse %s\n", scale->name);
    }
    return true;
}

void ggml_cuda_op_mul_add_sigmoid_scale(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mul, const ggml_tensor * add, const ggml_tensor * scale_node) {
    const ggml_tensor * x = mul->src[0];
    const ggml_tensor * s = mul->src[1];
    const ggml_tensor * b = add->src[1];

    float scale;
    float bias;
    memcpy(&scale, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&bias,  (const float *) scale_node->op_params + 1, sizeof(float));

    const int64_t k  = ggml_nelements(scale_node);
    const int     ne0 = (int) x->ne[0];
    const int64_t s0 = x->nb[0] / sizeof(float);
    const int64_t s1 = x->nb[1] / sizeof(float);

    const int num_blocks = (int) ((k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(mul_add_sigmoid_scale_f32, launch_params,
        (const float *) x->data, (const float *) s->data, (const float *) b->data, (float *) scale_node->data,
        scale, bias, ne0, s0, s1, k);
}
