#include "common.cuh"

#define CUDA_SOFTCAP_BLOCK_SIZE 256

void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src);

bool ggml_cuda_should_fuse_mul_add_sigmoid_scale(
        const ggml_tensor * mul, const ggml_tensor * add, const ggml_tensor * sig, const ggml_tensor * scale);
void ggml_cuda_op_mul_add_sigmoid_scale(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mul, const ggml_tensor * add, const ggml_tensor * scale_node);
