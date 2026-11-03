#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// residual ADD (add0, optionally a second ADD add1 of add0) -> RMS_NORM -> MUL in one launch: writes the sum and the
// MUL output, with the arithmetic of the separate ops. false: not applicable, nothing launched.
bool ggml_cuda_op_add_rms_norm_mul_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add0, ggml_tensor * add1,
                                         ggml_tensor * norm, ggml_tensor * mul_tensor);

// two independent same-shape RMS_NORM -> SCALE pairs (the GDN q and k l2 norms) in one launch.
// false: not applicable, nothing launched.
bool ggml_cuda_op_rms_norm_scale_pair_fused(ggml_backend_cuda_context & ctx, ggml_tensor * norm0, ggml_tensor * scale0,
                                            ggml_tensor * norm1, ggml_tensor * scale1);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
