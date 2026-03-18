#pragma once
#include <loger/ops/attention.hpp>
#include <loger/ops/mlp.hpp>
#include <torch/torch.h>

namespace loger {

/// Transformer block with RoPE positional encoding.
/// Equivalent to Python's BlockRope.
class BlockRopeImpl : public torch::nn::Module {
public:
    /// dim:       model dimension
    /// num_heads: attention heads
    /// mlp_ratio: hidden / dim ratio for MLP
    /// qkv_bias:  bias in QKV projection
    /// qk_norm:   LayerNorm on Q and K before attention
    /// init_values: LayerScale init (<=0 disables LayerScale)
    /// use_swiglu: use SwiGLU MLP (decoder blocks); else GELU MLP (encoder)
    BlockRopeImpl(int dim, int num_heads, float mlp_ratio = 4.0f,
                  bool qkv_bias = true, bool qk_norm = false,
                  float init_values = 0.0f, bool use_swiglu = true,
                  RoPE2D rope = nullptr);

    void load_weights(const class TensorStore& ts, const std::string& prefix);

    /// x:    (B, N, dim)
    /// xpos: (B, N, 2) or undefined
    torch::Tensor forward(torch::Tensor x, torch::Tensor xpos = {});

    /// For SWA: compute K,V without running the full block.
    std::pair<torch::Tensor, torch::Tensor>
    compute_kv_cache(torch::Tensor x, torch::Tensor xpos = {});

    /// Forward using external KV cache (cross-attention mode).
    torch::Tensor forward_with_kv_cache(torch::Tensor x,
                                        torch::Tensor k_cache,
                                        torch::Tensor v_cache,
                                        torch::Tensor xpos = {});

private:
    torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
    FlashAttentionRope attn_{nullptr};
    // LayerScale (optional)
    torch::nn::AnyModule ls1_, ls2_;
    bool use_ls_;
    // MLP (SwiGLU or GELU)
    torch::nn::AnyModule mlp_;
    bool use_swiglu_;
};
TORCH_MODULE(BlockRope);

/// Cross-attention-only block (used in task TransformerDecoders).
/// Equivalent to Python's CrossOnlyBlockRope.
class CrossOnlyBlockRopeImpl : public torch::nn::Module {
public:
    CrossOnlyBlockRopeImpl(int dim, int num_heads, float mlp_ratio = 4.0f,
                           bool qkv_bias = true, RoPE2D rope = nullptr);

    void load_weights(const class TensorStore& ts, const std::string& prefix);

    /// x:       (B, Nq, dim)  query tokens
    /// context: (B, Nk, dim)  encoder output (key/value source)
    /// xpos:    (B, Nq, 2)    optional query positions
    torch::Tensor forward(torch::Tensor x, torch::Tensor context,
                          torch::Tensor xpos = {});

private:
    torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr}, norm_ctx_{nullptr};
    CrossAttentionRope attn_{nullptr};
    SwiGLUMlp mlp_{nullptr};
};
TORCH_MODULE(CrossOnlyBlockRope);

} // namespace loger
