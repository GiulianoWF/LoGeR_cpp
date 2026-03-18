#pragma once
#include <loger/ops/rope2d.hpp>
#include <torch/torch.h>

namespace loger {

/// Self-attention with RoPE2D positional encoding and optional QK normalization.
/// Equivalent to Python's FlashAttentionRope.
class FlashAttentionRopeImpl : public torch::nn::Module {
public:
    FlashAttentionRopeImpl(int dim, int num_heads, bool qkv_bias = true,
                           bool qk_norm = false, RoPE2D rope = nullptr);

    void load_weights(const class TensorStore& ts, const std::string& prefix);

    /// x:    (B, N, dim)
    /// xpos: (B, N, 2) positions, or undefined (no RoPE applied)
    torch::Tensor forward(torch::Tensor x, torch::Tensor xpos = {});

    /// Compute K and V for KV-cache (used by SWA).
    std::pair<torch::Tensor, torch::Tensor>
    compute_kv(torch::Tensor x, torch::Tensor xpos = {});

    /// Cross-attend x (queries) against external (k_cache, v_cache).
    torch::Tensor forward_with_kv_cache(torch::Tensor x,
                                        torch::Tensor k_cache,
                                        torch::Tensor v_cache,
                                        torch::Tensor xpos = {});

    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }

private:
    int num_heads_, head_dim_;
    bool qk_norm_;
    torch::nn::Linear qkv_{nullptr}, proj_{nullptr};
    torch::nn::LayerNorm q_norm_{nullptr}, k_norm_{nullptr};
    RoPE2D rope_;
};
TORCH_MODULE(FlashAttentionRope);

/// Cross-attention: queries from x, keys/values from context.
/// Equivalent to Python's MemEffCrossAttentionRope.
class CrossAttentionRopeImpl : public torch::nn::Module {
public:
    CrossAttentionRopeImpl(int dim, int num_heads, bool qkv_bias = true,
                           RoPE2D rope = nullptr);

    void load_weights(const class TensorStore& ts, const std::string& prefix);

    /// x:       (B, Nq, dim)  query
    /// context: (B, Nk, dim)  key/value source
    /// xpos:    (B, Nq, 2)    query positions (optional)
    /// cpos:    (B, Nk, 2)    context positions (optional)
    torch::Tensor forward(torch::Tensor x, torch::Tensor context,
                          torch::Tensor xpos = {}, torch::Tensor cpos = {});

private:
    int num_heads_, head_dim_;
    torch::nn::Linear q_{nullptr}, kv_{nullptr}, proj_{nullptr};
    RoPE2D rope_;
};
TORCH_MODULE(CrossAttentionRope);

} // namespace loger
