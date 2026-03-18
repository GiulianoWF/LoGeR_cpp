#pragma once
#include <loger/ops/mlp.hpp>
#include <loger/io/weight_loader.hpp>
#include <torch/torch.h>

namespace loger {

/// Standard encoder attention (no RoPE, plain SDPA).
class EncoderAttentionImpl : public torch::nn::Module {
public:
    EncoderAttentionImpl(int dim, int num_heads, bool qkv_bias = true);
    void load_weights(const TensorStore& ts, const std::string& prefix);
    torch::Tensor forward(torch::Tensor x);

private:
    int num_heads_, head_dim_;
    torch::nn::Linear qkv_{nullptr}, proj_{nullptr};
};
TORCH_MODULE(EncoderAttention);

/// One DINOv2 transformer block: LN→Attn→LS1→residual, LN→MLP→LS2→residual.
class EncoderBlockImpl : public torch::nn::Module {
public:
    EncoderBlockImpl(int dim, int num_heads, float mlp_ratio = 4.0f);
    void load_weights(const TensorStore& ts, const std::string& prefix);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
    EncoderAttention attn_{nullptr};
    Mlp mlp_{nullptr};
    LayerScale ls1_{nullptr}, ls2_{nullptr};
};
TORCH_MODULE(EncoderBlock);

/// DINOv2 ViT-L/14 with register tokens.
/// Replicates the Python encoder's output: x_norm_patchtokens (B*N, hw, 1024).
class DinoV2EncoderImpl : public torch::nn::Module {
public:
    static constexpr int EMBED_DIM       = 1024;
    static constexpr int NUM_HEADS       = 16;
    static constexpr int DEPTH           = 24;
    static constexpr int PATCH_SIZE      = 14;
    static constexpr int NUM_REGISTERS   = 4;
    static constexpr int BASE_GRID_SIZE  = 37;  // 518/14 rounded

    DinoV2EncoderImpl();
    void load_weights(const TensorStore& ts, const std::string& prefix = "encoder");

    /// x: (B, 3, H, W) — H and W must be multiples of PATCH_SIZE
    /// Returns x_norm_patchtokens: (B, hw, EMBED_DIM)
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d patch_embed_{nullptr};
    torch::Tensor cls_token_;
    torch::Tensor pos_embed_;          // (1, 1+NUM_REGISTERS+BASE_GRID_SIZE^2, EMBED_DIM)
    torch::Tensor register_tokens_;    // (1, NUM_REGISTERS, EMBED_DIM)
    torch::nn::ModuleList blocks_;     // DEPTH x EncoderBlock
    torch::nn::LayerNorm norm_{nullptr};

    torch::Tensor interpolate_pos_encoding(torch::Tensor x, int w, int h);
    torch::Tensor prepare_tokens(torch::Tensor x);
};
TORCH_MODULE(DinoV2Encoder);

} // namespace loger
