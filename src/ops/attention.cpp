#include <loger/ops/attention.hpp>
#include <loger/ops/mlp.hpp>
#include <loger/io/weight_loader.hpp>

namespace {
// LayerNorm in float32 to match Python autocast behavior.
static torch::Tensor ln_f32(torch::nn::LayerNorm& ln, const torch::Tensor& x) {
    auto orig = x.scalar_type();
    auto xf = x.to(torch::kFloat32);
    auto w  = ln->weight.to(torch::kFloat32);
    auto b  = ln->bias.defined() ? ln->bias.to(torch::kFloat32) : torch::Tensor{};
    return torch::layer_norm(xf, ln->options.normalized_shape(), w, b, ln->options.eps()).to(orig);
}
} // anon

namespace loger {

// ---------------------------------------------------------------------------
// Mlp
// ---------------------------------------------------------------------------

MlpImpl::MlpImpl(int in_features, int hidden_features, int out_features,
                 bool bias) {
    if (hidden_features < 0) hidden_features = in_features * 4;
    if (out_features < 0)    out_features    = in_features;
    fc1_ = register_module("fc1", torch::nn::Linear(
        torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
    fc2_ = register_module("fc2", torch::nn::Linear(
        torch::nn::LinearOptions(hidden_features, out_features).bias(bias)));
}

void MlpImpl::load_weights(const TensorStore& ts, const std::string& prefix) {
    TensorStore::copy_weight(fc1_->weight, ts, prefix + ".fc1.weight");
    TensorStore::copy_weight_optional(fc1_->bias,   ts, prefix + ".fc1.bias");
    TensorStore::copy_weight(fc2_->weight, ts, prefix + ".fc2.weight");
    TensorStore::copy_weight_optional(fc2_->bias,   ts, prefix + ".fc2.bias");
}

torch::Tensor MlpImpl::forward(torch::Tensor x) {
    return fc2_->forward(torch::gelu(fc1_->forward(x)));
}

// ---------------------------------------------------------------------------
// SwiGLUMlp
// ---------------------------------------------------------------------------

SwiGLUMlpImpl::SwiGLUMlpImpl(int in_features, int hidden_features, bool bias) {
    if (hidden_features < 0) hidden_features = static_cast<int>(in_features * 8.0 / 3.0);
    // Align to multiple of 256 for efficiency (mirrors Python xformers SwiGLU)
    hidden_features = ((hidden_features + 255) / 256) * 256;
    fc1_ = register_module("fc1", torch::nn::Linear(
        torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
    fc2_ = register_module("fc2", torch::nn::Linear(
        torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
    fc3_ = register_module("fc3", torch::nn::Linear(
        torch::nn::LinearOptions(hidden_features, in_features).bias(bias)));
}

void SwiGLUMlpImpl::load_weights(const TensorStore& ts, const std::string& prefix) {
    TensorStore::copy_weight(fc1_->weight, ts, prefix + ".fc1.weight");
    TensorStore::copy_weight_optional(fc1_->bias, ts, prefix + ".fc1.bias");
    TensorStore::copy_weight(fc2_->weight, ts, prefix + ".fc2.weight");
    TensorStore::copy_weight_optional(fc2_->bias, ts, prefix + ".fc2.bias");
    TensorStore::copy_weight(fc3_->weight, ts, prefix + ".fc3.weight");
    TensorStore::copy_weight_optional(fc3_->bias, ts, prefix + ".fc3.bias");
}

torch::Tensor SwiGLUMlpImpl::forward(torch::Tensor x) {
    return fc3_->forward(torch::silu(fc1_->forward(x)) * fc2_->forward(x));
}

// ---------------------------------------------------------------------------
// LayerScale
// ---------------------------------------------------------------------------

LayerScaleImpl::LayerScaleImpl(int dim, float init_value) {
    gamma = register_parameter("gamma",
        torch::full({dim}, init_value));
}

void LayerScaleImpl::load_weights(const TensorStore& ts, const std::string& prefix) {
    TensorStore::copy_weight(gamma, ts, prefix + ".gamma");
}

torch::Tensor LayerScaleImpl::forward(torch::Tensor x) {
    return x * gamma;
}

// ---------------------------------------------------------------------------
// FlashAttentionRope
// ---------------------------------------------------------------------------

FlashAttentionRopeImpl::FlashAttentionRopeImpl(int dim, int num_heads,
                                               bool qkv_bias, bool qk_norm,
                                               RoPE2D rope)
    : num_heads_(num_heads), head_dim_(dim / num_heads), qk_norm_(qk_norm),
      rope_(std::move(rope)) {

    qkv_ = register_module("qkv", torch::nn::Linear(
        torch::nn::LinearOptions(dim, 3 * dim).bias(qkv_bias)));
    proj_ = register_module("proj", torch::nn::Linear(dim, dim));

    if (qk_norm) {
        q_norm_ = register_module("q_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim_})));
        k_norm_ = register_module("k_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim_})));
    }
}

void FlashAttentionRopeImpl::load_weights(const TensorStore& ts,
                                          const std::string& prefix) {
    TensorStore::copy_weight(qkv_->weight, ts, prefix + ".qkv.weight");
    TensorStore::copy_weight_optional(qkv_->bias, ts, prefix + ".qkv.bias");
    TensorStore::copy_weight(proj_->weight, ts, prefix + ".proj.weight");
    TensorStore::copy_weight_optional(proj_->bias, ts, prefix + ".proj.bias");
    if (qk_norm_) {
        TensorStore::copy_weight(q_norm_->weight, ts, prefix + ".q_norm.weight");
        TensorStore::copy_weight(k_norm_->weight, ts, prefix + ".k_norm.weight");
    }
}

torch::Tensor FlashAttentionRopeImpl::forward(torch::Tensor x,
                                              torch::Tensor xpos) {
    const auto B  = x.size(0);
    const auto N  = x.size(1);
    const auto C  = x.size(2);

    // QKV projection: (B, N, 3*C) → split → (B, H, N, D)
    auto qkv = qkv_->forward(x)
                    .reshape({B, N, 3, num_heads_, head_dim_})
                    .permute({2, 0, 3, 1, 4});  // (3, B, H, N, D)

    auto q = qkv[0], k = qkv[1], v = qkv[2];

    if (qk_norm_) {
        q = ln_f32(q_norm_, q);
        k = ln_f32(k_norm_, k);
    }

    // Apply RoPE if positions provided
    if (rope_.is_empty() == false && xpos.defined() && xpos.numel() > 0) {
        q = rope_->forward(q, xpos);
        k = rope_->forward(k, xpos);
    }

    // Scaled dot-product attention (uses Flash Attention kernel on CUDA)
    auto attn_out = at::scaled_dot_product_attention(q, k, v);

    // Merge heads: (B, H, N, D) → (B, N, C)
    attn_out = attn_out.permute({0, 2, 1, 3}).reshape({B, N, C});

    return proj_->forward(attn_out);
}

std::pair<torch::Tensor, torch::Tensor>
FlashAttentionRopeImpl::compute_kv(torch::Tensor x, torch::Tensor xpos) {
    const auto B  = x.size(0);
    const auto N  = x.size(1);

    auto qkv = qkv_->forward(x)
                    .reshape({B, N, 3, num_heads_, head_dim_})
                    .permute({2, 0, 3, 1, 4});
    auto k = qkv[1], v = qkv[2];

    // Apply QK-norm to K before caching (matches Python compute_kv)
    if (qk_norm_) {
        k = ln_f32(k_norm_, k);
    }

    if (rope_.is_empty() == false && xpos.defined() && xpos.numel() > 0) {
        k = rope_->forward(k, xpos);
    }

    return {k, v};
}

torch::Tensor FlashAttentionRopeImpl::forward_with_kv_cache(
    torch::Tensor x, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor xpos) {

    const auto B = x.size(0);
    const auto N = x.size(1);
    const auto C = x.size(2);

    auto qkv = qkv_->forward(x)
                    .reshape({B, N, 3, num_heads_, head_dim_})
                    .permute({2, 0, 3, 1, 4});
    auto q = qkv[0];

    if (qk_norm_) q = ln_f32(q_norm_, q);
    if (rope_.is_empty() == false && xpos.defined() && xpos.numel() > 0)
        q = rope_->forward(q, xpos);

    // Cross-attend against cached K, V
    auto attn_out = at::scaled_dot_product_attention(q, k_cache, v_cache);
    attn_out = attn_out.permute({0, 2, 1, 3}).reshape({B, N, C});
    return proj_->forward(attn_out);
}

// ---------------------------------------------------------------------------
// CrossAttentionRope
// ---------------------------------------------------------------------------

CrossAttentionRopeImpl::CrossAttentionRopeImpl(int dim, int num_heads,
                                               bool qkv_bias, RoPE2D rope)
    : num_heads_(num_heads), head_dim_(dim / num_heads),
      rope_(std::move(rope)) {

    q_  = register_module("q",  torch::nn::Linear(
        torch::nn::LinearOptions(dim, dim).bias(qkv_bias)));
    kv_ = register_module("kv", torch::nn::Linear(
        torch::nn::LinearOptions(dim, 2 * dim).bias(qkv_bias)));
    proj_ = register_module("proj", torch::nn::Linear(dim, dim));
}

void CrossAttentionRopeImpl::load_weights(const TensorStore& ts,
                                          const std::string& prefix) {
    TensorStore::copy_weight(q_->weight,    ts, prefix + ".q.weight");
    TensorStore::copy_weight_optional(q_->bias, ts, prefix + ".q.bias");
    TensorStore::copy_weight(kv_->weight,   ts, prefix + ".kv.weight");
    TensorStore::copy_weight_optional(kv_->bias, ts, prefix + ".kv.bias");
    TensorStore::copy_weight(proj_->weight, ts, prefix + ".proj.weight");
    TensorStore::copy_weight_optional(proj_->bias, ts, prefix + ".proj.bias");
}

torch::Tensor CrossAttentionRopeImpl::forward(torch::Tensor x,
                                              torch::Tensor context,
                                              torch::Tensor xpos,
                                              torch::Tensor cpos) {
    const auto B  = x.size(0);
    const auto Nq = x.size(1);
    const auto C  = x.size(2);
    const auto Nk = context.size(1);

    // Queries from x
    auto q = q_->forward(x)
                .reshape({B, Nq, num_heads_, head_dim_})
                .permute({0, 2, 1, 3});  // (B, H, Nq, D)

    // Keys and values from context
    auto kv = kv_->forward(context)
                  .reshape({B, Nk, 2, num_heads_, head_dim_})
                  .permute({2, 0, 3, 1, 4});  // (2, B, H, Nk, D)
    auto k = kv[0], v = kv[1];

    if (rope_.is_empty() == false) {
        if (xpos.defined() && xpos.numel() > 0) q = rope_->forward(q, xpos);
        if (cpos.defined() && cpos.numel() > 0) k = rope_->forward(k, cpos);
    }

    auto attn_out = at::scaled_dot_product_attention(q, k, v);
    attn_out = attn_out.permute({0, 2, 1, 3}).reshape({B, Nq, C});
    return proj_->forward(attn_out);
}

} // namespace loger
