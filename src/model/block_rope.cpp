#include <loger/model/block_rope.hpp>
#include <loger/io/weight_loader.hpp>

namespace loger {

// ---------------------------------------------------------------------------
// BlockRope
// ---------------------------------------------------------------------------

BlockRopeImpl::BlockRopeImpl(int dim, int num_heads, float mlp_ratio,
                             bool qkv_bias, bool qk_norm, float init_values,
                             bool use_swiglu, RoPE2D rope)
    : use_swiglu_(use_swiglu) {

    norm1_ = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm2_ = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));

    attn_ = register_module("attn",
        FlashAttentionRope(dim, num_heads, qkv_bias, qk_norm, rope));

    use_ls_ = (init_values > 0.0f);
    if (use_ls_) {
        auto ls1 = std::make_shared<LayerScaleImpl>(dim, init_values);
        auto ls2 = std::make_shared<LayerScaleImpl>(dim, init_values);
        register_module("ls1", ls1);
        register_module("ls2", ls2);
        ls1_ = torch::nn::AnyModule(ls1);
        ls2_ = torch::nn::AnyModule(ls2);
    }

    int hidden = static_cast<int>(dim * mlp_ratio);
    if (use_swiglu) {
        auto mlp = std::make_shared<SwiGLUMlpImpl>(dim, hidden);
        register_module("mlp", mlp);
        mlp_ = torch::nn::AnyModule(mlp);
    } else {
        auto mlp = std::make_shared<MlpImpl>(dim, hidden);
        register_module("mlp", mlp);
        mlp_ = torch::nn::AnyModule(mlp);
    }
}

void BlockRopeImpl::load_weights(const TensorStore& ts,
                                 const std::string& prefix) {
    TensorStore::copy_weight(norm1_->weight, ts, prefix + ".norm1.weight");
    TensorStore::copy_weight(norm1_->bias,   ts, prefix + ".norm1.bias");
    TensorStore::copy_weight(norm2_->weight, ts, prefix + ".norm2.weight");
    TensorStore::copy_weight(norm2_->bias,   ts, prefix + ".norm2.bias");
    attn_->load_weights(ts, prefix + ".attn");
    if (use_ls_) {
        ls1_.get<LayerScaleImpl>().load_weights(ts, prefix + ".ls1");
        ls2_.get<LayerScaleImpl>().load_weights(ts, prefix + ".ls2");
    }
    if (use_swiglu_)
        mlp_.get<SwiGLUMlpImpl>().load_weights(ts, prefix + ".mlp");
    else
        mlp_.get<MlpImpl>().load_weights(ts, prefix + ".mlp");
}

// LayerNorm in float32 to match Python autocast behavior (autocast promotes
// LayerNorm to float32, but C++ model.to(bf16) runs it in bf16).
static torch::Tensor ln_f32(torch::nn::LayerNorm& ln, const torch::Tensor& x) {
    auto orig = x.scalar_type();
    auto xf = x.to(torch::kFloat32);
    auto w  = ln->weight.to(torch::kFloat32);
    auto b  = ln->bias.defined() ? ln->bias.to(torch::kFloat32) : torch::Tensor{};
    return torch::layer_norm(xf, ln->options.normalized_shape(), w, b, ln->options.eps()).to(orig);
}

torch::Tensor BlockRopeImpl::forward(torch::Tensor x, torch::Tensor xpos) {
    // Attention branch
    auto attn_out = attn_->forward(ln_f32(norm1_, x), xpos);
    if (use_ls_) attn_out = ls1_.forward<torch::Tensor>(attn_out);
    x = x + attn_out;

    // MLP branch
    torch::Tensor mlp_out;
    if (use_swiglu_)
        mlp_out = mlp_.forward<torch::Tensor>(ln_f32(norm2_, x));
    else
        mlp_out = mlp_.forward<torch::Tensor>(ln_f32(norm2_, x));
    if (use_ls_) mlp_out = ls2_.forward<torch::Tensor>(mlp_out);
    x = x + mlp_out;

    return x;
}

std::pair<torch::Tensor, torch::Tensor>
BlockRopeImpl::compute_kv_cache(torch::Tensor x, torch::Tensor xpos) {
    return attn_->compute_kv(ln_f32(norm1_, x), xpos);
}

torch::Tensor BlockRopeImpl::forward_with_kv_cache(torch::Tensor x,
                                                   torch::Tensor k_cache,
                                                   torch::Tensor v_cache,
                                                   torch::Tensor xpos) {
    auto attn_out = attn_->forward_with_kv_cache(ln_f32(norm1_, x),
                                                  k_cache, v_cache, xpos);
    if (use_ls_) attn_out = ls1_.forward<torch::Tensor>(attn_out);
    x = x + attn_out;

    torch::Tensor mlp_out;
    if (use_swiglu_)
        mlp_out = mlp_.forward<torch::Tensor>(ln_f32(norm2_, x));
    else
        mlp_out = mlp_.forward<torch::Tensor>(ln_f32(norm2_, x));
    if (use_ls_) mlp_out = ls2_.forward<torch::Tensor>(mlp_out);
    x = x + mlp_out;

    return x;
}

// ---------------------------------------------------------------------------
// CrossOnlyBlockRope
// ---------------------------------------------------------------------------

CrossOnlyBlockRopeImpl::CrossOnlyBlockRopeImpl(int dim, int num_heads,
                                               float mlp_ratio, bool qkv_bias,
                                               RoPE2D rope) {
    norm1_    = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm2_    = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm_ctx_ = register_module("norm_ctx",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));

    attn_ = register_module("attn",
        CrossAttentionRope(dim, num_heads, qkv_bias, rope));

    int hidden = static_cast<int>(dim * mlp_ratio);
    mlp_ = register_module("mlp", SwiGLUMlp(dim, hidden));
}

void CrossOnlyBlockRopeImpl::load_weights(const TensorStore& ts,
                                          const std::string& prefix) {
    TensorStore::copy_weight(norm1_->weight,    ts, prefix + ".norm1.weight");
    TensorStore::copy_weight(norm1_->bias,      ts, prefix + ".norm1.bias");
    TensorStore::copy_weight(norm2_->weight,    ts, prefix + ".norm2.weight");
    TensorStore::copy_weight(norm2_->bias,      ts, prefix + ".norm2.bias");
    TensorStore::copy_weight(norm_ctx_->weight, ts, prefix + ".norm_ctx.weight");
    TensorStore::copy_weight(norm_ctx_->bias,   ts, prefix + ".norm_ctx.bias");
    attn_->load_weights(ts, prefix + ".attn");
    mlp_->load_weights(ts,  prefix + ".mlp");
}

torch::Tensor CrossOnlyBlockRopeImpl::forward(torch::Tensor x,
                                              torch::Tensor context,
                                              torch::Tensor xpos) {
    auto ctx_normed = norm_ctx_->forward(context);
    auto attn_out   = attn_->forward(norm1_->forward(x), ctx_normed, xpos);
    x = x + attn_out;
    x = x + mlp_->forward(norm2_->forward(x));
    return x;
}

} // namespace loger
