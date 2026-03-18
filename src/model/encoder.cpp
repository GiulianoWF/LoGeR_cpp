#include <loger/model/encoder.hpp>
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
// EncoderAttention
// ---------------------------------------------------------------------------

EncoderAttentionImpl::EncoderAttentionImpl(int dim, int num_heads, bool qkv_bias)
    : num_heads_(num_heads), head_dim_(dim / num_heads) {

    qkv_ = register_module("qkv", torch::nn::Linear(
        torch::nn::LinearOptions(dim, 3 * dim).bias(qkv_bias)));
    proj_ = register_module("proj", torch::nn::Linear(dim, dim));
}

void EncoderAttentionImpl::load_weights(const TensorStore& ts,
                                        const std::string& prefix) {
    TensorStore::copy_weight(qkv_->weight, ts, prefix + ".qkv.weight");
    TensorStore::copy_weight_optional(qkv_->bias, ts, prefix + ".qkv.bias");
    TensorStore::copy_weight(proj_->weight, ts, prefix + ".proj.weight");
    TensorStore::copy_weight_optional(proj_->bias, ts, prefix + ".proj.bias");
}

torch::Tensor EncoderAttentionImpl::forward(torch::Tensor x) {
    const auto B = x.size(0);
    const auto N = x.size(1);
    const auto C = x.size(2);

    auto qkv = qkv_->forward(x)
                    .reshape({B, N, 3, num_heads_, head_dim_})
                    .permute({2, 0, 3, 1, 4});
    auto q = qkv[0], k = qkv[1], v = qkv[2];

    auto out = at::scaled_dot_product_attention(q, k, v);
    out = out.permute({0, 2, 1, 3}).reshape({B, N, C});
    return proj_->forward(out);
}

// ---------------------------------------------------------------------------
// EncoderBlock
// ---------------------------------------------------------------------------

EncoderBlockImpl::EncoderBlockImpl(int dim, int num_heads, float mlp_ratio) {
    norm1_ = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm2_ = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    attn_  = register_module("attn", EncoderAttention(dim, num_heads));
    int hidden = static_cast<int>(dim * mlp_ratio);
    mlp_   = register_module("mlp", Mlp(dim, hidden));
    ls1_   = register_module("ls1", LayerScale(dim, 1.0f));
    ls2_   = register_module("ls2", LayerScale(dim, 1.0f));
}

void EncoderBlockImpl::load_weights(const TensorStore& ts,
                                    const std::string& prefix) {
    TensorStore::copy_weight(norm1_->weight, ts, prefix + ".norm1.weight");
    TensorStore::copy_weight(norm1_->bias,   ts, prefix + ".norm1.bias");
    TensorStore::copy_weight(norm2_->weight, ts, prefix + ".norm2.weight");
    TensorStore::copy_weight(norm2_->bias,   ts, prefix + ".norm2.bias");
    attn_->load_weights(ts, prefix + ".attn");
    mlp_->load_weights(ts,  prefix + ".mlp");
    ls1_->load_weights(ts,  prefix + ".ls1");
    ls2_->load_weights(ts,  prefix + ".ls2");
}

torch::Tensor EncoderBlockImpl::forward(torch::Tensor x) {
    x = x + ls1_->forward(attn_->forward(ln_f32(norm1_, x)));
    x = x + ls2_->forward(mlp_->forward(ln_f32(norm2_, x)));
    return x;
}

// ---------------------------------------------------------------------------
// DinoV2Encoder
// ---------------------------------------------------------------------------

DinoV2EncoderImpl::DinoV2EncoderImpl() {
    patch_embed_ = register_module("patch_embed",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, EMBED_DIM, PATCH_SIZE)
                              .stride(PATCH_SIZE)));

    cls_token_      = register_parameter("cls_token",
        torch::zeros({1, 1, EMBED_DIM}));
    pos_embed_      = register_parameter("pos_embed",
        torch::zeros({1, 1 + BASE_GRID_SIZE * BASE_GRID_SIZE,
                      EMBED_DIM}));
    register_tokens_ = register_parameter("register_tokens",
        torch::zeros({1, NUM_REGISTERS, EMBED_DIM}));

    register_module("blocks", blocks_);
    for (int i = 0; i < DEPTH; ++i)
        blocks_->push_back(EncoderBlock(EMBED_DIM, NUM_HEADS));

    norm_ = register_module("norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({EMBED_DIM})));
}

void DinoV2EncoderImpl::load_weights(const TensorStore& ts,
                                     const std::string& prefix) {
    TensorStore::copy_weight(patch_embed_->weight, ts,
        prefix + ".patch_embed.proj.weight");
    TensorStore::copy_weight_optional(patch_embed_->bias, ts,
        prefix + ".patch_embed.proj.bias");

    TensorStore::copy_weight(cls_token_,       ts, prefix + ".cls_token");
    TensorStore::copy_weight(pos_embed_,       ts, prefix + ".pos_embed");
    TensorStore::copy_weight(register_tokens_, ts, prefix + ".register_tokens");

    for (int i = 0; i < DEPTH; ++i) {
        auto& blk = blocks_->at<EncoderBlockImpl>(i);
        blk.load_weights(ts, prefix + ".blocks." + std::to_string(i));
    }

    TensorStore::copy_weight(norm_->weight, ts, prefix + ".norm.weight");
    TensorStore::copy_weight(norm_->bias,   ts, prefix + ".norm.bias");
}

// Bicubic interpolation of patch positional embeddings for non-standard input sizes.
// Mirrors the Python `interpolate_pos_encoding` method.
torch::Tensor DinoV2EncoderImpl::interpolate_pos_encoding(torch::Tensor x,
                                                           int w, int h) {
    // x: (B, N_patches+1+NUM_REGS, EMBED_DIM)
    // x: (B, 1+hw, EMBED_DIM) — cls + patches only (registers not yet inserted)
    const int N = x.size(1) - 1;  // number of patch tokens in x
    const int M = pos_embed_.size(1) - 1;  // base number of patch positions

    if (N == M && w == h)
        return pos_embed_;

    // pos_embed_ shape: (1, 1+BASE_GRID^2, EMBED_DIM)
    auto class_pos_embed = pos_embed_.index(
        {torch::indexing::Slice(), torch::indexing::Slice(0, 1)});
    auto patch_pos_embed = pos_embed_.index(
        {torch::indexing::Slice(), torch::indexing::Slice(1)});

    const int dim = x.size(-1);
    const int w0  = w / PATCH_SIZE;
    const int h0  = h / PATCH_SIZE;

    // Reshape patch_pos_embed to 2D spatial for bicubic interp
    auto sqrt_M = static_cast<int>(std::round(std::sqrt(static_cast<double>(M))));
    patch_pos_embed = patch_pos_embed
        .reshape({1, sqrt_M, sqrt_M, dim})
        .permute({0, 3, 1, 2});  // (1, dim, sqrt_M, sqrt_M)

    patch_pos_embed = torch::nn::functional::interpolate(
        patch_pos_embed.to(torch::kFloat32),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{h0, w0})
            .mode(torch::kBicubic)
            .align_corners(false)
    ).to(x.dtype());

    // (1, dim, h0, w0) → (1, h0*w0, dim)
    patch_pos_embed = patch_pos_embed
        .permute({0, 2, 3, 1})
        .reshape({1, h0 * w0, dim});

    return torch::cat({class_pos_embed, patch_pos_embed}, /*dim=*/1);
}

torch::Tensor DinoV2EncoderImpl::prepare_tokens(torch::Tensor x) {
    // x: (B, 3, H, W)
    const int B = x.size(0);
    const int W = x.size(3);
    const int H = x.size(2);

    // Patch embedding: (B, EMBED_DIM, h, w) → (B, hw, EMBED_DIM)
    auto patches = patch_embed_->forward(x)
                       .flatten(2)
                       .transpose(1, 2);  // (B, hw, EMBED_DIM)

    // CLS token: (B, 1, EMBED_DIM)
    auto cls = cls_token_.expand({B, -1, -1});

    // Add positional embedding to cls+patches BEFORE inserting register tokens.
    // pos_embed covers only [cls, patches], registers get no positional encoding.
    auto tokens = torch::cat({cls, patches}, /*dim=*/1);  // (B, 1+hw, EMBED_DIM)
    tokens = tokens + interpolate_pos_encoding(tokens, W, H);

    // Insert register tokens between cls and patches: (B, 1+NUM_REGS+hw, EMBED_DIM)
    auto regs = register_tokens_.expand({B, -1, -1});
    tokens = torch::cat({
        tokens.index({torch::indexing::Slice(),
                      torch::indexing::Slice(0, 1)}),   // cls
        regs,                                            // registers
        tokens.index({torch::indexing::Slice(),
                      torch::indexing::Slice(1)})        // patches
    }, /*dim=*/1);

    return tokens;
}

torch::Tensor DinoV2EncoderImpl::forward(torch::Tensor x) {
    // x: (B, 3, H, W)
    auto tokens = prepare_tokens(x);  // (B, 1+NUM_REGS+hw, EMBED_DIM)

    for (int i = 0; i < DEPTH; ++i)
        tokens = blocks_->at<EncoderBlockImpl>(i).forward(tokens);

    tokens = ln_f32(norm_, tokens);

    // Return only patch tokens (skip cls + register tokens): (B, hw, EMBED_DIM)
    return tokens.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(1 + NUM_REGISTERS),
        torch::indexing::Slice()
    });
}

} // namespace loger
