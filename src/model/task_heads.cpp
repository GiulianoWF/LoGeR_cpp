#include <loger/model/task_heads.hpp>
#include <loger/io/weight_loader.hpp>
#include <filesystem>
#include <fstream>

namespace {
void save_debug_td(const torch::Tensor& t, const std::string& path) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path().string().empty()
            ? "." : std::filesystem::path(path).parent_path());
    auto vec = torch::pickle_save(t.detach().to(torch::kCPU).to(torch::kFloat32).contiguous());
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(vec.data(), static_cast<std::streamsize>(vec.size()));
    std::cout << "[debug-td] saved " << path << " shape=" << t.sizes() << "\n";
}
} // anon

namespace loger {

// ---------------------------------------------------------------------------
// ResConvBlock
// ---------------------------------------------------------------------------

ResConvBlockImpl::ResConvBlockImpl(int dim_in, int dim_out)
    : need_skip_(dim_in != dim_out) {

    res_conv1_ = register_module("res_conv1",
        torch::nn::Linear(dim_in, dim_out));
    res_conv2_ = register_module("res_conv2",
        torch::nn::Linear(dim_out, dim_out));
    res_conv3_ = register_module("res_conv3",
        torch::nn::Linear(dim_out, dim_out));
    if (need_skip_)
        skip_ = register_module("skip",
            torch::nn::Linear(torch::nn::LinearOptions(dim_in, dim_out).bias(false)));
}

void ResConvBlockImpl::load_weights(const TensorStore& ts,
                                    const std::string& prefix) {
    TensorStore::copy_weight(res_conv1_->weight, ts, prefix + ".res_conv1.weight");
    TensorStore::copy_weight_optional(res_conv1_->bias, ts, prefix + ".res_conv1.bias");
    TensorStore::copy_weight(res_conv2_->weight, ts, prefix + ".res_conv2.weight");
    TensorStore::copy_weight_optional(res_conv2_->bias, ts, prefix + ".res_conv2.bias");
    TensorStore::copy_weight(res_conv3_->weight, ts, prefix + ".res_conv3.weight");
    TensorStore::copy_weight_optional(res_conv3_->bias, ts, prefix + ".res_conv3.bias");
    if (need_skip_) {
        TensorStore::copy_weight(skip_->weight, ts, prefix + ".skip.weight");
    }
}

torch::Tensor ResConvBlockImpl::forward(torch::Tensor x) {
    auto residual = x;
    if (need_skip_) residual = skip_->forward(x);
    auto out = torch::relu(res_conv1_->forward(x));
    out = torch::relu(res_conv2_->forward(out));
    out = torch::relu(res_conv3_->forward(out));
    return out + residual;
}

// ---------------------------------------------------------------------------
// TransformerDecoder
// ---------------------------------------------------------------------------

TransformerDecoderImpl::TransformerDecoderImpl(int in_dim, int out_dim,
                                               int dec_embed_dim, int depth,
                                               int num_heads, RoPE2D rope) {
    projects_ = register_module("projects",
        torch::nn::Linear(in_dim, dec_embed_dim));

    register_module("blocks", blocks_);
    for (int i = 0; i < depth; ++i)
        blocks_->push_back(
            BlockRope(dec_embed_dim, num_heads,
                      /*mlp_ratio=*/4.0f, /*qkv_bias=*/true, /*qk_norm=*/false,
                      /*init_values=*/0.0f, /*use_swiglu=*/false, rope));

    linear_out_ = register_module("linear_out",
        torch::nn::Linear(dec_embed_dim, out_dim));
}

void TransformerDecoderImpl::load_weights(const TensorStore& ts,
                                          const std::string& prefix) {
    TensorStore::copy_weight(projects_->weight, ts, prefix + ".projects.weight");
    TensorStore::copy_weight_optional(projects_->bias, ts, prefix + ".projects.bias");

    const int depth = static_cast<int>(blocks_->size());
    for (int i = 0; i < depth; ++i) {
        blocks_->ptr<BlockRopeImpl>(i)->load_weights(
            ts, prefix + ".blocks." + std::to_string(i));
    }

    TensorStore::copy_weight(linear_out_->weight, ts, prefix + ".linear_out.weight");
    TensorStore::copy_weight_optional(linear_out_->bias, ts, prefix + ".linear_out.bias");
}

torch::Tensor TransformerDecoderImpl::forward(torch::Tensor hidden,
                                              torch::Tensor xpos) {
    // Project input
    auto x = projects_->forward(hidden);  // (BN, hw, dec_embed_dim)

    const int depth = static_cast<int>(blocks_->size());
    for (int i = 0; i < depth; ++i) {
        x = blocks_->ptr<BlockRopeImpl>(i)->forward(x, xpos);
    }

    auto out = linear_out_->forward(x);  // (BN, hw, out_dim)

    // Debug: save per-block outputs
    if (std::getenv("LOGER_DEBUG_TD") && !debug_tag_.empty()) {
        static bool done = false;
        if (!done) {
            // Re-run to save intermediates (forward already ran, but this is debug-only)
            auto xd = projects_->forward(hidden);
            save_debug_td(xd, "debug_cpp/" + debug_tag_ + "_projects.pt");
            for (int i = 0; i < depth; ++i) {
                xd = blocks_->ptr<BlockRopeImpl>(i)->forward(xd, xpos);
                save_debug_td(xd, "debug_cpp/" + debug_tag_ + "_block" + std::to_string(i) + ".pt");
            }
            if (debug_tag_ == "conf_decoder") done = true;
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// LinearPts3d
// ---------------------------------------------------------------------------

LinearPts3dImpl::LinearPts3dImpl(int patch_size, int dec_embed_dim,
                                 int output_dim)
    : patch_size_(patch_size), output_dim_(output_dim) {

    proj_ = register_module("proj",
        torch::nn::Linear(dec_embed_dim,
                          output_dim * patch_size * patch_size));
}

void LinearPts3dImpl::load_weights(const TensorStore& ts,
                                   const std::string& prefix) {
    TensorStore::copy_weight(proj_->weight, ts, prefix + ".proj.weight");
    TensorStore::copy_weight_optional(proj_->bias, ts, prefix + ".proj.bias");
}

torch::Tensor LinearPts3dImpl::forward(torch::Tensor tokens, int H, int W) {
    // tokens: (BN, hw, dec_embed_dim)
    const int BN = tokens.size(0);
    const int hw = tokens.size(1);
    const int ph = H / patch_size_;
    const int pw = W / patch_size_;

    // Project: (BN, hw, output_dim * patch_size^2)
    auto feat = proj_->forward(tokens);

    // Reshape to spatial: (BN, output_dim*ps^2, ph, pw)
    feat = feat.transpose(1, 2)  // (BN, output_dim*ps^2, hw)
               .reshape({BN, output_dim_ * patch_size_ * patch_size_, ph, pw});

    // Pixel shuffle: (BN, output_dim, H, W)
    feat = torch::pixel_shuffle(feat, patch_size_);

    // Return: (BN, H, W, output_dim)
    return feat.permute({0, 2, 3, 1}).contiguous();
}

// ---------------------------------------------------------------------------
// CameraHead
// ---------------------------------------------------------------------------

CameraHeadImpl::CameraHeadImpl(int dec_embed_dim) {
    register_module("res_conv", res_conv_);
    res_conv_->push_back(ResConvBlock(dec_embed_dim, dec_embed_dim));
    res_conv_->push_back(ResConvBlock(dec_embed_dim, dec_embed_dim));

    more_mlps_ = torch::nn::Sequential(
        torch::nn::Linear(dec_embed_dim, dec_embed_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(dec_embed_dim, dec_embed_dim),
        torch::nn::ReLU()
    );
    register_module("more_mlps", more_mlps_);

    fc_t_   = register_module("fc_t",   torch::nn::Linear(dec_embed_dim, 3));
    fc_rot_ = register_module("fc_rot", torch::nn::Linear(dec_embed_dim, 9));
}

void CameraHeadImpl::load_weights(const TensorStore& ts,
                                  const std::string& prefix) {
    const int n_res = static_cast<int>(res_conv_->size());
    for (int i = 0; i < n_res; ++i) {
        auto& blk = res_conv_->at<ResConvBlockImpl>(i);
        blk.load_weights(ts, prefix + ".res_conv." + std::to_string(i));
    }

    // more_mlps: Sequential with 2 Linear layers (indices 0 and 2 because ReLU is 1)
    auto& seq = *more_mlps_;
    auto* l0 = seq[0]->as<torch::nn::LinearImpl>();
    auto* l2 = seq[2]->as<torch::nn::LinearImpl>();
    TensorStore::copy_weight(l0->weight, ts, prefix + ".more_mlps.0.weight");
    TensorStore::copy_weight_optional(l0->bias, ts, prefix + ".more_mlps.0.bias");
    TensorStore::copy_weight(l2->weight, ts, prefix + ".more_mlps.2.weight");
    TensorStore::copy_weight_optional(l2->bias, ts, prefix + ".more_mlps.2.bias");

    TensorStore::copy_weight(fc_t_->weight,   ts, prefix + ".fc_t.weight");
    TensorStore::copy_weight_optional(fc_t_->bias, ts, prefix + ".fc_t.bias");
    TensorStore::copy_weight(fc_rot_->weight, ts, prefix + ".fc_rot.weight");
    TensorStore::copy_weight_optional(fc_rot_->bias, ts, prefix + ".fc_rot.bias");
}

torch::Tensor CameraHeadImpl::svd_orthogonalize(torch::Tensor m) {
    // m: (B, 3, 3) — convert to SO(3)
    auto orig_dtype = m.dtype();
    m = m.reshape({-1, 3, 3}).to(torch::kFloat32);  // SVD requires float32
    // Normalize rows for numerical stability
    auto m_norm = torch::nn::functional::normalize(
        m, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
    auto m_T = m_norm.transpose(-1, -2);

    // SVD (float32 only on CUDA)
    auto [u, s, vh] = torch::linalg_svd(m_T, /*full_matrices=*/true);  // u:(B,3,3), s:(B,3), vh:(B,3,3)

    // Ensure det=+1 (avoid reflections)
    auto det = torch::linalg_det(torch::bmm(u, vh));
    auto diag_vals = torch::stack({
        torch::ones_like(det),
        torch::ones_like(det),
        det
    }, /*dim=*/-1);  // (B, 3)
    auto D = torch::diag_embed(diag_vals);  // (B, 3, 3)

    auto R = torch::bmm(torch::bmm(u, D), vh);  // (B, 3, 3)
    return R.transpose(-1, -2).to(orig_dtype);  // cast back to original dtype
}

torch::Tensor CameraHeadImpl::convert_pose_to_4x4(torch::Tensor rot,
                                                   torch::Tensor t) {
    // rot: (B, 3, 3), t: (B, 3)
    const int B = rot.size(0);
    auto pose = torch::eye(4, rot.options()).unsqueeze(0).expand({B, -1, -1}).clone();
    pose.index_put_({torch::indexing::Slice(),
                     torch::indexing::Slice(0, 3),
                     torch::indexing::Slice(0, 3)}, rot);
    pose.index_put_({torch::indexing::Slice(),
                     torch::indexing::Slice(0, 3),
                     3}, t);
    return pose;  // (B, 4, 4)
}

torch::Tensor CameraHeadImpl::forward(torch::Tensor feat,
                                      int patch_h, int patch_w) {
    // feat: (B*N, hw, dec_embed_dim)
    const auto input_dtype = feat.scalar_type();  // preserve for final cast
    const int BN  = feat.size(0);
    const int hw  = feat.size(1);
    const int dim = feat.size(2);

    // Apply ResConvBlocks (treating spatial tokens as a sequence)
    for (int i = 0; i < static_cast<int>(res_conv_->size()); ++i)
        feat = res_conv_->at<ResConvBlockImpl>(i).forward(feat);

    // Global average pool over spatial dimension: (B*N, dim)
    auto pooled = feat.mean(1);

    // Additional MLPs
    pooled = more_mlps_->forward(pooled);

    // Run fc_t and fc_rot in float32, matching Python's
    //   with autocast(enabled=False): fc_t(feat.float())
    // In Python the model weights are float32; in C++ model->to(bf16) converts them
    // to bf16, so we must cast both input AND weights to float32 to avoid a dtype
    // mismatch in addmm.
    auto pooled_f32 = pooled.to(torch::kFloat32);
    auto t_w   = fc_t_->weight.to(torch::kFloat32);
    auto rot_w = fc_rot_->weight.to(torch::kFloat32);
    torch::Tensor t_b, rot_b;
    if (fc_t_->bias.defined())   t_b   = fc_t_->bias.to(torch::kFloat32);
    if (fc_rot_->bias.defined()) rot_b = fc_rot_->bias.to(torch::kFloat32);
    auto t   = torch::linear(pooled_f32, t_w,   t_b);    // (B*N, 3)  float32
    auto rot = torch::linear(pooled_f32, rot_w, rot_b);  // (B*N, 9)  float32

    // Orthogonalize rotation
    rot = svd_orthogonalize(rot.reshape({BN, 3, 3}));  // (B*N, 3, 3)

    // Assemble 4×4 SE(3) matrix and cast back to model dtype
    return convert_pose_to_4x4(rot, t).to(input_dtype);  // (B*N, 4, 4)
}

} // namespace loger
