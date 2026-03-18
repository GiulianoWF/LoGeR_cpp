#pragma once
#include <loger/model/block_rope.hpp>
#include <loger/io/weight_loader.hpp>
#include <torch/torch.h>

namespace loger {

/// Residual linear block (1×1 "conv" with skip connection).
/// Equivalent to Python's ResConvBlock.
class ResConvBlockImpl : public torch::nn::Module {
public:
    ResConvBlockImpl(int dim_in, int dim_out);
    void load_weights(const TensorStore& ts, const std::string& prefix);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear res_conv1_{nullptr}, res_conv2_{nullptr}, res_conv3_{nullptr};
    torch::nn::Linear skip_{nullptr};
    bool need_skip_;
};
TORCH_MODULE(ResConvBlock);

/// Transformer decoder: projects input, runs BlockRope self-attention blocks, projects output.
/// Equivalent to Python's TransformerDecoder.
class TransformerDecoderImpl : public torch::nn::Module {
public:
    TransformerDecoderImpl(int in_dim, int out_dim, int dec_embed_dim = 1024,
                           int depth = 5, int num_heads = 16,
                           RoPE2D rope = nullptr);

    void load_weights(const TensorStore& ts, const std::string& prefix);

    /// hidden: (B*N, hw, in_dim)  — concatenated encoder outputs
    /// xpos:   (B*N, hw, 2)       — optional positions
    /// Returns: (B*N, hw, out_dim)
    torch::Tensor forward(torch::Tensor hidden, torch::Tensor xpos = {});

    /// Optional tag for debug saves (e.g. "point_decoder", "conf_decoder")
    std::string debug_tag_;

private:
    torch::nn::Linear projects_{nullptr};    // in_dim → dec_embed_dim
    torch::nn::ModuleList blocks_;           // depth × BlockRope
    torch::nn::Linear linear_out_{nullptr};  // dec_embed_dim → out_dim
};
TORCH_MODULE(TransformerDecoder);

/// Projects patch tokens to dense 3D point predictions via pixel_shuffle.
/// Equivalent to Python's LinearPts3d.
class LinearPts3dImpl : public torch::nn::Module {
public:
    /// patch_size: upsampling factor (14)
    /// dec_embed_dim: input feature dimension
    /// output_dim: 3 for full xyz, 2 for xy only, 1 for z only
    LinearPts3dImpl(int patch_size, int dec_embed_dim, int output_dim = 3);

    void load_weights(const TensorStore& ts, const std::string& prefix);

    /// tokens: (B*N, hw, dec_embed_dim)
    /// Returns: (B*N, H, W, output_dim)
    torch::Tensor forward(torch::Tensor tokens, int H, int W);

private:
    int patch_size_, output_dim_;
    torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(LinearPts3d);

/// Predicts SE(3) camera poses from decoder features.
/// Equivalent to Python's CameraHead.
class CameraHeadImpl : public torch::nn::Module {
public:
    explicit CameraHeadImpl(int dec_embed_dim = 512);

    void load_weights(const TensorStore& ts, const std::string& prefix);

    /// feat:    (B*N, hw, dec_embed_dim)
    /// Returns: (B*N, 4, 4) SE(3) camera-to-world matrices
    torch::Tensor forward(torch::Tensor feat, int patch_h, int patch_w);

private:
    torch::nn::ModuleList res_conv_;      // 2 × ResConvBlock
    torch::nn::Sequential more_mlps_{nullptr};
    torch::nn::Linear fc_t_{nullptr};    // → 3 (translation)
    torch::nn::Linear fc_rot_{nullptr};  // → 9 (rotation matrix flattened)

    torch::Tensor svd_orthogonalize(torch::Tensor m);
    torch::Tensor convert_pose_to_4x4(torch::Tensor rot, torch::Tensor t);
};
TORCH_MODULE(CameraHead);

} // namespace loger
