#pragma once
#include <torch/torch.h>

namespace loger {

/// Standard MLP with GELU activation (used in DINOv2 encoder blocks).
class MlpImpl : public torch::nn::Module {
public:
    MlpImpl(int in_features, int hidden_features = -1, int out_features = -1,
            bool bias = true);

    void load_weights(const class TensorStore& ts, const std::string& prefix);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};
TORCH_MODULE(Mlp);

/// SwiGLU MLP: hidden = silu(fc1(x)) * fc2(x); out = fc3(hidden).
/// Used in decoder BlockRope layers.
class SwiGLUMlpImpl : public torch::nn::Module {
public:
    SwiGLUMlpImpl(int in_features, int hidden_features = -1, bool bias = true);

    void load_weights(const class TensorStore& ts, const std::string& prefix);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc3_{nullptr};
};
TORCH_MODULE(SwiGLUMlp);

/// LayerScale: element-wise scale by a learnable gamma parameter.
class LayerScaleImpl : public torch::nn::Module {
public:
    explicit LayerScaleImpl(int dim, float init_value = 1.0f);

    void load_weights(const class TensorStore& ts, const std::string& prefix);

    torch::Tensor forward(torch::Tensor x);

    torch::Tensor gamma;
};
TORCH_MODULE(LayerScale);

} // namespace loger
