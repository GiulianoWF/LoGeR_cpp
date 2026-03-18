#pragma once
#include <loger/io/weight_loader.hpp>
#include <torch/torch.h>
#include <optional>

namespace loger {

/// Simple RMSNorm — torch::nn::RMSNorm is not exposed in the libtorch C++ API.
class RMSNormImpl : public torch::nn::Module {
public:
    explicit RMSNormImpl(int dim, double eps = 1e-5)
        : eps_(eps) {
        weight = register_parameter("weight", torch::ones({dim}));
    }
    torch::Tensor forward(torch::Tensor x) {
        auto rms = x.to(torch::kFloat32).pow(2).mean(-1, /*keepdim=*/true).add(eps_).sqrt();
        return (x / rms.to(x.dtype())) * weight.to(x.dtype());
    }
    torch::Tensor weight;
private:
    double eps_;
};
TORCH_MODULE(RMSNorm);

/// Per-batch TTT fast-weight state for one FastWeightGluMLPMultihead layer.
struct TTTState {
    torch::Tensor w0;  // (B * num_heads, head_dim, inter_dim)
    torch::Tensor w1;  // (B * num_heads, inter_dim, head_dim)
    torch::Tensor w2;  // (B * num_heads, head_dim, inter_dim)
    // Weight norms (for re-normalization after update)
    torch::Tensor w0_norm;
    torch::Tensor w1_norm;
    torch::Tensor w2_norm;
    // Optional momentum buffers
    torch::Tensor m0, m1, m2;

    bool defined() const { return w0.defined(); }
};

/// Newton-Schulz orthogonalization (5 steps by default).
/// G: (B, D_in, D_out) — returns same shape, approximately orthonormal.
torch::Tensor zeropower_newtonschulz5(torch::Tensor G, int steps = 5);

/// SiLU backward: dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
torch::Tensor silu_backprop(torch::Tensor dy, torch::Tensor x_pre_act);

/// Test-Time Training layer using fast GLU-MLP with multi-head fast weights.
/// Equivalent to Python's FastWeightGluMLPMultihead.
class FastWeightGluMLPMultiheadImpl : public torch::nn::Module {
public:
    FastWeightGluMLPMultiheadImpl(int dim, int head_dim, int inter_multi = 4,
                                  int muon_update_steps = 5,
                                  int ttt_update_steps = 1,
                                  bool use_momentum = false,
                                  bool pre_norm = false);

    void load_weights(const TensorStore& ts, const std::string& prefix);

    /// x:     (B, L, dim)  — token sequence
    /// state: current fast-weight state (or undefined for initialization)
    /// Returns: (output (B, L, dim), updated TTTState)
    std::pair<torch::Tensor, TTTState>
    forward(torch::Tensor x, TTTState state);

    /// Initialize state from base weights (expanded to batch B).
    TTTState init_state(int B, torch::Device device, torch::Dtype dtype) const;

    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }

    // Base fast weights (trained offline, shape: num_heads x head_dim x inter_dim etc.)
    torch::Tensor w0, w1, w2;

private:
    int dim_, num_heads_, head_dim_, inter_dim_;
    int muon_steps_, ttt_update_steps_;
    bool use_momentum_, pre_norm_;

    // Learnable projections
    torch::nn::Linear to_qkv_{nullptr};   // (dim, 3*dim)
    torch::nn::Linear c_proj_{nullptr};   // (dim, dim)
    torch::nn::Linear lr_fc_{nullptr};    // (dim, num_heads*3) — lr for w0,w1,w2
    float base_lr_inv_;
    RMSNorm o_norm_{nullptr};  // (head_dim,)
    RMSNorm pre_norm_ln_{nullptr};

    std::pair<torch::Tensor, TTTState>
    apply_and_update(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                     torch::Tensor lr, TTTState state);
};
TORCH_MODULE(FastWeightGluMLPMultihead);

} // namespace loger
