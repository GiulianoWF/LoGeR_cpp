#pragma once
#include <map>
#include <utility>
#include <torch/torch.h>

namespace loger {

/// 2D Rotary Position Embedding.
/// Splits the feature dim in half: applies 1D RoPE to the y-half using y-coords,
/// and 1D RoPE to the x-half using x-coords, then concatenates.
class RoPE2DImpl : public torch::nn::Module {
public:
    explicit RoPE2DImpl(float freq = 100.0f, float F0 = 1.0f);

    /// tokens:    (B, num_heads, N, D)
    /// positions: (B, N, 2)  — integer (y, x) grid coords
    /// Returns:   (B, num_heads, N, D) with RoPE applied
    torch::Tensor forward(torch::Tensor tokens, torch::Tensor positions);

private:
    float base_;
    float F0_;

    struct CacheKey {
        int D, seq_len;
        torch::Device device;
        torch::Dtype dtype;
        bool operator<(const CacheKey& o) const {
            if (D != o.D) return D < o.D;
            if (seq_len != o.seq_len) return seq_len < o.seq_len;
            if (device != o.device) {
                if (device.type() != o.device.type())
                    return device.type() < o.device.type();
                return device.index() < o.device.index();
            }
            return dtype < o.dtype;
        }
    };
    std::map<CacheKey, std::pair<torch::Tensor, torch::Tensor>> cache_;

    std::pair<torch::Tensor, torch::Tensor>
    get_cos_sin(int D, int seq_len, torch::Device device, torch::Dtype dtype);

    static torch::Tensor rotate_half(torch::Tensor x);

    torch::Tensor apply_rope1d(torch::Tensor tokens,   // (B, H, N, D)
                               torch::Tensor pos1d,    // (B, N) int64
                               torch::Tensor cos,      // (max_seq, D)
                               torch::Tensor sin);     // (max_seq, D)
};
TORCH_MODULE(RoPE2D);

/// Generates and caches (B, h*w, 2) integer position grids.
class PositionGetter {
public:
    /// Returns (B, h*w, 2) tensor with (y, x) coordinates, replicated B times.
    torch::Tensor operator()(int B, int h, int w, torch::Device device);

private:
    std::map<std::pair<int,int>, torch::Tensor> cache_;
};

} // namespace loger
