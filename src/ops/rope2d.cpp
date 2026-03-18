#include <loger/ops/rope2d.hpp>
#include <stdexcept>

namespace loger {

// ---------------------------------------------------------------------------
// RoPE2D
// ---------------------------------------------------------------------------

RoPE2DImpl::RoPE2DImpl(float freq, float F0) : base_(freq), F0_(F0) {}

std::pair<torch::Tensor, torch::Tensor>
RoPE2DImpl::get_cos_sin(int D, int seq_len, torch::Device device, torch::Dtype dtype) {
    CacheKey key{D, seq_len, device, dtype};
    auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second;

    // Inverse frequencies: (D/2,)
    auto arange_D = torch::arange(0, D, 2, torch::TensorOptions().dtype(torch::kFloat32));
    auto inv_freq = 1.0f / torch::pow(base_, arange_D / static_cast<float>(D));  // (D/2,)

    // Positions: (seq_len,)
    auto t = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kFloat32));

    // Outer product: (seq_len, D/2)
    auto freqs = torch::outer(t, inv_freq) * F0_;

    // Duplicate to get (seq_len, D)
    auto emb = torch::cat({freqs, freqs}, /*dim=*/-1);

    auto cos = emb.cos().to(dtype).to(device);
    auto sin = emb.sin().to(dtype).to(device);

    cache_[key] = {cos, sin};
    return {cos, sin};
}

// Rotate the last dim by 90°: split in half, negate second, swap.
// x: (..., D) → (..., D)
torch::Tensor RoPE2DImpl::rotate_half(torch::Tensor x) {
    auto half = x.size(-1) / 2;
    auto x1 = x.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, half)});
    auto x2 = x.index({torch::indexing::Ellipsis, torch::indexing::Slice(half)});
    return torch::cat({-x2, x1}, /*dim=*/-1);
}

// tokens: (B, H, N, D)
// pos1d:  (B, N) int64 — position indices
// cos:    (max_seq, D)
// sin:    (max_seq, D)
torch::Tensor RoPE2DImpl::apply_rope1d(torch::Tensor tokens,
                                       torch::Tensor pos1d,
                                       torch::Tensor cos,
                                       torch::Tensor sin) {
    // Look up sin/cos by position index: (B, N, D)
    auto cos_e = torch::embedding(cos, pos1d.reshape({-1}))
                     .reshape({pos1d.size(0), pos1d.size(1), cos.size(1)});
    auto sin_e = torch::embedding(sin, pos1d.reshape({-1}))
                     .reshape({pos1d.size(0), pos1d.size(1), sin.size(1)});

    // Add head dimension: (B, 1, N, D)
    cos_e = cos_e.unsqueeze(1);
    sin_e = sin_e.unsqueeze(1);

    return (tokens * cos_e) + (rotate_half(tokens) * sin_e);
}

torch::Tensor RoPE2DImpl::forward(torch::Tensor tokens, torch::Tensor positions) {
    // tokens:    (B, H, N, D)
    // positions: (B, N, 2) — (y, x) integer coords
    if (!positions.defined() || positions.numel() == 0)
        return tokens;

    const int D    = tokens.size(-1);
    const int half = D / 2;
    int seq_len    = positions.max().item<int>() + 1;
    // Guard against seq_len being too small
    if (seq_len < 1) seq_len = 1;

    auto [cos, sin] = get_cos_sin(half, seq_len,
                                  tokens.device(), tokens.scalar_type());

    // Split tokens into y-half and x-half along last dim
    auto tok_y = tokens.index({torch::indexing::Ellipsis,
                                torch::indexing::Slice(0, half)});
    auto tok_x = tokens.index({torch::indexing::Ellipsis,
                                torch::indexing::Slice(half)});

    // positions: (B, N, 2) → separate y and x as (B, N) int64
    auto pos_y = positions.index({torch::indexing::Ellipsis, 0}).to(torch::kInt64);
    auto pos_x = positions.index({torch::indexing::Ellipsis, 1}).to(torch::kInt64);

    tok_y = apply_rope1d(tok_y, pos_y, cos, sin);
    tok_x = apply_rope1d(tok_x, pos_x, cos, sin);

    return torch::cat({tok_y, tok_x}, /*dim=*/-1);
}

// ---------------------------------------------------------------------------
// PositionGetter
// ---------------------------------------------------------------------------

torch::Tensor PositionGetter::operator()(int B, int h, int w, torch::Device device) {
    auto cache_key = std::make_pair(h, w);
    auto it = cache_.find(cache_key);

    if (it == cache_.end()) {
        // Build (h*w, 2) grid of (y, x) integer coordinates
        auto ys = torch::arange(h, torch::TensorOptions().dtype(torch::kInt64));
        auto xs = torch::arange(w, torch::TensorOptions().dtype(torch::kInt64));
        auto grid_y = ys.unsqueeze(1).expand({h, w}).reshape({-1});
        auto grid_x = xs.unsqueeze(0).expand({h, w}).reshape({-1});
        auto pos = torch::stack({grid_y, grid_x}, /*dim=*/-1);  // (h*w, 2)
        cache_[cache_key] = pos;
        it = cache_.find(cache_key);
    }

    // Replicate to batch: (B, h*w, 2)
    return it->second.to(device).unsqueeze(0).expand({B, -1, -1});
}

} // namespace loger
