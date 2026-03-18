#pragma once
#include <torch/torch.h>

namespace loger {

/// Add homogeneous coordinate: [..., 3] → [..., 4] with last dim = 1.
torch::Tensor homogenize_points(torch::Tensor pts);

/// Invert SE(3) matrix: (B, 4, 4) → (B, 4, 4).
torch::Tensor se3_inverse(torch::Tensor T);

/// Detect depth discontinuities via max-pooling.
/// depth: (B, N, H, W) — returns bool mask same shape.
torch::Tensor depth_edge(torch::Tensor depth, float rtol = 0.03f);

/// Apply sigmoid and zero out edge pixels in confidence map.
/// conf:  (B, N, H, W, 1)
/// edges: (B, N, H, W) bool
torch::Tensor apply_conf_postprocess(torch::Tensor conf, torch::Tensor edges);

} // namespace loger
