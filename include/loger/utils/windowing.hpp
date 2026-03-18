#pragma once
#include <torch/torch.h>
#include <vector>

namespace loger {

struct WindowPrediction {
    torch::Tensor points;        // (B, N_w, H, W, 3)
    torch::Tensor local_points;  // (B, N_w, H, W, 3)
    torch::Tensor conf;          // (B, N_w, H, W, 1)
    torch::Tensor camera_poses;  // (B, N_w, 4, 4)

    /// Move all tensors to a device (e.g. torch::kCPU to offload from GPU).
    WindowPrediction to(torch::Device dev) const {
        return {points.to(dev), local_points.to(dev),
                conf.to(dev), camera_poses.to(dev)};
    }
};

/// Compute window start indices given total frames, window_size, overlap_size.
std::vector<int> compute_window_starts(int N, int window_size, int overlap_size);

/// Merge predictions from all windows by simple concatenation (drop overlaps).
WindowPrediction merge_simple(const std::vector<WindowPrediction>& preds,
                              int window_size, int overlap_size);

/// SE(3) rigid-body alignment between overlap regions.
/// Transforms curr_pred's camera_poses and points to align with prev_aligned.
struct SE3AlignResult {
    torch::Tensor scale;   // (B,) scale factor
    torch::Tensor R;       // (B, 3, 3) rotation
    torch::Tensor t;       // (B, 3) translation
};

SE3AlignResult estimate_se3(const WindowPrediction& prev_aligned,
                            const WindowPrediction& curr_raw,
                            int overlap_size,
                            bool allow_scale = false,
                            const std::string& scale_mode = "median");

WindowPrediction apply_se3(const WindowPrediction& pred,
                           const SE3AlignResult& T,
                           bool allow_scale = false);

/// Full SE3-aligned merge over all windows.
WindowPrediction merge_se3(std::vector<WindowPrediction>& preds,
                           int overlap_size, bool allow_scale = false);

} // namespace loger
