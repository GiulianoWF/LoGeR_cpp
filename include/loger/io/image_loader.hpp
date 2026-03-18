#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>

namespace loger {

class ImageLoader {
public:
    static constexpr int PIXEL_LIMIT = 255000;
    static constexpr int PATCH_SIZE  = 14;

    /// Collect image paths from a directory (sorted) or extract from video.
    static std::vector<std::string> collect_paths(
        const std::string& input_path,
        int start_frame = 0,
        int end_frame   = -1,
        int stride      = 1);

    /// Load images, resize to a uniform (TARGET_W, TARGET_H) that is:
    ///   - a multiple of PATCH_SIZE in both dimensions
    ///   - total pixels <= PIXEL_LIMIT
    /// Returns: (N, 3, H, W) float32 in [0, 1]
    static torch::Tensor load_and_preprocess(
        const std::vector<std::string>& paths,
        int target_w = 0,   // 0 = auto from first image
        int target_h = 0);

    /// Compute target resolution (multiples of PATCH_SIZE, within PIXEL_LIMIT).
    static std::pair<int,int> compute_target_size(int W_orig, int H_orig,
                                                  int pixel_limit = PIXEL_LIMIT);
};

} // namespace loger
