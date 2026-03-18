#include <loger/utils/windowing.hpp>
#include <loger/utils/geometry.hpp>
#include <stdexcept>

namespace loger {

std::vector<int> compute_window_starts(int N, int window_size, int overlap_size) {
    if (window_size <= 0 || window_size >= N) return {0};
    std::vector<int> starts;
    int step = window_size - overlap_size;
    if (step <= 0) throw std::invalid_argument("overlap_size must be < window_size");
    for (int s = 0; s < N; s += step) {
        starts.push_back(s);
        if (s + window_size >= N) break;
    }
    return starts;
}

WindowPrediction merge_simple(const std::vector<WindowPrediction>& preds,
                               int window_size, int overlap_size) {
    // Drop overlapping prefix of each window (except first) and concatenate.
    if (preds.empty()) throw std::runtime_error("merge_simple: empty predictions");

    std::vector<torch::Tensor> pts, lpts, conf, poses;
    for (size_t i = 0; i < preds.size(); ++i) {
        int skip = (i == 0) ? 0 : overlap_size;
        using S = torch::indexing::Slice;
        pts  .push_back(preds[i].points      .index({S(), S(skip)}));
        lpts .push_back(preds[i].local_points .index({S(), S(skip)}));
        conf .push_back(preds[i].conf         .index({S(), S(skip)}));
        poses.push_back(preds[i].camera_poses .index({S(), S(skip)}));
    }
    return {
        torch::cat(pts,   1),
        torch::cat(lpts,  1),
        torch::cat(conf,  1),
        torch::cat(poses, 1)
    };
}

// ---------------------------------------------------------------------------
// SE3 alignment
// ---------------------------------------------------------------------------

SE3AlignResult estimate_se3(const WindowPrediction& prev_aligned,
                            const WindowPrediction& curr_raw,
                            int overlap_size,
                            bool allow_scale,
                            const std::string& scale_mode) {
    // Take the last `overlap_size` frames of prev and first `overlap_size` of curr.
    using S = torch::indexing::Slice;
    const int B = prev_aligned.camera_poses.size(0);

    // Camera poses from overlap region
    auto prev_poses = prev_aligned.camera_poses
                          .index({S(), S(-overlap_size, torch::indexing::None)});
    auto curr_poses = curr_raw.camera_poses
                          .index({S(), S(0, overlap_size)});

    // Translations
    auto prev_t = prev_poses.index({S(), S(), S(0,3), 3});  // (B, ov, 3)
    auto curr_t = curr_poses.index({S(), S(), S(0,3), 3});

    SE3AlignResult T;
    T.scale = torch::ones({B}, prev_t.options());
    T.t     = torch::zeros({B, 3}, prev_t.options());
    T.R     = torch::eye(3, prev_t.options()).unsqueeze(0).expand({B,-1,-1});

    if (allow_scale && scale_mode == "median") {
        // Estimate scale from depth ratio of overlap translations
        auto prev_depth = prev_t.norm(2, -1);  // (B, ov)
        auto curr_depth = curr_t.norm(2, -1);  // (B, ov)
        auto valid = (curr_depth > 1e-6f) & (prev_depth > 1e-6f);

        for (int b = 0; b < B; ++b) {
            auto ratio = (prev_depth[b] / curr_depth[b].clamp_min(1e-6f))
                             .masked_select(valid[b]);
            if (ratio.numel() > 0) {
                auto sorted = std::get<0>(ratio.sort());
                T.scale[b] = sorted[sorted.size(0) / 2];
            }
        }
    }

    // Estimate relative rotation from the first overlap frame pair
    // R_rel = R_prev @ R_curr^T (aligns curr frame to prev)
    auto prev_R = prev_poses.index({S(), 0, S(0,3), S(0,3)});  // (B,3,3)
    auto curr_R = curr_poses.index({S(), 0, S(0,3), S(0,3)});
    T.R = torch::bmm(prev_R, curr_R.transpose(-1,-2));

    // Translation: t_aligned = t_prev[0] - scale * R_rel @ t_curr[0]
    auto prev_t0 = prev_t.index({S(), 0, S()});  // (B, 3)
    auto curr_t0 = curr_t.index({S(), 0, S()});
    T.t = prev_t0 - T.scale.unsqueeze(-1)
          * torch::bmm(T.R, curr_t0.unsqueeze(-1)).squeeze(-1);

    return T;
}

WindowPrediction apply_se3(const WindowPrediction& pred,
                           const SE3AlignResult& T,
                           bool allow_scale) {
    WindowPrediction out = pred;

    const int B = pred.camera_poses.size(0);
    const int N = pred.camera_poses.size(1);

    // Transform camera poses: new_pose = T * curr_pose
    // T is a (B, 4, 4) rigid transform
    auto T4 = torch::eye(4, pred.camera_poses.options())
                  .unsqueeze(0).expand({B, -1, -1}).clone();

    T4.index_put_({torch::indexing::Slice(),
                   torch::indexing::Slice(0,3),
                   torch::indexing::Slice(0,3)}, T.R);
    T4.index_put_({torch::indexing::Slice(),
                   torch::indexing::Slice(0,3), 3},
                  T.t);
    if (allow_scale)
        T4.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(0,3), 3},
                      T.t * T.scale.unsqueeze(-1));

    // Apply to each frame: (B, N, 4, 4) = T4.unsqueeze(1) @ poses
    auto T4_exp = T4.unsqueeze(1).expand({B, N, -1, -1});
    out.camera_poses = torch::bmm(
        T4_exp.reshape({B*N,4,4}),
        pred.camera_poses.reshape({B*N,4,4})
    ).reshape({B, N, 4, 4});

    // Scale global points if allow_scale
    if (allow_scale) {
        auto scale_view = T.scale.reshape({B,1,1,1,1});
        out.points = pred.points * scale_view;
    }

    // Re-derive global points from transformed poses and local points
    auto hom = homogenize_points(out.local_points);  // (B,N,H,W,4)
    // einsum "bnij, bnhwj -> bnhwi"
    out.points = torch::einsum("bnij,bnhwj->bnhwi",
        {out.camera_poses, hom})
        .index({torch::indexing::Ellipsis, torch::indexing::Slice(0,3)});

    return out;
}

WindowPrediction merge_se3(std::vector<WindowPrediction>& preds,
                           int overlap_size, bool allow_scale) {
    if (preds.empty()) throw std::runtime_error("merge_se3: empty predictions");
    if (preds.size() == 1) return preds[0];

    // Align each window to the previous, then drop overlapping prefix
    std::vector<torch::Tensor> pts, lpts, conf, poses;

    auto aligned = preds[0];
    using S = torch::indexing::Slice;

    pts  .push_back(aligned.points.index({S(), S()}));
    lpts .push_back(aligned.local_points.index({S(), S()}));
    conf .push_back(aligned.conf.index({S(), S()}));
    poses.push_back(aligned.camera_poses.index({S(), S()}));

    for (size_t i = 1; i < preds.size(); ++i) {
        auto T = estimate_se3(aligned, preds[i], overlap_size, allow_scale);
        auto curr_aligned = apply_se3(preds[i], T, allow_scale);

        // Drop first `overlap_size` frames (they overlap with previous window)
        pts  .push_back(curr_aligned.points.index({S(), S(overlap_size)}));
        lpts .push_back(curr_aligned.local_points.index({S(), S(overlap_size)}));
        conf .push_back(curr_aligned.conf.index({S(), S(overlap_size)}));
        poses.push_back(curr_aligned.camera_poses.index({S(), S(overlap_size)}));

        // Update aligned = full current window (for next iteration's overlap ref)
        aligned = curr_aligned;
    }

    return {
        torch::cat(pts,   1),
        torch::cat(lpts,  1),
        torch::cat(conf,  1),
        torch::cat(poses, 1)
    };
}

} // namespace loger
