#include <loger/utils/geometry.hpp>

namespace loger {

torch::Tensor homogenize_points(torch::Tensor pts) {
    // pts: (..., 3) → (..., 4) with last coordinate = 1
    auto ones = torch::ones_like(
        pts.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 1)}));
    return torch::cat({pts, ones}, /*dim=*/-1);
}

torch::Tensor se3_inverse(torch::Tensor T) {
    // T: (B, 4, 4) SE(3) → T^-1
    auto R = T.index({torch::indexing::Slice(),
                       torch::indexing::Slice(0, 3),
                       torch::indexing::Slice(0, 3)});
    auto t = T.index({torch::indexing::Slice(),
                       torch::indexing::Slice(0, 3), 3});

    auto Rt = R.transpose(-1, -2);
    auto t_inv = -torch::bmm(Rt, t.unsqueeze(-1)).squeeze(-1);

    const int B = T.size(0);
    auto T_inv = torch::eye(4, T.options()).unsqueeze(0).expand({B, -1, -1}).clone();
    T_inv.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(0, 3),
                       torch::indexing::Slice(0, 3)}, Rt);
    T_inv.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(0, 3), 3}, t_inv);
    return T_inv;
}

torch::Tensor depth_edge(torch::Tensor depth, float rtol) {
    // depth: (B, N, H, W)
    // Returns bool mask same shape — true where depth discontinuity.
    const int B = depth.size(0);
    const int N = depth.size(1);
    const int H = depth.size(2);
    const int W = depth.size(3);

    auto d = depth.reshape({B * N, 1, H, W});  // (BN, 1, H, W)

    // Max pool 3×3 to find local maximum depth
    auto d_max = torch::nn::functional::max_pool2d(
        d, torch::nn::functional::MaxPool2dFuncOptions(3).padding(1).stride(1));

    // Edge where local max differs significantly from original
    auto edge = (d_max - d) > (rtol * d.abs());
    return edge.squeeze(1).reshape({B, N, H, W});
}

torch::Tensor apply_conf_postprocess(torch::Tensor conf, torch::Tensor edges) {
    // conf:  (B, N, H, W, 1)
    // edges: (B, N, H, W) bool
    conf = torch::sigmoid(conf);
    conf.index_put_({edges.unsqueeze(-1).expand_as(conf)},
                    torch::zeros({1}, conf.options()));
    return conf;
}

} // namespace loger
