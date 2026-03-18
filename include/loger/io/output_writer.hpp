#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>

namespace loger {

class OutputWriter {
public:
    /// Write a PLY point cloud (binary little-endian, XYZ + RGB).
    /// points: (N_total, 3) float, colors: (N_total, 3) uint8, conf: (N_total,) float
    static void write_ply(const std::string& path,
                          torch::Tensor points,
                          torch::Tensor colors,
                          torch::Tensor conf,
                          float conf_threshold = 0.1f);

    /// Write camera trajectory in TUM format: timestamp tx ty tz qx qy qz qw
    static void write_trajectory(const std::string& path,
                                 torch::Tensor camera_poses,   // (N, 4, 4)
                                 const std::vector<float>& timestamps = {});

    /// Write per-frame tensors as a .pt archive loadable by Python's torch.load().
    /// points: (N,H,W,3) float32, conf: (N,H,W,1) float32 [0,1],
    /// images: (N,H,W,3) float32 [0,1], camera_poses: (N,4,4) float32
    static void write_pt(const std::string& path,
                         torch::Tensor points,
                         torch::Tensor conf,
                         torch::Tensor images,
                         torch::Tensor camera_poses);

    /// Write a single float32 tensor to a .npy file (NumPy v1.0 format).
    static void save_npy(const std::string& path, torch::Tensor t);
};

} // namespace loger
