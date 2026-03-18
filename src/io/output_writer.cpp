#include <loger/io/output_writer.hpp>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>

namespace loger {

void OutputWriter::write_ply(const std::string& path,
                              torch::Tensor points,
                              torch::Tensor colors,
                              torch::Tensor conf,
                              float conf_threshold) {
    // Filter by confidence
    auto mask   = conf > conf_threshold;
    auto pts_f  = points.index({mask});   // (K, 3)
    auto col_f  = colors.index({mask});   // (K, 3)

    const int64_t K = pts_f.size(0);
    pts_f = pts_f.to(torch::kFloat32).contiguous().cpu();
    col_f = col_f.to(torch::kUInt8).contiguous().cpu();

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("OutputWriter: cannot open: " + path);

    // PLY header
    f << "ply\n"
      << "format binary_little_endian 1.0\n"
      << "element vertex " << K << "\n"
      << "property float x\n"
      << "property float y\n"
      << "property float z\n"
      << "property uchar red\n"
      << "property uchar green\n"
      << "property uchar blue\n"
      << "end_header\n";

    // Binary data: x y z r g b per vertex — write in chunks for speed
    auto pts_acc = pts_f.accessor<float, 2>();
    auto col_acc = col_f.accessor<uint8_t, 2>();

    const int64_t CHUNK = 100000;
    // 15 bytes per vertex: 3 floats (12) + 3 uint8 (3)
    std::vector<char> buf(static_cast<size_t>(std::min(K, CHUNK)) * 15);

    for (int64_t base = 0; base < K; base += CHUNK) {
        int64_t end = std::min(base + CHUNK, K);
        size_t buf_pos = 0;
        for (int64_t i = base; i < end; ++i) {
            float xyz[3] = {pts_acc[i][0], pts_acc[i][1], pts_acc[i][2]};
            std::memcpy(&buf[buf_pos], xyz, 12);  buf_pos += 12;
            buf[buf_pos++] = static_cast<char>(col_acc[i][0]);
            buf[buf_pos++] = static_cast<char>(col_acc[i][1]);
            buf[buf_pos++] = static_cast<char>(col_acc[i][2]);
        }
        f.write(buf.data(), static_cast<std::streamsize>(buf_pos));

        std::cout << "\r[ply] Writing vertices: "
                  << std::min(base + CHUNK, K) << "/" << K << std::flush;
    }
    std::cout << "\n";

    f.close();
}

static torch::Tensor rotation_to_quaternion(torch::Tensor R) {
    // R: (N, 3, 3) → quaternion (N, 4) in xyzw order
    // Using Shepperd's method
    const int N = R.size(0);
    R = R.to(torch::kFloat32).cpu();
    auto q = torch::zeros({N, 4});
    auto R_acc = R.accessor<float, 3>();
    auto q_acc = q.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        float r00=R_acc[i][0][0], r01=R_acc[i][0][1], r02=R_acc[i][0][2];
        float r10=R_acc[i][1][0], r11=R_acc[i][1][1], r12=R_acc[i][1][2];
        float r20=R_acc[i][2][0], r21=R_acc[i][2][1], r22=R_acc[i][2][2];

        float trace = r00 + r11 + r22;
        float qw, qx, qy, qz;

        if (trace > 0) {
            float s = 0.5f / std::sqrt(trace + 1.0f);
            qw = 0.25f / s;
            qx = (r21 - r12) * s;
            qy = (r02 - r20) * s;
            qz = (r10 - r01) * s;
        } else if (r00 > r11 && r00 > r22) {
            float s = 2.0f * std::sqrt(1.0f + r00 - r11 - r22);
            qw = (r21 - r12) / s;
            qx = 0.25f * s;
            qy = (r01 + r10) / s;
            qz = (r02 + r20) / s;
        } else if (r11 > r22) {
            float s = 2.0f * std::sqrt(1.0f + r11 - r00 - r22);
            qw = (r02 - r20) / s;
            qx = (r01 + r10) / s;
            qy = 0.25f * s;
            qz = (r12 + r21) / s;
        } else {
            float s = 2.0f * std::sqrt(1.0f + r22 - r00 - r11);
            qw = (r10 - r01) / s;
            qx = (r02 + r20) / s;
            qy = (r12 + r21) / s;
            qz = 0.25f * s;
        }
        q_acc[i][0] = qx; q_acc[i][1] = qy;
        q_acc[i][2] = qz; q_acc[i][3] = qw;
    }
    return q;
}

void OutputWriter::write_trajectory(const std::string& path,
                                     torch::Tensor camera_poses,
                                     const std::vector<float>& timestamps) {
    // camera_poses: (N, 4, 4)
    const int N = camera_poses.size(0);
    auto poses = camera_poses.to(torch::kFloat32).cpu();

    auto R = poses.index({torch::indexing::Slice(),
                           torch::indexing::Slice(0,3),
                           torch::indexing::Slice(0,3)});  // (N,3,3)
    auto t = poses.index({torch::indexing::Slice(),
                           torch::indexing::Slice(0,3), 3});  // (N,3)

    auto quats = rotation_to_quaternion(R);  // (N, 4) xyzw
    auto t_acc = t.accessor<float, 2>();
    auto q_acc = quats.accessor<float, 2>();

    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("OutputWriter: cannot open: " + path);

    f << "# timestamp tx ty tz qx qy qz qw\n";
    f << std::fixed << std::setprecision(6);

    for (int i = 0; i < N; ++i) {
        float ts = (i < static_cast<int>(timestamps.size())) ?
                   timestamps[i] : static_cast<float>(i);
        f << ts << " "
          << t_acc[i][0] << " " << t_acc[i][1] << " " << t_acc[i][2] << " "
          << q_acc[i][0] << " " << q_acc[i][1] << " "
          << q_acc[i][2] << " " << q_acc[i][3] << "\n";
    }
}

// Strip extension: "output.pt" → "output"
static std::string strip_ext(const std::string& path) {
    auto pos = path.rfind('.');
    return (pos != std::string::npos) ? path.substr(0, pos) : path;
}

// Write a float32 tensor to a .npy file (NumPy v1.0 format).
// Python loads it with np.load(path) — no torch dependency needed.
void OutputWriter::save_npy(const std::string& path, torch::Tensor t) {
    t = t.cpu().to(torch::kFloat32).contiguous();

    // Build shape tuple string: "(50, 434, 574, 3, )"
    std::string shape_str = "(";
    for (int i = 0; i < t.dim(); ++i) {
        shape_str += std::to_string(t.size(i)) + ", ";
    }
    shape_str += ")";

    // Header dict required by the NPY spec
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";

    // Pad with spaces so that (10 + HEADER_LEN) is a multiple of 64; end with '\n'
    size_t raw_total = 10 + hdr.size() + 1;  // 10 = magic(6)+ver(2)+len(2), +1 for '\n'
    size_t padded    = ((raw_total + 63) / 64) * 64;
    hdr += std::string(padded - raw_total, ' ') + '\n';

    uint16_t hdr_len = static_cast<uint16_t>(hdr.size());

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("save_npy: cannot open " + path);

    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
    f.write(reinterpret_cast<const char*>(magic), 8);
    f.write(reinterpret_cast<const char*>(&hdr_len), 2);
    f.write(hdr.c_str(), hdr_len);
    f.write(reinterpret_cast<const char*>(t.data_ptr<float>()),
            t.numel() * sizeof(float));
}

void OutputWriter::write_pt(const std::string& path,
                             torch::Tensor points,
                             torch::Tensor conf,
                             torch::Tensor images,
                             torch::Tensor camera_poses) {
    // Saves as 4 .npy files loadable with np.load() in Python.
    // e.g. path="output.pt" → output_points.npy, output_conf.npy, ...
    std::string base = strip_ext(path);
    const char* names[] = {"points", "conf", "images", "poses"};
    torch::Tensor tensors[] = {points, conf, images, camera_poses};
    for (int i = 0; i < 4; ++i) {
        std::string npy_path = base + "_" + names[i] + ".npy";
        std::cout << "[npy] Writing " << names[i] << "..." << std::flush;
        save_npy(npy_path, tensors[i]);
        std::cout << " done\n";
    }
}

} // namespace loger
