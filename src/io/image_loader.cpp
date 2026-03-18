#include <loger/io/image_loader.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

namespace loger {

static bool is_image_ext(const std::string& ext) {
    static const std::vector<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
    };
    std::string low = ext;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);
    return std::find(exts.begin(), exts.end(), low) != exts.end();
}

std::vector<std::string> ImageLoader::collect_paths(
    const std::string& input_path,
    int start_frame, int end_frame, int stride) {

    std::vector<std::string> paths;
    fs::path p(input_path);

    if (fs::is_directory(p)) {
        // Collect and sort image files
        for (const auto& entry : fs::directory_iterator(p)) {
            if (entry.is_regular_file() &&
                is_image_ext(entry.path().extension().string()))
                paths.push_back(entry.path().string());
        }
        std::sort(paths.begin(), paths.end());
    } else {
        // Treat as video — extract frames
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened())
            throw std::runtime_error("ImageLoader: cannot open video: " + input_path);

        int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        for (int fi = 0; fi < total; ++fi) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;
            // Save to temp path (simplification: could also keep in memory)
            std::string tmp = "/tmp/loger_frame_" + std::to_string(fi) + ".png";
            cv::imwrite(tmp, frame);
            paths.push_back(tmp);
        }
    }

    // Apply frame range and stride
    if (end_frame > 0 && end_frame < static_cast<int>(paths.size()))
        paths.resize(end_frame);
    if (start_frame > 0)
        paths.erase(paths.begin(),
                    paths.begin() + std::min(start_frame,
                                             static_cast<int>(paths.size())));

    if (stride > 1) {
        std::vector<std::string> strided;
        for (size_t i = 0; i < paths.size(); i += stride)
            strided.push_back(paths[i]);
        paths = strided;
    }

    std::cout << "[ImageLoader] Found " << paths.size() << " images\n";
    return paths;
}

std::pair<int,int> ImageLoader::compute_target_size(int W_orig, int H_orig,
                                                     int pixel_limit) {
    // Scale down to fit pixel_limit while keeping aspect ratio
    float scale = std::sqrt(static_cast<float>(pixel_limit)
                             / static_cast<float>(W_orig * H_orig));
    if (scale > 1.0f) scale = 1.0f;

    int W = static_cast<int>(std::round(W_orig * scale / PATCH_SIZE)) * PATCH_SIZE;
    int H = static_cast<int>(std::round(H_orig * scale / PATCH_SIZE)) * PATCH_SIZE;

    W = std::max(W, PATCH_SIZE);
    H = std::max(H, PATCH_SIZE);

    return {W, H};
}

torch::Tensor ImageLoader::load_and_preprocess(
    const std::vector<std::string>& paths,
    int target_w, int target_h) {

    if (paths.empty()) throw std::runtime_error("ImageLoader: no paths provided");

    // Determine target size from first image if not given
    if (target_w <= 0 || target_h <= 0) {
        cv::Mat first = cv::imread(paths[0]);
        if (first.empty())
            throw std::runtime_error("ImageLoader: cannot read: " + paths[0]);
        auto [tw, th] = compute_target_size(first.cols, first.rows);
        target_w = tw;
        target_h = th;
    }

    std::cout << "[ImageLoader] Resizing to " << target_w << "x" << target_h << "\n";

    const int N = static_cast<int>(paths.size());
    auto tensor = torch::zeros({N, 3, target_h, target_w}, torch::kFloat32);

    for (int i = 0; i < N; ++i) {
        cv::Mat img = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[ImageLoader] Warning: empty image: " << paths[i] << "\n";
            continue;
        }

        // Resize
        cv::Mat resized;
        cv::resize(img, resized, {target_w, target_h}, 0, 0, cv::INTER_LINEAR);

        // BGR → RGB, uint8 → float32 / 255
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        // Copy to tensor: (3, H, W)
        for (int c = 0; c < 3; ++c) {
            cv::Mat ch;
            cv::extractChannel(rgb, ch, c);
            auto slice = tensor[i][c];
            std::memcpy(slice.data_ptr<float>(), ch.data,
                        target_h * target_w * sizeof(float));
        }
    }

    return tensor;  // (N, 3, H, W)
}

} // namespace loger
