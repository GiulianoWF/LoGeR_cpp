#include <loger/model/pi3.hpp>
#include <loger/io/image_loader.hpp>
#include <loger/io/output_writer.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <chrono>

// ---------------------------------------------------------------------------
// Simple argument parser
// ---------------------------------------------------------------------------

struct Args {
    std::string input        = "data/examples/office";
    std::string model_name   = "ckpts/LoGeR/latest.pt";
    std::string config       = "ckpts/LoGeR/original_config.yaml";
    std::string output_ply   = "output.ply";
    std::string output_traj  = "trajectory.txt";
    std::string output_pt    = "";
    int window_size          = 32;
    int overlap_size         = 3;
    int start_frame          = 0;
    int end_frame            = -1;
    int stride               = 1;
    float conf_threshold     = 0.1f;
    bool no_ttt              = false;
    bool no_swa              = false;
    bool se3                 = true;
    bool cpu                 = false;
    std::string save_windows = "";
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --input PATH          Input image directory or video\n"
              << "  --model_name PATH     Checkpoint .pt file\n"
              << "  --config PATH         Config YAML file\n"
              << "  --output_ply PATH     Output PLY point cloud\n"
              << "  --output_traj PATH    Output trajectory TXT\n"
              << "  --window_size N       Sliding window size (default: 32)\n"
              << "  --overlap_size N      Window overlap (default: 3)\n"
              << "  --start_frame N       Start frame index\n"
              << "  --end_frame N         End frame index (-1 = all)\n"
              << "  --stride N            Frame stride\n"
              << "  --conf_threshold F    Confidence filter (default: 0.1)\n"
              << "  --no_ttt              Disable test-time training\n"
              << "  --no_swa              Disable sliding window attention\n"
              << "  --no_se3              Disable SE3 alignment\n"
              << "  --output_pt PATH      Output per-frame .pt for visualizer\n"
              << "  --cpu                 Force CPU inference\n"
              << "  --save_windows DIR    Save raw per-window predictions as .npy\n"
              << "  --help                Show this message\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };
        if      (arg == "--input")          a.input         = next();
        else if (arg == "--model_name")     a.model_name    = next();
        else if (arg == "--config")         a.config        = next();
        else if (arg == "--output_ply")     a.output_ply    = next();
        else if (arg == "--output_traj")    a.output_traj   = next();
        else if (arg == "--output_pt")      a.output_pt     = next();
        else if (arg == "--window_size")    a.window_size   = std::stoi(next());
        else if (arg == "--overlap_size")   a.overlap_size  = std::stoi(next());
        else if (arg == "--start_frame")    a.start_frame   = std::stoi(next());
        else if (arg == "--end_frame")      a.end_frame     = std::stoi(next());
        else if (arg == "--stride")         a.stride        = std::stoi(next());
        else if (arg == "--conf_threshold") a.conf_threshold= std::stof(next());
        else if (arg == "--no_ttt")         a.no_ttt        = true;
        else if (arg == "--no_swa")         a.no_swa        = true;
        else if (arg == "--no_se3")         a.se3           = false;
        else if (arg == "--cpu")            a.cpu           = true;
        else if (arg == "--save_windows")   a.save_windows  = next();
        else if (arg == "--help") { print_usage(argv[0]); std::exit(0); }
        else std::cerr << "Warning: unknown argument: " << arg << "\n";
    }
    return a;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    // Device selection
    torch::Device device = torch::kCPU;
    if (!args.cpu && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "[main] Using CUDA device\n";
    } else {
        std::cout << "[main] Using CPU\n";
    }

    // Inference dtype
    torch::Dtype dtype = torch::kFloat32;
    if (device.is_cuda()) {
        // Use bfloat16 on Ampere+ (compute capability >= 8.0)
        // For RTX 5090 (sm_120) bfloat16 is well supported
        dtype = torch::kBFloat16;
        std::cout << "[main] Using bfloat16\n";
    }

    // Load config
    auto cfg = loger::load_config(args.config);
    cfg.se3_align = args.se3;

    // Build and load model
    std::cout << "[main] Building model...\n";
    loger::Pi3 model(cfg);
    model->load_weights(args.model_name);
    model->to(device);
    model->to(dtype);
    model->eval();

    // Load images
    auto paths = loger::ImageLoader::collect_paths(
        args.input, args.start_frame, args.end_frame, args.stride);
    if (paths.empty()) {
        std::cerr << "[main] No images found in: " << args.input << "\n";
        return 1;
    }

    auto imgs = loger::ImageLoader::load_and_preprocess(paths);
    // imgs: (N, 3, H, W) float32 in [0,1]

    // Move to device and dtype, add batch dimension
    imgs = imgs.to(device).to(dtype).unsqueeze(0);  // (1, N, 3, H, W)

    // Run inference
    std::cout << "[main] Running inference on " << paths.size()
              << " frames (window=" << args.window_size
              << ", overlap=" << args.overlap_size << ")...\n";

    loger::Pi3Impl::InferenceResult result;
    {
        torch::NoGradGuard no_grad;
        result = model->forward(imgs,
                                args.window_size,
                                args.overlap_size,
                                args.se3,
                                /*reset_every=*/0,
                                args.no_ttt,
                                args.no_swa,
                                args.save_windows);
    }

    std::cout << "[main] Inference complete. Writing outputs...\n";

    // Prepare flattened point cloud
    const int B = result.points.size(0);
    const int N = result.points.size(1);
    const int H = result.points.size(2);
    const int W = result.points.size(3);

    std::cout << "[main] Preparing point cloud tensors..." << std::flush;
    auto pts_flat  = result.points.squeeze(0)
                            .reshape({N * H * W, 3})
                            .to(torch::kFloat32).cpu();
    auto conf_flat = result.conf.squeeze(0).squeeze(-1)
                            .reshape({N * H * W})
                            .to(torch::kFloat32).cpu();

    // Build RGB colors from original images: (N, 3, H, W) → (N, H, W, 3)
    auto imgs_cpu = imgs.squeeze(0)
                        .to(torch::kFloat32).cpu()
                        .permute({0, 2, 3, 1})  // (N, H, W, 3)
                        .clamp(0.0f, 1.0f)
                        .mul(255.0f)
                        .to(torch::kUInt8)
                        .reshape({N * H * W, 3});
    std::cout << " done\n";

    loger::OutputWriter::write_ply(args.output_ply, pts_flat, imgs_cpu, conf_flat,
                                   args.conf_threshold);
    std::cout << "[main] Wrote point cloud to: " << args.output_ply << "\n";

    // Write trajectory (squeeze batch dim)
    auto poses_n = result.camera_poses.squeeze(0)
                         .to(torch::kFloat32).cpu();  // (N, 4, 4)
    loger::OutputWriter::write_trajectory(args.output_traj, poses_n);
    std::cout << "[main] Wrote trajectory to: " << args.output_traj << "\n";

    // Optionally write per-frame .pt for the frame-by-frame visualizer
    if (!args.output_pt.empty()) {
        // images: (N, H, W, 3) float32 [0,1]
        auto imgs_nhw3 = imgs.squeeze(0)
                             .to(torch::kFloat32).cpu()
                             .permute({0, 2, 3, 1})
                             .clamp(0.0f, 1.0f)
                             .contiguous();
        loger::OutputWriter::write_pt(
            args.output_pt,
            result.points.squeeze(0).to(torch::kFloat32).cpu().contiguous(),
            result.conf.squeeze(0).to(torch::kFloat32).cpu().contiguous(),
            imgs_nhw3,
            poses_n);
        std::cout << "[main] Wrote per-frame data to: " << args.output_pt << "\n";
    }

    return 0;
}
