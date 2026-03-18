#include <loger/model/pi3.hpp>
#include <loger/io/image_loader.hpp>
#include <iostream>
#include <cassert>

int main(int argc, char* argv[]) {
    std::cout << "=== test_e2e ===\n";
    if (argc < 3) {
        std::cout << "Usage: test_e2e <checkpoint.pt> <config.yaml>\n";
        std::cout << "Skipping (no args provided)\n";
        return 0;
    }

    auto cfg = loger::load_config(argv[2]);
    loger::Pi3 model(cfg);
    model->load_weights(argv[1]);
    model->eval();

    // Small synthetic input: 4 frames, 56x42 (multiples of 14)
    const int N=4, H=56, W=42;
    auto imgs = torch::rand({1, N, 3, H, W});

    torch::NoGradGuard ng;
    auto result = model->forward(imgs, /*window_size=*/4, /*overlap_size=*/1,
                                  /*se3=*/false, /*reset_every=*/0,
                                  /*no_ttt=*/false, /*no_swa=*/false);

    assert(result.points.size(1) == N);
    assert(result.points.size(2) == H);
    assert(result.points.size(3) == W);
    assert(result.camera_poses.size(1) == N);
    assert(!torch::any(torch::isnan(result.points)).item<bool>());
    std::cout << "  E2E output shapes: PASS\n";
    std::cout << "  E2E no NaN: PASS\n";
    std::cout << "=== ALL PASS ===\n";
    return 0;
}
