#include <loger/io/weight_loader.hpp>
#include <loger/model/encoder.hpp>
#include <iostream>
#include <cassert>

int main(int argc, char* argv[]) {
    std::cout << "=== test_encoder ===\n";
    if (argc < 2) {
        std::cout << "Usage: test_encoder <path_to_checkpoint.pt>\n";
        std::cout << "Skipping weight loading test (no checkpoint provided)\n";

        // At least test that encoder constructs without error
        loger::DinoV2Encoder enc;
        std::cout << "  DinoV2Encoder construction: PASS\n";

        // Test forward shape with random weights
        auto dummy_img = torch::randn({2, 3, 14*4, 14*3});  // (2, 3, 56, 42)
        auto out = enc->forward(dummy_img);
        assert(out.size(0) == 2);
        assert(out.size(1) == 4 * 3);      // 12 patches
        assert(out.size(2) == loger::DinoV2EncoderImpl::EMBED_DIM);
        std::cout << "  DinoV2Encoder forward shape: PASS\n";
        std::cout << "=== ALL PASS ===\n";
        return 0;
    }

    // Load from checkpoint
    loger::TensorStore ts;
    ts.load(argv[1]);

    loger::DinoV2Encoder enc;
    enc->load_weights(ts, "encoder");
    enc->eval();

    auto dummy_img = torch::randn({1, 3, 504, 280});
    torch::NoGradGuard ng;
    auto out = enc->forward(dummy_img);
    // Expected: (1, 36*20, 1024) = (1, 720, 1024)
    assert(out.size(0) == 1);
    assert(out.size(1) == 720);
    assert(out.size(2) == 1024);
    std::cout << "  Encoder output shape (1,720,1024): PASS\n";
    std::cout << "=== ALL PASS ===\n";
    return 0;
}
