#include <loger/ops/rope2d.hpp>
#include <iostream>
#include <cassert>

int main() {
    std::cout << "=== test_rope2d ===\n";

    // Test PositionGetter
    loger::PositionGetter pg;
    auto pos = pg(2, 4, 6, torch::kCPU);  // (2, 24, 2)
    assert(pos.size(0) == 2);
    assert(pos.size(1) == 24);
    assert(pos.size(2) == 2);
    // Top-left patch should be (0, 0)
    assert(pos[0][0][0].item<int>() == 0);
    assert(pos[0][0][1].item<int>() == 0);
    // Second column: (0, 1)
    assert(pos[0][1][1].item<int>() == 1);
    std::cout << "  PositionGetter: PASS\n";

    // Test RoPE2D forward — just check output shape and no NaN
    loger::RoPE2D rope(100.0f);
    auto tokens = torch::randn({2, 8, 24, 64});  // (B, H, N, D)
    auto positions = pg(2, 4, 6, torch::kCPU).to(torch::kInt64);  // (2, 24, 2)
    // Shift positions by 1 (as done in model)
    positions = positions + 1;
    auto out = rope->forward(tokens, positions);
    assert(out.sizes() == tokens.sizes());
    assert(!torch::any(torch::isnan(out)).item<bool>());
    std::cout << "  RoPE2D forward shape: PASS\n";
    std::cout << "  RoPE2D no NaN: PASS\n";

    std::cout << "=== ALL PASS ===\n";
    return 0;
}
