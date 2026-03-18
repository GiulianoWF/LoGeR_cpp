#include <loger/model/ttt.hpp>
#include <iostream>
#include <cassert>

int main() {
    std::cout << "=== test_ttt ===\n";

    // Test Newton-Schulz
    auto G = torch::randn({4, 512, 512});
    auto G_ortho = loger::zeropower_newtonschulz5(G, 5);
    // Check approximate orthogonality: G^T @ G ≈ I
    auto GtG = torch::bmm(G_ortho.transpose(1,2).to(torch::kFloat32),
                           G_ortho.to(torch::kFloat32));
    auto eye = torch::eye(512).unsqueeze(0).expand({4,-1,-1});
    auto diff = (GtG - eye).abs().mean().item<float>();
    std::cout << "  Newton-Schulz mean deviation from I: " << diff << "\n";
    assert(diff < 0.1f);
    std::cout << "  Newton-Schulz orthogonality: PASS\n";

    // Test FastWeightGluMLPMultihead construction
    loger::FastWeightGluMLPMultihead ttt(1024, 512, 4, 5, 1, false, false);
    std::cout << "  FastWeightGluMLPMultihead construction: PASS\n";

    // Test init_state
    auto state = ttt->init_state(1, torch::kCPU, torch::kFloat32);
    assert(state.defined());
    assert(state.w0.size(0) == 2);  // num_heads = 1024/512 = 2
    std::cout << "  init_state shape: PASS\n";

    // Test forward (random weights, just check shapes)
    auto x = torch::randn({1, 32, 1024});
    auto [out, new_state] = ttt->forward(x, state);
    assert(out.sizes() == x.sizes());
    assert(!torch::any(torch::isnan(out)).item<bool>());
    std::cout << "  TTT forward shape: PASS\n";
    std::cout << "  TTT no NaN: PASS\n";

    std::cout << "=== ALL PASS ===\n";
    return 0;
}
