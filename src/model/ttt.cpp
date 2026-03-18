#include <loger/model/ttt.hpp>
#include <loger/io/weight_loader.hpp>
#include <cmath>

namespace loger {

// ---------------------------------------------------------------------------
// Newton-Schulz orthogonalization
// ---------------------------------------------------------------------------

torch::Tensor zeropower_newtonschulz5(torch::Tensor G, int steps) {
    // G: (B, D_in, D_out) — works with rectangular matrices too.
    bool transposed = G.size(1) > G.size(2);
    if (transposed) G = G.transpose(1, 2).contiguous();

    // Cast to bfloat16 for numerical stability (mirrors Python)
    G = G.to(torch::kBFloat16);

    // Normalize by Frobenius norm
    auto norm = G.norm(/*p=*/2, {1, 2}, /*keepdim=*/true) + 1e-7f;
    G = G / norm;

    // Quintic Newton-Schulz iterations
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    for (int i = 0; i < steps; ++i) {
        auto A = torch::bmm(G, G.transpose(1, 2));  // (B, D, D)
        auto B_ = b * A + c * torch::bmm(A, A);
        G = a * G + torch::bmm(B_, G);
    }

    if (transposed) G = G.transpose(1, 2).contiguous();
    return G;
}

// ---------------------------------------------------------------------------
// SiLU backward
// ---------------------------------------------------------------------------

torch::Tensor silu_backprop(torch::Tensor dy, torch::Tensor x_pre_act) {
    // SiLU: f(x) = x * sigma(x)
    // f'(x) = sigma(x) * (1 + x * (1 - sigma(x)))
    auto sigma = torch::sigmoid(x_pre_act);
    return dy * sigma * (1.0f + x_pre_act * (1.0f - sigma));
}

// ---------------------------------------------------------------------------
// FastWeightGluMLPMultihead
// ---------------------------------------------------------------------------

FastWeightGluMLPMultiheadImpl::FastWeightGluMLPMultiheadImpl(
    int dim, int head_dim, int inter_multi, int muon_update_steps,
    int ttt_update_steps, bool use_momentum, bool pre_norm)
    : dim_(dim),
      num_heads_(dim / head_dim),
      head_dim_(head_dim),
      inter_dim_(head_dim * inter_multi),
      muon_steps_(muon_update_steps),
      ttt_update_steps_(ttt_update_steps),
      use_momentum_(use_momentum),
      pre_norm_(pre_norm),
      base_lr_inv_(std::log(std::exp(0.1f) - 1.0f))  // softplus_inv(0.1)
{
    // Base fast weights: per-head, shape (num_heads, head_dim, inter_dim) etc.
    w0 = register_parameter("w0",
        torch::zeros({num_heads_, head_dim_, inter_dim_}));
    w1 = register_parameter("w1",
        torch::zeros({num_heads_, inter_dim_, head_dim_}));
    w2 = register_parameter("w2",
        torch::zeros({num_heads_, head_dim_, inter_dim_}));

    // QKV projection: projects dim → 3*dim, then reshape per head
    to_qkv_ = register_module("to_qkv",
        torch::nn::Linear(torch::nn::LinearOptions(dim, 3 * dim).bias(false)));

    // Output projection
    c_proj_ = register_module("c_proj",
        torch::nn::Linear(dim, dim));

    // Per-head learning rate: outputs (num_heads * 3) scalars — has bias
    lr_fc_ = register_module("lr_fc",
        torch::nn::Linear(dim, num_heads_ * 3));

    // Output normalization
    o_norm_ = register_module("o_norm", RMSNorm(head_dim_));

    if (pre_norm_) {
        pre_norm_ln_ = register_module("pre_norm", RMSNorm(dim));
    }
}

void FastWeightGluMLPMultiheadImpl::load_weights(const TensorStore& ts,
                                                  const std::string& prefix) {
    TensorStore::copy_weight(w0, ts, prefix + ".w0");
    TensorStore::copy_weight(w1, ts, prefix + ".w1");
    TensorStore::copy_weight(w2, ts, prefix + ".w2");

    TensorStore::copy_weight(to_qkv_->weight, ts, prefix + ".to_qkv.weight");
    TensorStore::copy_weight(c_proj_->weight, ts, prefix + ".c_proj.weight");
    TensorStore::copy_weight_optional(c_proj_->bias, ts, prefix + ".c_proj.bias");
    TensorStore::copy_weight(lr_fc_->weight, ts, prefix + ".lr_fc.weight");
    TensorStore::copy_weight_optional(lr_fc_->bias, ts, prefix + ".lr_fc.bias");
    TensorStore::copy_weight(o_norm_->weight, ts, prefix + ".o_norm.weight");

    if (pre_norm_) {
        // pre_norm is RMSNorm — weight only, no bias
        TensorStore::copy_weight(pre_norm_ln_->weight, ts,
            prefix + ".pre_norm.weight");
    }
}

TTTState FastWeightGluMLPMultiheadImpl::init_state(int B, torch::Device device,
                                                    torch::Dtype dtype) const {
    TTTState s;
    // Expand base weights to batch: (num_heads, D, D) → (B*num_heads, D, D)
    s.w0 = w0.to(device).to(dtype).unsqueeze(0)
              .expand({B, -1, -1, -1})
              .reshape({B * num_heads_, head_dim_, inter_dim_})
              .contiguous();
    s.w1 = w1.to(device).to(dtype).unsqueeze(0)
              .expand({B, -1, -1, -1})
              .reshape({B * num_heads_, inter_dim_, head_dim_})
              .contiguous();
    s.w2 = w2.to(device).to(dtype).unsqueeze(0)
              .expand({B, -1, -1, -1})
              .reshape({B * num_heads_, head_dim_, inter_dim_})
              .contiguous();

    // Store initial weight norms for re-normalization after updates
    s.w0_norm = s.w0.norm(2, {1, 2}, /*keepdim=*/true);
    s.w1_norm = s.w1.norm(2, {1, 2}, /*keepdim=*/true);
    s.w2_norm = s.w2.norm(2, {1, 2}, /*keepdim=*/true);

    return s;
}

// Core apply+update logic.
// Matches Python fast_weight_swish_glu_weight_norm_mini_batch_apply with
// ttt_op_order = [TTTOperator(apply=True, update=False),
//                 TTTOperator(apply=False, update=True)]
//
// q/k/v:  (B*H, L, D)    — projected queries/keys/values
// lr:     (B, L, H*3)    — per-position learning rates (already after softplus)
// state:  fast weights
std::pair<torch::Tensor, TTTState>
FastWeightGluMLPMultiheadImpl::apply_and_update(torch::Tensor q,
                                                 torch::Tensor k,
                                                 torch::Tensor v,
                                                 torch::Tensor lr,
                                                 TTTState state) {
    torch::NoGradGuard ng;
    const int BH = q.size(0);
    const int L  = q.size(1);
    const int B  = BH / num_heads_;

    auto& w0_cur = state.w0;  // (B*H, D, Di)
    auto& w1_cur = state.w1;  // (B*H, Di, D)
    auto& w2_cur = state.w2;  // (B*H, D, Di)

    // Learning rates: (B, L, H*3) → rearrange to (BH, L, 1) for each lr.
    // Python: "b l (lrs h d) -> lrs (b h) l d" with lrs=3, d=1.
    // Layout: position i has values [lr0_h0, lr0_h1, ..., lr1_h0, ..., lr2_hN]
    // i.e. [lrs, h] major = slice by 3 first, then heads.
    auto lr_bhld = lr.reshape({B, L, 3, num_heads_})   // [B, L, lrs=3, H]
                      .permute({0, 3, 1, 2})            // [B, H, L, 3]
                      .reshape({BH, L, 3});

    using SL = torch::indexing::Slice;
    auto lr0 = lr_bhld.index({SL(), SL(), SL(0, 1)});  // (BH, L, 1)
    auto lr1 = lr_bhld.index({SL(), SL(), SL(1, 2)});
    auto lr2 = lr_bhld.index({SL(), SL(), SL(2, 3)});

    // Precompute initial weight norms for re-normalization after updates.
    // Python: w_norm = w.detach().norm(dim=1, keepdim=True)  (norm along dim1)
    auto w0_norm = w0_cur.detach().norm(2, {1}, /*keepdim=*/true);
    auto w1_norm = w1_cur.detach().norm(2, {1}, /*keepdim=*/true);
    auto w2_norm = w2_cur.detach().norm(2, {1}, /*keepdim=*/true);

    // --- PASS 1 (apply=True, update=False): compute output using CURRENT weights ---
    // output = (silu(q @ w0) * (q @ w2)) @ w1
    auto gate_pre   = torch::bmm(q, w0_cur);           // (BH, L, Di)
    auto hidden_pre = torch::bmm(q, w2_cur);           // (BH, L, Di)
    auto output     = torch::bmm(torch::silu(gate_pre) * hidden_pre, w1_cur);  // (BH, L, D)
    // o_norm applied after the loop in Python: apply it here on output directly
    output = o_norm_->forward(output);

    // --- PASS 2 (apply=False, update=True): update weights using k and v ---
    // Matches Python's update step which computes gate/hidden from k, not q.
    for (int step = 0; step < ttt_update_steps_; ++step) {
        // Gate and hidden from k (matching Python which uses ki, not qi, for update)
        auto k_gate_pre   = torch::bmm(k, w0_cur);                        // (BH, L, Di)
        auto k_hidden_pre = torch::bmm(k, w2_cur);                        // (BH, L, Di)
        auto k_gate       = torch::silu(k_gate_pre);
        auto k_hidden     = k_gate * k_hidden_pre;                        // (BH, L, Di)

        // Gradients: objective = tr(v^T @ (silu(k@w0) * (k@w2)) @ w1)
        // Python: dhidden = vi @ w1.T   (uses v directly, not pred_err)
        auto dhidden   = torch::bmm(v.to(w1_cur.dtype()),
                                    w1_cur.transpose(1, 2));               // (BH, L, Di)
        // dhidden_before_mul = dhidden * silu(k@w0)
        auto dhidden_bm = dhidden * k_gate;                                // (BH, L, Di)
        // dgate = dhidden * (k@w2)
        auto dgate     = dhidden * k_hidden_pre;                          // (BH, L, Di)
        // silu backprop on gate
        auto dgate_ba  = silu_backprop(dgate, k_gate_pre);               // (BH, L, Di)

        // Gradients for each weight (cast lr to weight dtype)
        // w1_grad = (k_hidden * lr1).T @ v
        auto dw1 = torch::bmm((k_hidden * lr1.to(k_hidden.dtype())).transpose(1, 2),
                               v.to(w1_cur.dtype()));                     // (BH, Di, D)
        // w0_grad = (k * lr0).T @ dgate_ba
        auto dw0 = torch::bmm((k * lr0.to(k.dtype())).transpose(1, 2),
                               dgate_ba.to(k.dtype()));                   // (BH, D, Di)
        // w2_grad = (k * lr2).T @ dhidden_bm
        auto dw2 = torch::bmm((k * lr2.to(k.dtype())).transpose(1, 2),
                               dhidden_bm.to(k.dtype()));                 // (BH, D, Di)

        // Apply Newton-Schulz (MUON) orthogonalization
        if (muon_steps_ > 0) {
            dw0 = zeropower_newtonschulz5(dw0.to(torch::kFloat32), muon_steps_);
            dw1 = zeropower_newtonschulz5(dw1.to(torch::kFloat32), muon_steps_);
            dw2 = zeropower_newtonschulz5(dw2.to(torch::kFloat32), muon_steps_);
        }

        // Gradient ascent: add gradients to weights
        w0_cur = w0_cur + dw0.to(w0_cur.dtype());
        w1_cur = w1_cur + dw1.to(w1_cur.dtype());
        w2_cur = w2_cur + dw2.to(w2_cur.dtype());

        // Re-normalize: w = w / norm(w) * initial_norm  (dim=1 matches Python)
        auto cur_norm0 = w0_cur.norm(2, {1}, /*keepdim=*/true).clamp_min(1e-5f);
        w0_cur = w0_cur / cur_norm0 * w0_norm.to(w0_cur.dtype());
        auto cur_norm1 = w1_cur.norm(2, {1}, /*keepdim=*/true).clamp_min(1e-5f);
        w1_cur = w1_cur / cur_norm1 * w1_norm.to(w1_cur.dtype());
        auto cur_norm2 = w2_cur.norm(2, {1}, /*keepdim=*/true).clamp_min(1e-5f);
        w2_cur = w2_cur / cur_norm2 * w2_norm.to(w2_cur.dtype());
    }

    state.w0 = w0_cur;
    state.w1 = w1_cur;
    state.w2 = w2_cur;

    return {output, state};
}

std::pair<torch::Tensor, TTTState>
FastWeightGluMLPMultiheadImpl::forward(torch::Tensor x, TTTState state) {
    // x: (B, L, dim)
    const int B = x.size(0);
    const int L = x.size(1);

    if (pre_norm_) x = pre_norm_ln_->forward(x);

    // QKV projection with silu activation (mirrors Python: F.silu(self.to_qkv(x)))
    auto qkv = torch::silu(to_qkv_->forward(x));  // (B, L, 3*dim)

    // Per-head learning rates — run lr_fc in float32 (Python: autocast disabled),
    // then apply softplus to ensure positive LRs. Matches Python:
    //   lr = softplus(lr_fc(x.float()) + base_lr_inv)
    auto lr = torch::softplus(
        lr_fc_->forward(x).to(torch::kFloat32) + base_lr_inv_);  // (B, L, H*3)

    // Split QKV: (B, L, 3*dim) → rearrange "b l (qkv h d) -> qkv (b h) l d"
    auto q = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(0, dim_)})
               .reshape({B, L, num_heads_, head_dim_})
               .permute({0, 2, 1, 3})
               .reshape({B * num_heads_, L, head_dim_});

    auto k = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(dim_, 2 * dim_)})
               .reshape({B, L, num_heads_, head_dim_})
               .permute({0, 2, 1, 3})
               .reshape({B * num_heads_, L, head_dim_});

    auto v = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(2 * dim_)})
               .reshape({B, L, num_heads_, head_dim_})
               .permute({0, 2, 1, 3})
               .reshape({B * num_heads_, L, head_dim_});

    // L2-normalize q and k (mirrors Python)
    q = q / (q.norm(2, {-1}, /*keepdim=*/true) + 1e-5f);
    k = k / (k.norm(2, {-1}, /*keepdim=*/true) + 1e-5f);

    // Initialize state from base weights if not provided
    if (!state.defined()) {
        state = init_state(B, x.device(), x.scalar_type());
    }

    // Apply TTT: forward pass + weight update
    auto [ttt_out, new_state] = apply_and_update(q, k, v, lr, state);

    // Reshape output: (B*H, L, D) → (B, L, dim)
    auto out = ttt_out.reshape({B, num_heads_, L, head_dim_})
                      .permute({0, 2, 1, 3})
                      .reshape({B, L, dim_});

    // Output projection + residual
    out = c_proj_->forward(out);

    return {out, new_state};
}

} // namespace loger
