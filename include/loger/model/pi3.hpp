#pragma once
#include <loger/model/encoder.hpp>
#include <loger/model/block_rope.hpp>
#include <loger/model/ttt.hpp>
#include <loger/model/task_heads.hpp>
#include <loger/ops/rope2d.hpp>
#include <loger/utils/windowing.hpp>
#include <loger/io/weight_loader.hpp>
#include <torch/torch.h>
#include <vector>
#include <string>

namespace loger {

/// Configuration loaded from original_config.yaml.
struct Pi3Config {
    std::vector<int> ttt_insert_after = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34};
    std::vector<int> attn_insert_after = {10,18,26,34};
    int ttt_head_dim    = 512;
    int ttt_inter_multi = 4;
    int muon_steps      = 5;
    int ttt_update_steps = 1;
    bool use_momentum   = false;
    bool ttt_pre_norm   = false;
    bool se3_align      = false;
};

Pi3Config load_config(const std::string& yaml_path);

/// Per-window state carrying TTT fast weights and SWA KV caches.
struct WindowState {
    // TTT: one TTTState per ttt_insert_after position
    std::vector<TTTState> ttt;
    // SWA: (K, V) cache per attn_insert_after position
    std::vector<std::pair<torch::Tensor, torch::Tensor>> swa_kv;

    bool initialized = false;
    void reset_ttt(const std::vector<FastWeightGluMLPMultihead>& ttt_layers,
                   int B, torch::Device dev, torch::Dtype dtype);
    void clear();
};

/// Main Pi3 model: DINOv2 encoder + 36-layer BlockRope decoder + TTT + task heads.
class Pi3Impl : public torch::nn::Module {
public:
    static constexpr int DEC_DIM      = 1024;
    static constexpr int DEC_HEADS    = 16;
    static constexpr int DEC_DEPTH    = 36;
    static constexpr int PATCH_START  = 6;   // 5 register + 1 PE token

    explicit Pi3Impl(Pi3Config cfg = {});

    /// Load all weights from a .pt state dict file.
    void load_weights(const std::string& ckpt_path);

    struct InferenceResult {
        torch::Tensor points;        // (B, N, H, W, 3) global
        torch::Tensor local_points;  // (B, N, H, W, 3) camera-relative
        torch::Tensor conf;          // (B, N, H, W, 1) after sigmoid + edge mask
        torch::Tensor camera_poses;  // (B, N, 4, 4)
    };

    /// imgs: (B, N, 3, H, W) float32 in [0, 1]
    /// If save_windows_dir is non-empty, saves each raw window prediction as
    /// .npy files into that directory (for offline merge experimentation) and
    /// skips the SE3 merge — the returned result is a simple concatenation.
    InferenceResult forward(torch::Tensor imgs,
                            int window_size   = 32,
                            int overlap_size  = 3,
                            bool se3_align    = true,
                            int reset_every   = 0,
                            bool turn_off_ttt = false,
                            bool turn_off_swa = false,
                            const std::string& save_windows_dir = "");

private:
    Pi3Config cfg_;

    DinoV2Encoder encoder_{nullptr};
    torch::nn::ModuleList decoder_;           // DEC_DEPTH × BlockRope
    torch::nn::ModuleList ttt_layers_;        // cfg_.ttt_insert_after.size() × FastWeightGluMLPMultihead
    torch::nn::ModuleList ttt_gate_projs_;    // same size × Linear(DEC_DIM, 1)
    torch::nn::ModuleList swa_layers_;        // cfg_.attn_insert_after.size() × BlockRope
    torch::nn::ModuleList swa_gate_projs_;    // same size × Linear(DEC_DIM, 1)

    torch::Tensor register_token_;   // (1, 1, 5, DEC_DIM)
    torch::Tensor pe_token_0_, pe_token_1_, pe_token_2_;

    TransformerDecoder point_decoder_{nullptr};
    TransformerDecoder conf_decoder_{nullptr};
    TransformerDecoder camera_decoder_{nullptr};
    LinearPts3d  point_head_{nullptr};
    LinearPts3d  conf_head_{nullptr};
    CameraHead   camera_head_{nullptr};

    RoPE2D rope_{nullptr};
    PositionGetter pos_getter_;

    torch::Tensor image_mean_, image_std_;  // (3,) ImageNet stats

    // --- Internal helpers ---
    struct DecodeOutput {
        // Full sequence including PATCH_START special tokens.
        // hidden: (B*N, PATCH_START+hw, 2*DEC_DIM) — concat of last 2 layer outputs
        // pos:    (B*N, PATCH_START+hw, 2)          — zero for special tokens, +1-shifted for patches
        torch::Tensor hidden;
        torch::Tensor pos;
    };

    DecodeOutput decode(torch::Tensor hidden_input,
                        int N, int H, int W,
                        int window_size, int overlap_size,
                        WindowState& state,
                        bool turn_off_ttt, bool turn_off_swa);

    WindowPrediction run_window(torch::Tensor imgs_w,
                                int window_size, int overlap_size,
                                WindowState& state,
                                bool turn_off_ttt, bool turn_off_swa);

    // Map from layer index to TTT/SWA list index
    std::unordered_map<int,int> ttt_layer_idx_, swa_layer_idx_;
};
TORCH_MODULE(Pi3);

} // namespace loger
