#include <loger/model/pi3.hpp>
#include <loger/utils/geometry.hpp>
#include <loger/io/output_writer.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <chrono>

namespace {

// Save a single tensor to a file in pickle format (loadable with torch.load in Python).
// Creates the parent directory if needed.
void save_debug_tensor(const torch::Tensor& t, const std::string& path) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path().string().empty()
            ? "." : std::filesystem::path(path).parent_path());
    auto vec = torch::pickle_save(t.detach().to(torch::kCPU).to(torch::kFloat32).contiguous());
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(vec.data(), static_cast<std::streamsize>(vec.size()));
    std::cout << "[debug] saved " << path
              << " shape=" << t.sizes() << "\n";
}

// CUDA-aware wall-clock helper: syncs GPU before taking timestamp.
struct CudaTimer {
    using Clock = std::chrono::high_resolution_clock;
    torch::Device dev;
    Clock::time_point t0;

    explicit CudaTimer(torch::Device d) : dev(d) {
        if (dev.is_cuda()) torch::cuda::synchronize();
        t0 = Clock::now();
    }
    // Returns elapsed milliseconds since construction (with CUDA sync).
    double elapsed_ms() const {
        if (dev.is_cuda()) torch::cuda::synchronize();
        auto t1 = Clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

} // anonymous namespace

namespace loger {

// ---------------------------------------------------------------------------
// Config loading (minimal YAML-ish parser for the simple config format)
// ---------------------------------------------------------------------------

Pi3Config load_config(const std::string& yaml_path) {
    Pi3Config cfg;
    std::ifstream f(yaml_path);
    if (!f.is_open()) {
        std::cerr << "[Pi3Config] Could not open " << yaml_path
                  << " — using defaults\n";
        return cfg;
    }

    auto parse_int_list = [](const std::string& s) {
        std::vector<int> result;
        std::string inner = s;
        // Remove [ ]
        inner.erase(std::remove(inner.begin(), inner.end(), '['), inner.end());
        inner.erase(std::remove(inner.begin(), inner.end(), ']'), inner.end());
        std::istringstream ss(inner);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
            if (!tok.empty()) result.push_back(std::stoi(tok));
        }
        return result;
    };

    std::string line;
    while (std::getline(f, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        // Trim whitespace
        auto trim = [](std::string& s) {
            size_t start = s.find_first_not_of(" \t");
            size_t end   = s.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) s = "";
            else s = s.substr(start, end - start + 1);
        };
        trim(key); trim(val);

        if (key == "ttt_insert_after" && !val.empty())
            cfg.ttt_insert_after = parse_int_list(val);
        else if (key == "attn_insert_after" && !val.empty())
            cfg.attn_insert_after = parse_int_list(val);
        else if (key == "ttt_head_dim")    cfg.ttt_head_dim    = std::stoi(val);
        else if (key == "ttt_inter_multi") cfg.ttt_inter_multi = std::stoi(val);
        else if (key == "ttt_pre_norm")    cfg.ttt_pre_norm    = (val == "true");
        else if (key == "se3")             cfg.se3_align       = (val == "true");
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// WindowState
// ---------------------------------------------------------------------------

void WindowState::reset_ttt(
    const std::vector<FastWeightGluMLPMultihead>& layers,
    int B, torch::Device dev, torch::Dtype dtype) {
    ttt.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
        ttt[i] = layers[i]->init_state(B, dev, dtype);
}

void WindowState::clear() {
    ttt.clear();
    swa_kv.clear();
    initialized = false;
}

// ---------------------------------------------------------------------------
// Pi3
// ---------------------------------------------------------------------------

Pi3Impl::Pi3Impl(Pi3Config cfg) : cfg_(cfg) {

    // Build index maps
    for (size_t i = 0; i < cfg_.ttt_insert_after.size(); ++i)
        ttt_layer_idx_[cfg_.ttt_insert_after[i]] = static_cast<int>(i);
    for (size_t i = 0; i < cfg_.attn_insert_after.size(); ++i)
        swa_layer_idx_[cfg_.attn_insert_after[i]] = static_cast<int>(i);

    // Encoder
    encoder_ = register_module("encoder", DinoV2Encoder());

    // Shared RoPE (freq=100)
    rope_ = register_module("rope", RoPE2D(100.0f));

    // Register ModuleLists so model->to() moves their parameters
    register_module("decoder",        decoder_);
    register_module("ttt_layers",     ttt_layers_);
    register_module("ttt_gate_projs", ttt_gate_projs_);
    register_module("swa_layers",     swa_layers_);
    register_module("swa_gate_projs", swa_gate_projs_);

    // Decoder blocks — Python uses standard Mlp(GELU), not SwiGLU
    for (int i = 0; i < DEC_DEPTH; ++i)
        decoder_->push_back(BlockRope(DEC_DIM, DEC_HEADS,
                                       /*mlp_ratio=*/4.0f,
                                       /*qkv_bias=*/true,
                                       /*qk_norm=*/true,
                                       /*init_values=*/0.01f,
                                       /*use_swiglu=*/false,
                                       rope_));

    // TTT layers
    for (int idx : cfg_.ttt_insert_after) {
        (void)idx;
        ttt_layers_->push_back(
            FastWeightGluMLPMultihead(DEC_DIM, cfg_.ttt_head_dim,
                                       cfg_.ttt_inter_multi,
                                       cfg_.muon_steps,
                                       cfg_.ttt_update_steps,
                                       cfg_.use_momentum,
                                       cfg_.ttt_pre_norm));
        ttt_gate_projs_->push_back(
            torch::nn::Linear(torch::nn::LinearOptions(DEC_DIM, 1).bias(true)));
    }

    // SWA layers
    for (int idx : cfg_.attn_insert_after) {
        (void)idx;
        swa_layers_->push_back(BlockRope(DEC_DIM, DEC_HEADS,
                                          /*mlp_ratio=*/4.0f,
                                          /*qkv_bias=*/true,
                                          /*qk_norm=*/true,
                                          /*init_values=*/0.01f,
                                          /*use_swiglu=*/false,
                                          rope_));
        swa_gate_projs_->push_back(
            torch::nn::Linear(torch::nn::LinearOptions(DEC_DIM, 1).bias(true)));
    }

    // Special tokens
    register_token_ = register_parameter("register_token",
        torch::zeros({1, 1, 5, DEC_DIM}));
    pe_token_0_ = register_parameter("pe_token_0", torch::zeros({1, 1, 1, DEC_DIM}));
    pe_token_1_ = register_parameter("pe_token_1", torch::zeros({1, 1, 1, DEC_DIM}));
    pe_token_2_ = register_parameter("pe_token_2", torch::zeros({1, 1, 1, DEC_DIM}));

    // Task decoders/heads (in=2*DEC_DIM because we concat last 2 decoder outputs)
    point_decoder_  = register_module("point_decoder",
        TransformerDecoder(2 * DEC_DIM, DEC_DIM, DEC_DIM, 5, DEC_HEADS, rope_));
    point_decoder_->debug_tag_ = "point_decoder";
    conf_decoder_   = register_module("conf_decoder",
        TransformerDecoder(2 * DEC_DIM, DEC_DIM, DEC_DIM, 5, DEC_HEADS, rope_));
    conf_decoder_->debug_tag_ = "conf_decoder";
    camera_decoder_ = register_module("camera_decoder",
        TransformerDecoder(2 * DEC_DIM, 512, DEC_DIM, 5, DEC_HEADS, rope_));

    point_head_  = register_module("point_head",
        LinearPts3d(DinoV2EncoderImpl::PATCH_SIZE, DEC_DIM, 3));
    conf_head_   = register_module("conf_head",
        LinearPts3d(DinoV2EncoderImpl::PATCH_SIZE, DEC_DIM, 1));
    camera_head_ = register_module("camera_head", CameraHead(512));

    // ImageNet normalization buffers
    image_mean_ = register_buffer("image_mean",
        torch::tensor({0.485f, 0.456f, 0.406f}).reshape({1,3,1,1}));
    image_std_  = register_buffer("image_std",
        torch::tensor({0.229f, 0.224f, 0.225f}).reshape({1,3,1,1}));
}

void Pi3Impl::load_weights(const std::string& ckpt_path) {
    TensorStore ts;
    ts.load(ckpt_path);

    encoder_->load_weights(ts, "encoder");

    for (int i = 0; i < DEC_DEPTH; ++i)
        decoder_->at<BlockRopeImpl>(i).load_weights(
            ts, "decoder." + std::to_string(i));

    const int n_ttt = static_cast<int>(ttt_layers_->size());
    for (int i = 0; i < n_ttt; ++i) {
        ttt_layers_->at<FastWeightGluMLPMultiheadImpl>(i).load_weights(
            ts, "ttt_layers." + std::to_string(i));
        TensorStore::copy_weight(
            ttt_gate_projs_->at<torch::nn::LinearImpl>(i).weight,
            ts, "ttt_gate_projs." + std::to_string(i) + ".weight");
        TensorStore::copy_weight_optional(
            ttt_gate_projs_->at<torch::nn::LinearImpl>(i).bias,
            ts, "ttt_gate_projs." + std::to_string(i) + ".bias");
    }

    const int n_swa = static_cast<int>(swa_layers_->size());
    for (int i = 0; i < n_swa; ++i) {
        swa_layers_->at<BlockRopeImpl>(i).load_weights(
            ts, "swa_layers." + std::to_string(i));
        TensorStore::copy_weight(
            swa_gate_projs_->at<torch::nn::LinearImpl>(i).weight,
            ts, "swa_gate_projs." + std::to_string(i) + ".weight");
        TensorStore::copy_weight_optional(
            swa_gate_projs_->at<torch::nn::LinearImpl>(i).bias,
            ts, "swa_gate_projs." + std::to_string(i) + ".bias");
    }

    TensorStore::copy_weight(register_token_, ts, "register_token");
    TensorStore::copy_weight(pe_token_0_,     ts, "pe_token_0");
    TensorStore::copy_weight(pe_token_1_,     ts, "pe_token_1");
    TensorStore::copy_weight(pe_token_2_,     ts, "pe_token_2");

    point_decoder_->load_weights(ts,  "point_decoder");
    conf_decoder_->load_weights(ts,   "conf_decoder");
    camera_decoder_->load_weights(ts, "camera_decoder");
    point_head_->load_weights(ts,     "point_head");
    conf_head_->load_weights(ts,      "conf_head");
    camera_head_->load_weights(ts,    "camera_head");

    TensorStore::copy_weight_optional(image_mean_, ts, "image_mean");
    TensorStore::copy_weight_optional(image_std_,  ts, "image_std");

    std::cout << "[Pi3] Weights loaded from " << ckpt_path << "\n";

    // Debug: dump key weights for comparison
    if (std::getenv("LOGER_DEBUG_WEIGHTS")) {
        for (const auto& item : conf_decoder_->named_parameters()) {
            const auto& n = item.key();
            if (n.find("projects.weight") != std::string::npos ||
                n.find("blocks.0.norm1.weight") != std::string::npos ||
                n.find("blocks.0.attn.qkv.weight") != std::string::npos ||
                n.find("linear_out.weight") != std::string::npos) {
                save_debug_tensor(item.value(), "debug_cpp/conf_decoder." + n + ".pt");
            }
        }
        for (const auto& item : point_decoder_->named_parameters()) {
            if (item.key().find("projects.weight") != std::string::npos) {
                save_debug_tensor(item.value(), "debug_cpp/point_decoder." + item.key() + ".pt");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

Pi3Impl::DecodeOutput Pi3Impl::decode(torch::Tensor hidden_input,
                                       int N, int H, int W,
                                       int window_size, int overlap_size,
                                       WindowState& state,
                                       bool turn_off_ttt,
                                       bool turn_off_swa) {
    const int BN  = hidden_input.size(0);
    const int B   = BN / N;
    const int hw  = hidden_input.size(1);
    const int ph  = H / DinoV2EncoderImpl::PATCH_SIZE;
    const int pw  = W / DinoV2EncoderImpl::PATCH_SIZE;

    // Build position grid: (B*N, hw, 2), then shift +1 (special tokens take 0)
    auto pos_patches = pos_getter_(BN, ph, pw, hidden_input.device())
                           .to(torch::kInt64);  // (BN, hw, 2)
    pos_patches = pos_patches + 1;  // shift so patch positions start at 1

    // Prepend zero positions for special tokens: (BN, PATCH_START+hw, 2)
    auto pos_special = torch::zeros({BN, PATCH_START, 2},
                                     pos_patches.options());
    auto pos_full = torch::cat({pos_special, pos_patches}, 1);  // (BN, PATCH_START+hw, 2)

    // --- Prepend special tokens ---
    // register_token_: (1,1,5,DEC_DIM) → (B, N, 5, DEC_DIM) → (BN, 5, DEC_DIM)
    auto reg_tok = register_token_.expand({B, N, -1, -1})
                       .reshape({BN, 5, DEC_DIM});

    // Assign PE tokens per frame based on window overlap position (matches Python).
    // pe_token_0: frames that overlap with the PREVIOUS window
    // pe_token_1: non-overlap "middle" frames
    // pe_token_2: frames that overlap with the NEXT window
    const int eff_ws = (window_size > 0) ? window_size : N;
    const int num_overlap_prev = std::min(overlap_size, N);
    const int num_other        = std::min(std::max(eff_ws - 2 * overlap_size, 0),
                                          N - num_overlap_prev);
    const int num_overlap_next = std::max(
        std::min({overlap_size, N, N - num_overlap_prev - num_other}), 0);

    std::vector<torch::Tensor> pe_parts;
    if (num_overlap_prev > 0)
        pe_parts.push_back(pe_token_0_.expand({B, num_overlap_prev, -1, -1}));
    if (num_other > 0)
        pe_parts.push_back(pe_token_1_.expand({B, num_other, -1, -1}));
    if (num_overlap_next > 0)
        pe_parts.push_back(pe_token_2_.expand({B, num_overlap_next, -1, -1}));
    auto pe_tok_full = torch::cat(pe_parts, /*dim=*/1)  // (B, N, 1, DEC_DIM)
                           .reshape({BN, 1, DEC_DIM});

    // Full sequence: 5 reg + 1 pe + hw patches = PATCH_START + hw
    auto hidden = torch::cat({reg_tok, pe_tok_full, hidden_input}, /*dim=*/1);

    // --- Decode ---
    torch::Tensor prev_hidden, second_prev_hidden;

    static bool comp_debug_saved = false;
    const bool do_comp_debug = !comp_debug_saved && (std::getenv("LOGER_DEBUG_COMPONENTS") != nullptr);

    for (int i = 0; i < DEC_DEPTH; ++i) {
        torch::Tensor blk_input;
        torch::Tensor pos_for_blk;

        if (i % 2 == 0) {
            // Frame-level attention: treat each frame independently
            blk_input   = hidden.reshape({BN, PATCH_START + hw, DEC_DIM});
            pos_for_blk = pos_full;  // (BN, PATCH_START+hw, 2)
        } else {
            // Global attention: all frames together
            blk_input   = hidden.reshape({B, N * (PATCH_START + hw), DEC_DIM});
            pos_for_blk = pos_full.reshape({B, N * (PATCH_START + hw), 2});
        }

        blk_input = decoder_->at<BlockRopeImpl>(i).forward(blk_input, pos_for_blk);

        // Reshape back to (BN, PATCH_START+hw, DEC_DIM) for next iteration
        hidden = blk_input.reshape({BN, PATCH_START + hw, DEC_DIM});

        // Component-level debug saves (set LOGER_DEBUG_COMPONENTS=1)
        if (do_comp_debug) {
            save_debug_tensor(hidden,
                "debug_cpp/layer_" + std::to_string(i) + "_post_block.pt");
        }

        // --- TTT ---
        auto ttt_it = ttt_layer_idx_.find(i);
        if (ttt_it != ttt_layer_idx_.end() && !turn_off_ttt) {
            const int ti = ttt_it->second;
            auto& ttt_layer = ttt_layers_->at<FastWeightGluMLPMultiheadImpl>(ti);
            auto& gate_proj = ttt_gate_projs_->at<torch::nn::LinearImpl>(ti);

            // Initialize TTT state if needed
            if (!state.initialized || static_cast<int>(state.ttt.size()) <= ti
                || !state.ttt[ti].defined()) {
                if (static_cast<int>(state.ttt.size()) <= ti)
                    state.ttt.resize(ti + 1);
                state.ttt[ti] = ttt_layer.init_state(B, hidden.device(),
                                                       hidden.scalar_type());
            }

            // Reshape for TTT: (B, N*(PATCH_START+hw), DEC_DIM)
            auto h_for_ttt = hidden.reshape({B, N * (PATCH_START + hw), DEC_DIM});
            auto [ttt_out, new_state] = ttt_layer.forward(h_for_ttt, state.ttt[ti]);
            state.ttt[ti] = new_state;

            // Gating + residual: gate is computed from the INPUT (h_for_ttt), not the output.
            // Matches Python: gate_scale = silu(gate_proj(tokens_in))
            auto gate = torch::silu(gate_proj.forward(h_for_ttt));
            h_for_ttt = h_for_ttt + ttt_out * gate;
            hidden = h_for_ttt.reshape({BN, PATCH_START + hw, DEC_DIM});

            // Component debug: post-TTT
            if (do_comp_debug) {
                save_debug_tensor(hidden,
                    "debug_cpp/layer_" + std::to_string(i) + "_post_ttt.pt");
                save_debug_tensor(ttt_out,
                    "debug_cpp/layer_" + std::to_string(i) + "_ttt_out.pt");
                save_debug_tensor(gate,
                    "debug_cpp/layer_" + std::to_string(i) + "_ttt_gate.pt");
            }
        }

        // --- SWA ---
        auto swa_it = swa_layer_idx_.find(i);
        if (swa_it != swa_layer_idx_.end() && !turn_off_swa) {
            const int si = swa_it->second;
            auto& swa_layer = swa_layers_->at<BlockRopeImpl>(si);
            auto& gate_proj = swa_gate_projs_->at<torch::nn::LinearImpl>(si);

            // Ensure SWA state exists
            if (static_cast<int>(state.swa_kv.size()) <= si)
                state.swa_kv.resize(si + 1);

            auto h_for_swa = hidden.reshape({B, N * (PATCH_START + hw), DEC_DIM});
            auto pos_swa   = pos_full.reshape({B, N * (PATCH_START + hw), 2});

            torch::Tensor swa_out;
            if (state.swa_kv[si].first.defined()) {
                swa_out = swa_layer.forward_with_kv_cache(
                    h_for_swa,
                    state.swa_kv[si].first,
                    state.swa_kv[si].second,
                    pos_swa);
            } else {
                swa_out = swa_layer.forward(h_for_swa, pos_swa);
            }

            // Update KV cache
            auto [k_new, v_new] = swa_layer.compute_kv_cache(h_for_swa, pos_swa);
            state.swa_kv[si] = {k_new, v_new};

            auto gate = torch::silu(gate_proj.forward(swa_out));
            h_for_swa = h_for_swa + swa_out * gate;
            hidden = h_for_swa.reshape({BN, PATCH_START + hw, DEC_DIM});

            // Component debug: post-SWA
            if (do_comp_debug) {
                save_debug_tensor(hidden,
                    "debug_cpp/layer_" + std::to_string(i) + "_post_swa.pt");
                save_debug_tensor(swa_out,
                    "debug_cpp/layer_" + std::to_string(i) + "_swa_out.pt");
                save_debug_tensor(gate,
                    "debug_cpp/layer_" + std::to_string(i) + "_swa_gate.pt");
                if (i == DEC_DEPTH - 2) comp_debug_saved = true;  // after last SWA
            }
        }

        // Per-layer debug saves (set LOGER_DEBUG_LAYERS=1)
        {
            static bool layer_debug_done = false;
            if (!layer_debug_done && std::getenv("LOGER_DEBUG_LAYERS")) {
                save_debug_tensor(hidden,
                    "debug_cpp/decoder_layer_" + std::to_string(i) + ".pt");
                if (i == DEC_DEPTH - 1) layer_debug_done = true;
            }
        }

        // Track last 2 decoder outputs (indices 34 and 35)
        if (i == DEC_DEPTH - 2) second_prev_hidden = hidden;
        if (i == DEC_DEPTH - 1) prev_hidden         = hidden;
    }

    // Return the FULL sequence (special tokens + patch tokens) so that the task
    // decoders can attend to register/PE tokens — matching Python behaviour.
    // The caller (run_window) strips PATCH_START tokens before the task heads.
    auto concat_hidden = torch::cat({second_prev_hidden, prev_hidden}, /*dim=*/-1);
    // (BN, PATCH_START+hw, 2*DEC_DIM)

    state.initialized = true;

    return {concat_hidden, pos_full};  // (BN, PATCH_START+hw, 2)
}

// ---------------------------------------------------------------------------
// run_window
// ---------------------------------------------------------------------------

WindowPrediction Pi3Impl::run_window(torch::Tensor imgs_w,
                                      int window_size, int overlap_size,
                                      WindowState& state,
                                      bool turn_off_ttt,
                                      bool turn_off_swa) {
    // imgs_w: (B, N_w, 3, H, W) in [0,1]
    const int B   = imgs_w.size(0);
    const int N   = imgs_w.size(1);
    const int H   = imgs_w.size(3);
    const int W   = imgs_w.size(4);

    const bool timing = std::getenv("LOGER_TIMING") != nullptr;
    auto dev = imgs_w.device();

    // Normalize
    auto imgs = imgs_w.reshape({B * N, 3, H, W});
    imgs = (imgs - image_mean_.to(imgs.device())) / image_std_.to(imgs.device());

    // Encode
    CudaTimer t_enc(dev);
    auto hidden_input = encoder_->forward(imgs);  // (BN, hw, DEC_DIM)
    double ms_encode = t_enc.elapsed_ms();

    // Decode — returns FULL sequence: (BN, PATCH_START+hw, 2*DEC_DIM) + matching pos
    CudaTimer t_dec(dev);
    auto [hidden, pos] = decode(hidden_input, N, H, W,
                                 window_size, overlap_size,
                                 state, turn_off_ttt, turn_off_swa);
    double ms_decode = t_dec.elapsed_ms();

    // Debug: save first-window tensors for comparison with Python reference.
    static bool debug_saved = false;
    const bool do_debug = !debug_saved && (std::getenv("LOGER_DEBUG") != nullptr);
    if (do_debug) {
        debug_saved = true;
        save_debug_tensor(hidden_input,   "debug_cpp/encoder_out.pt");
        save_debug_tensor(hidden,         "debug_cpp/decoder_concat.pt");
    }

    // Task decoders receive the full sequence (PATCH_START special tokens + patch tokens)
    // so they can attend to register/PE context — matching Python.
    CudaTimer t_heads(dev);
    using SL = torch::indexing::Slice;

    // --- Points ---
    auto point_hidden_full = point_decoder_->forward(hidden, pos);  // (BN, PATCH_START+hw, DEC_DIM)
    auto point_hidden = point_hidden_full.index({SL(), SL(PATCH_START), SL()});  // (BN, hw, DEC_DIM)
    auto pts_out      = point_head_->forward(point_hidden, H, W);  // (BN, H, W, 3)
    // Decode local points: [x*z, y*z, z] where z=exp(z_raw)
    auto xy  = pts_out.index({SL(), SL(), SL(), SL(0, 2)});
    auto z   = pts_out.index({SL(), SL(), SL(), SL(2, 3)});
    z = torch::exp(z.clamp_max(15.0f));
    auto local_pts = torch::cat({xy * z, z}, -1);  // (BN, H, W, 3)
    local_pts = local_pts.reshape({B, N, H, W, 3});

    // --- Confidence ---
    auto conf_hidden_full = conf_decoder_->forward(hidden, pos);  // (BN, PATCH_START+hw, DEC_DIM)
    auto conf_hidden = conf_hidden_full.index({SL(), SL(PATCH_START), SL()});  // (BN, hw, DEC_DIM)
    auto conf        = conf_head_->forward(conf_hidden, H, W);  // (BN, H, W, 1)
    conf = conf.reshape({B, N, H, W, 1});

    if (do_debug) {
        save_debug_tensor(conf_hidden_full, "debug_cpp/conf_decoder_out.pt");
        save_debug_tensor(point_hidden_full, "debug_cpp/point_decoder_out.pt");
    }

    // --- Camera poses ---
    int ph = H / DinoV2EncoderImpl::PATCH_SIZE;
    int pw = W / DinoV2EncoderImpl::PATCH_SIZE;
    auto cam_hidden_full = camera_decoder_->forward(hidden, pos);  // (BN, PATCH_START+hw, 512)
    auto cam_hidden = cam_hidden_full.index({SL(), SL(PATCH_START), SL()});  // (BN, hw, 512)
    auto cam_poses  = camera_head_->forward(cam_hidden, ph, pw);  // (BN, 4, 4)
    cam_poses = cam_poses.reshape({B, N, 4, 4});

    if (do_debug) {
        save_debug_tensor(pts_out,               "debug_cpp/point_head_raw.pt");
        save_debug_tensor(local_pts,             "debug_cpp/local_points.pt");
        save_debug_tensor(conf.reshape({B*N, H, W, 1}), "debug_cpp/conf_raw.pt");
        save_debug_tensor(cam_poses,             "debug_cpp/camera_poses.pt");
    }

    // --- Global points via unprojection ---
    auto hom    = homogenize_points(local_pts);  // (B, N, H, W, 4)
    auto points = torch::einsum("bnij,bnhwj->bnhwi",
                                {cam_poses, hom})
                      .index({torch::indexing::Ellipsis,
                               torch::indexing::Slice(0, 3)});  // (B, N, H, W, 3)

    double ms_heads = t_heads.elapsed_ms();

    if (timing) {
        double ms_total = ms_encode + ms_decode + ms_heads;
        std::cout << "[timing] window " << N << " frames: "
                  << "encode=" << std::fixed << std::setprecision(1) << ms_encode << "ms  "
                  << "decode=" << ms_decode << "ms  "
                  << "heads=" << ms_heads << "ms  "
                  << "total=" << ms_total << "ms  "
                  << "(" << ms_total / N << " ms/frame)\n";
    }

    return {points, local_pts, conf, cam_poses};
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

Pi3Impl::InferenceResult Pi3Impl::forward(torch::Tensor imgs,
                                           int window_size, int overlap_size,
                                           bool se3_align, int reset_every,
                                           bool turn_off_ttt, bool turn_off_swa,
                                           const std::string& save_windows_dir) {
    // imgs: (B, N, 3, H, W)
    const int B = imgs.size(0);
    const int N = imgs.size(1);
    const int H = imgs.size(3);
    const int W = imgs.size(4);

    const bool timing = std::getenv("LOGER_TIMING") != nullptr;
    auto dev = imgs.device();

    auto starts = compute_window_starts(N, window_size, overlap_size);

    if (timing) {
        std::cout << "[timing] " << N << " total frames, "
                  << starts.size() << " window(s), window_size="
                  << window_size << ", overlap=" << overlap_size << "\n";
    }

    CudaTimer t_total(dev);
    WindowState state;

    const size_t n_windows = starts.size();
    using S = torch::indexing::Slice;

    // Write metadata for the Python merge script
    if (!save_windows_dir.empty()) {
        std::filesystem::create_directories(save_windows_dir);
        std::ofstream meta(save_windows_dir + "/meta.txt");
        meta << "n_windows " << n_windows << "\n"
             << "window_size " << window_size << "\n"
             << "overlap_size " << overlap_size << "\n"
             << "total_frames " << N << "\n"
             << "H " << H << "\n"
             << "W " << W << "\n";
        for (size_t i = 0; i < starts.size(); ++i)
            meta << "start " << i << " " << starts[i] << "\n";
    }

    // Pre-allocate output on CPU (float32) — only points, conf, poses.
    // local_points is consumed per-window for depth_edge then discarded.
    auto cpu_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto out_points = torch::empty({B, N, H, W, 3}, cpu_f32);
    auto out_conf   = torch::empty({B, N, H, W, 1}, cpu_f32);
    auto out_poses  = torch::empty({B, N, 4, 4},    cpu_f32);

    // For SE3 alignment: keep only the last overlap_size frames of previous
    // aligned window (camera_poses + local_points needed by estimate_se3/apply_se3).
    WindowPrediction overlap_ref;
    int write_offset = 0;

    for (size_t wi = 0; wi < n_windows; ++wi) {
        int s   = starts[wi];
        int end = std::min(s + (window_size > 0 ? window_size : N), N);

        std::cout << "\r[inference] Window " << (wi + 1) << "/" << n_windows
                  << " (frames " << s << "-" << (end - 1) << ")" << std::flush;

        auto imgs_w = imgs.index({S(), S(s, end)});

        // Reset TTT state periodically if requested
        if (reset_every > 0 && wi > 0 && (wi % reset_every == 0)) {
            std::vector<FastWeightGluMLPMultihead> ttt_vec;
            for (int i = 0; i < static_cast<int>(ttt_layers_->size()); ++i)
                ttt_vec.push_back(ttt_layers_->ptr<FastWeightGluMLPMultiheadImpl>(i));
            state.reset_ttt(ttt_vec, B, imgs.device(), imgs.scalar_type());
        }

        // Run model on GPU, then move result to CPU
        auto pred = run_window(imgs_w, window_size, overlap_size,
                               state, turn_off_ttt, turn_off_swa)
                        .to(torch::kCPU);

        // Save raw (pre-merge) window predictions if requested
        if (!save_windows_dir.empty()) {
            std::filesystem::create_directories(save_windows_dir);
            auto prefix = save_windows_dir + "/window_" + std::to_string(wi);
            OutputWriter::save_npy(prefix + "_points.npy",      pred.points);
            OutputWriter::save_npy(prefix + "_local_points.npy", pred.local_points);
            OutputWriter::save_npy(prefix + "_conf.npy",         pred.conf);
            OutputWriter::save_npy(prefix + "_poses.npy",        pred.camera_poses);
            // Save images for this window: (B, N_w, 3, H, W) → (B, N_w, H, W, 3)
            auto imgs_nhwc = imgs_w.to(torch::kFloat32).cpu()
                                 .permute({0, 1, 3, 4, 2})
                                 .clamp(0.0f, 1.0f)
                                 .contiguous();
            OutputWriter::save_npy(prefix + "_images.npy", imgs_nhwc);
        }

        // SE3 alignment against previous window's overlap region
        if (wi > 0 && se3_align) {
            auto T = estimate_se3(overlap_ref, pred, overlap_size, /*allow_scale=*/false);
            pred = apply_se3(pred, T, /*allow_scale=*/false);
        }

        // Determine frames to write (skip overlap prefix for windows after first)
        int skip    = (wi == 0) ? 0 : overlap_size;
        int n_write = pred.points.size(1) - skip;

        // Per-window confidence post-processing (avoids storing local_points)
        auto z_chan = pred.local_points.index({S(), S(skip, torch::indexing::None),
                                               S(), S(), 2});
        auto edges  = depth_edge(z_chan);
        auto conf_pp = apply_conf_postprocess(
            pred.conf.index({S(), S(skip, torch::indexing::None)}), edges);

        // Copy into pre-allocated output
        out_points.index_put_({S(), S(write_offset, write_offset + n_write)},
                              pred.points.index({S(), S(skip, torch::indexing::None)})
                                  .to(torch::kFloat32));
        out_conf.index_put_({S(), S(write_offset, write_offset + n_write)},
                            conf_pp.to(torch::kFloat32));
        out_poses.index_put_({S(), S(write_offset, write_offset + n_write)},
                             pred.camera_poses.index({S(), S(skip, torch::indexing::None)})
                                 .to(torch::kFloat32));
        write_offset += n_write;

        // Keep overlap reference for next window (only last overlap_size frames)
        if (wi + 1 < n_windows && overlap_size > 0) {
            overlap_ref = WindowPrediction{
                pred.points.index({S(), S(-overlap_size, torch::indexing::None)}),
                pred.local_points.index({S(), S(-overlap_size, torch::indexing::None)}),
                pred.conf.index({S(), S(-overlap_size, torch::indexing::None)}),
                pred.camera_poses.index({S(), S(-overlap_size, torch::indexing::None)})
            };
        }
    }
    std::cout << "\n";

    // Trim in case last window was shorter
    out_points = out_points.index({S(), S(0, write_offset)});
    out_conf   = out_conf.index({S(), S(0, write_offset)});
    out_poses  = out_poses.index({S(), S(0, write_offset)});

    double ms_total_elapsed = t_total.elapsed_ms();
    if (timing) {
        std::cout << "[timing] total=" << std::fixed << std::setprecision(1)
                  << ms_total_elapsed << "ms for " << N
                  << " frames → " << ms_total_elapsed / N << " ms/frame ("
                  << 1000.0 / (ms_total_elapsed / N) << " fps)\n";
    }

    return {out_points, /*local_points=*/out_points, out_conf, out_poses};
}

} // namespace loger
