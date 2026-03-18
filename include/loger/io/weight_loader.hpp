#pragma once
#include <string>
#include <unordered_map>
#include <torch/torch.h>

namespace loger {

/// Flat key->tensor store loaded from a PyTorch .pt state dict file.
class TensorStore {
public:
    /// Load state dict from path. Handles:
    ///  - "model_state_dict" wrapper key
    ///  - "module." DDP prefix stripping
    void load(const std::string& path);

    /// Get tensor by exact key. Throws if not found.
    torch::Tensor get(const std::string& key) const;

    /// Check if key exists.
    bool has(const std::string& key) const;

    /// List all keys (for debugging).
    std::vector<std::string> keys() const;

    /// Copy a weight into a parameter tensor. Asserts shape match,
    /// casts dtype to match dest if needed.
    static void copy_weight(torch::Tensor& dest,
                            const TensorStore& store,
                            const std::string& key);

    /// Same as copy_weight but doesn't throw if key is absent.
    static void copy_weight_optional(torch::Tensor& dest,
                                     const TensorStore& store,
                                     const std::string& key);

private:
    std::unordered_map<std::string, torch::Tensor> store_;

    void insert_from_dict(const c10::Dict<c10::IValue, c10::IValue>& dict,
                          const std::string& prefix = "");
};

} // namespace loger
