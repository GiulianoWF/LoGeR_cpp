#include <loger/io/weight_loader.hpp>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace loger {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string strip_module_prefix(const std::string& key) {
    const std::string pfx = "module.";
    if (key.size() > pfx.size() && key.substr(0, pfx.size()) == pfx)
        return key.substr(pfx.size());
    return key;
}

void TensorStore::insert_from_dict(
    const c10::Dict<c10::IValue, c10::IValue>& dict,
    const std::string& /*prefix*/)
{
    for (const auto& kv : dict) {
        const std::string raw_key = kv.key().toStringRef();
        const std::string key     = strip_module_prefix(raw_key);

        if (kv.value().isTensor()) {
            store_[key] = kv.value().toTensor();
        } else if (kv.value().isGenericDict()) {
            // Nested dict — shouldn't happen for flat state_dicts,
            // but handle gracefully by recursing with key prefix.
            auto nested = kv.value().toGenericDict();
            for (const auto& nkv : nested) {
                const std::string nkey = key + "." + nkv.key().toStringRef();
                if (nkv.value().isTensor())
                    store_[strip_module_prefix(nkey)] = nkv.value().toTensor();
            }
        }
        // Skip non-tensor values (e.g. metadata stored alongside weights)
    }
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void TensorStore::load(const std::string& path) {
    store_.clear();

    // Checkpoints must be pre-converted from collections.OrderedDict to a
    // plain Python dict before use (see scripts/convert_checkpoint.py).
    // The C++ libtorch unpickler cannot handle collections.OrderedDict
    // (GLOBAL/REDUCE opcode pattern); plain dicts use EMPTY_DICT/SETITEMS
    // which it handles natively.
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("TensorStore: cannot open file: " + path);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(size);
    if (!file.read(bytes.data(), size))
        throw std::runtime_error("TensorStore: failed to read file: " + path);

    // torch::pickle_load (libtorch 2.7+) is a zip reader that looks for the pickle
    // at "data/data.pkl" inside the archive.  Checkpoints saved as 'latest.pt' use
    // the prefix 'latest', not 'data', so pickle_load returns None and crashes.
    // scripts/convert_checkpoint.py re-saves via a temp 'data.pt' to fix this.
    //
    // If the file is NOT a zip at all (raw legacy pickle), the reader will also fail.
    // In either case the error from torch::pickle_load is the authoritative one.
    c10::IValue ivalue = torch::pickle_load(bytes);

    // The top-level value may be a dict or an OrderedDict (both appear as
    // GenericDict in IValue).
    if (!ivalue.isGenericDict())
        throw std::runtime_error(
            "TensorStore: expected a dict at top level, got: " +
            ivalue.tagKind());

    auto top_dict = ivalue.toGenericDict();

    // Check for "model_state_dict" wrapper (some checkpoints wrap the state dict)
    if (top_dict.contains("model_state_dict")) {
        auto inner = top_dict.at("model_state_dict");
        if (inner.isGenericDict())
            insert_from_dict(inner.toGenericDict());
        else
            throw std::runtime_error(
                "TensorStore: 'model_state_dict' key is not a dict");
    } else {
        insert_from_dict(top_dict);
    }

    std::cout << "[TensorStore] Loaded " << store_.size()
              << " tensors from " << path << "\n";
}

torch::Tensor TensorStore::get(const std::string& key) const {
    auto it = store_.find(key);
    if (it == store_.end())
        throw std::runtime_error("TensorStore: key not found: " + key);
    return it->second;
}

bool TensorStore::has(const std::string& key) const {
    return store_.count(key) > 0;
}

std::vector<std::string> TensorStore::keys() const {
    std::vector<std::string> out;
    out.reserve(store_.size());
    for (const auto& kv : store_)
        out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

void TensorStore::copy_weight(torch::Tensor& dest,
                              const TensorStore& store,
                              const std::string& key) {
    auto src = store.get(key);

    // Shape check
    if (dest.sizes() != src.sizes()) {
        auto sizes_str = [](at::IntArrayRef s) {
            std::ostringstream ss; ss << s; return ss.str();
        };
        throw std::runtime_error(
            "TensorStore::copy_weight shape mismatch for key '" + key +
            "': dest=" + sizes_str(dest.sizes()) +
            " src=" + sizes_str(src.sizes()));
    }

    torch::NoGradGuard ng;
    dest.copy_(src.to(dest.dtype()));
}

void TensorStore::copy_weight_optional(torch::Tensor& dest,
                                       const TensorStore& store,
                                       const std::string& key) {
    if (store.has(key))
        copy_weight(dest, store, key);
}

} // namespace loger
