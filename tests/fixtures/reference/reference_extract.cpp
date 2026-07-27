#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static uint32_t f32_bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static std::string hex_bits(float value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(8) << f32_bits(value);
    return stream.str();
}

static void write_u32le(std::ofstream & output, uint32_t value) {
    std::array<char, 4> bytes = {
        static_cast<char>(value), static_cast<char>(value >> 8),
        static_cast<char>(value >> 16), static_cast<char>(value >> 24),
    };
    output.write(bytes.data(), bytes.size());
}

static void write_tokens(const fs::path & path, const std::vector<llama_token> & tokens) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    for (const llama_token token : tokens) write_u32le(output, static_cast<uint32_t>(token));
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

static void write_logits(const fs::path & path, const float * logits, int32_t count) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    for (int32_t index = 0; index < count; ++index) {
        if (!std::isfinite(logits[index])) throw std::runtime_error("non-finite model logit");
        write_u32le(output, f32_bits(logits[index]));
    }
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

struct Step {
    int step;
    llama_token selected;
    float selected_logit;
    std::array<std::pair<llama_token, float>, 8> top;
};

static Step select_step(const float * logits, int32_t vocab_count, int step) {
    llama_token selected = 0;
    float best = logits[0];
    if (!std::isfinite(best)) throw std::runtime_error("non-finite model logit");
    std::vector<std::pair<llama_token, float>> candidates;
    candidates.reserve(static_cast<size_t>(vocab_count));
    for (int32_t token = 0; token < vocab_count; ++token) {
        const float value = logits[token];
        if (!std::isfinite(value)) throw std::runtime_error("non-finite model logit");
        if (value > best) {
            best = value;
            selected = token;
        }
        candidates.emplace_back(token, value);
    }
    std::partial_sort(candidates.begin(), candidates.begin() + 8, candidates.end(),
        [](const auto & left, const auto & right) {
            if (left.second != right.second) return left.second > right.second;
            return left.first < right.first;
        });
    Step result{step, selected, best, {}};
    std::copy_n(candidates.begin(), 8, result.top.begin());
    return result;
}

static void write_steps(const fs::path & path, const std::vector<Step> & steps) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output << "{\"steps\":[";
    for (size_t index = 0; index < steps.size(); ++index) {
        if (index) output << ',';
        const Step & step = steps[index];
        output << "{\"selected_id\":" << step.selected
               << ",\"selected_logit_bits\":\"" << hex_bits(step.selected_logit)
               << "\",\"step\":" << step.step << ",\"top8\":[";
        for (size_t top_index = 0; top_index < step.top.size(); ++top_index) {
            if (top_index) output << ',';
            output << "{\"id\":" << step.top[top_index].first
                   << ",\"logit_bits\":\"" << hex_bits(step.top[top_index].second) << "\"}";
        }
        output << "]}";
    }
    output << "]}\n";
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

static constexpr llama_token TOKEN_IM_START = 151644;
static constexpr llama_token TOKEN_IM_END = 151645;

struct Prompt {
    std::string rendered;
    std::vector<llama_token> tokens;
};

static void validate_user(const std::string & user) {
    if (user.empty() || user.size() > 512) throw std::runtime_error("user message length out of bounds");
    for (const unsigned char byte : user) {
        if (byte < 0x20 || byte > 0x7e) throw std::runtime_error("user message is not printable ASCII");
    }
}

static void append_ordinary_tokens(
    const llama_vocab * vocab,
    std::string_view text,
    std::vector<llama_token> & tokens
) {
    if (text.empty() || text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("ordinary prompt segment length out of bounds");
    }
    const int32_t text_size = static_cast<int32_t>(text.size());
    const int32_t required = -llama_tokenize(
        vocab, text.data(), text_size, nullptr, 0, false, false
    );
    if (required <= 0 || required > 480) {
        throw std::runtime_error("ordinary prompt segment token count out of bounds");
    }
    const size_t old_size = tokens.size();
    tokens.resize(old_size + static_cast<size_t>(required));
    const int32_t written = llama_tokenize(
        vocab, text.data(), text_size, tokens.data() + old_size, required, false, false
    );
    if (written != required) throw std::runtime_error("ordinary prompt segment tokenization failed");
}

static Prompt build_segmented_prompt(const llama_vocab * vocab, const std::string & user) {
    validate_user(user);
    Prompt prompt;
    prompt.rendered.reserve(160 + user.size());
    prompt.tokens.reserve(64);
    const auto marker = [&prompt](std::string_view rendered, llama_token token) {
        prompt.rendered.append(rendered);
        prompt.tokens.push_back(token);
    };
    const auto ordinary = [&prompt, vocab](std::string_view text) {
        prompt.rendered.append(text);
        append_ordinary_tokens(vocab, text, prompt.tokens);
    };

    // Only these renderer-owned markers enter the token stream as control IDs.
    marker("<|im_start|>", TOKEN_IM_START);
    ordinary("system");
    ordinary("\n");
    ordinary("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
    marker("<|im_end|>", TOKEN_IM_END);
    ordinary("\n");
    marker("<|im_start|>", TOKEN_IM_START);
    ordinary("user");
    ordinary("\n");
    ordinary(user);
    marker("<|im_end|>", TOKEN_IM_END);
    ordinary("\n");
    marker("<|im_start|>", TOKEN_IM_START);
    ordinary("assistant");
    ordinary("\n");

    if (prompt.tokens.empty() || prompt.tokens.size() > 480) {
        throw std::runtime_error("prompt token count out of bounds");
    }
    return prompt;
}

int main(int argc, char ** argv) {
    const bool tokenize_only = argc == 5 && std::string_view(argv[1]) == "--tokenize-only";
    const int argument_offset = tokenize_only ? 1 : 0;
    if (argc != 4 + argument_offset) {
        std::cerr << "REFERENCE_USAGE expected: reference_extract [--tokenize-only] MODEL USER OUTPUT_DIR\n";
        return 50;
    }
    const fs::path model_path = argv[1 + argument_offset];
    const std::string user = argv[2 + argument_offset];
    const fs::path output_dir = argv[3 + argument_offset];
    try {
        fs::create_directories(output_dir);
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.use_mmap = true;
        model_params.use_direct_io = false;
        model_params.use_mlock = false;
        model_params.check_tensors = true;
        model_params.use_extra_bufts = false;
        model_params.no_host = false;
        model_params.no_alloc = false;
        llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (model == nullptr) throw std::runtime_error("llama_model_load_from_file failed");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t vocab_count = llama_vocab_n_tokens(vocab);
        if (vocab_count != 151936) throw std::runtime_error("vocabulary identity mismatch");
        if (llama_vocab_eos(vocab) != TOKEN_IM_END) {
            throw std::runtime_error("EOS/control-token identity mismatch");
        }

        Prompt prompt = build_segmented_prompt(vocab, user);
        {
            std::ofstream output(output_dir / "prompt.txt", std::ios::binary | std::ios::trunc);
            output.write(prompt.rendered.data(), static_cast<std::streamsize>(prompt.rendered.size()));
            if (!output) throw std::runtime_error("failed writing prompt.txt");
        }
        write_tokens(output_dir / "prompt_tokens.u32le", prompt.tokens);
        if (tokenize_only) {
            llama_model_free(model);
            llama_backend_free();
            std::cout << "REFERENCE_TOKENIZATION_PASS prompt_tokens=" << prompt.tokens.size()
                      << " output=" << output_dir << "\n";
            return 0;
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 512;
        context_params.n_batch = 512;
        context_params.n_ubatch = 512;
        context_params.n_seq_max = 1;
        context_params.n_outputs_max = 1;
        context_params.n_threads = 1;
        context_params.n_threads_batch = 1;
        context_params.type_k = GGML_TYPE_F32;
        context_params.type_v = GGML_TYPE_F32;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.embeddings = false;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        context_params.no_perf = true;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) throw std::runtime_error("llama_init_from_model failed");
        const int32_t prompt_token_count = static_cast<int32_t>(prompt.tokens.size());
        if (llama_decode(context, llama_batch_get_one(prompt.tokens.data(), prompt_token_count)) != 0) {
            throw std::runtime_error("prompt llama_decode failed");
        }

        const float * logits = llama_get_logits_ith(context, -1);
        if (logits == nullptr) throw std::runtime_error("prompt logits unavailable");
        write_logits(output_dir / "prompt_final_logits.f32le", logits, vocab_count);
        std::vector<llama_token> continuation;
        std::vector<Step> steps;
        for (int step_index = 0; step_index < 16; ++step_index) {
            Step step = select_step(logits, vocab_count, step_index);
            continuation.push_back(step.selected);
            steps.push_back(step);
            if (step.selected == llama_vocab_eos(vocab)) break;
            llama_token token = step.selected;
            if (llama_decode(context, llama_batch_get_one(&token, 1)) != 0) {
                throw std::runtime_error("continuation llama_decode failed");
            }
            logits = llama_get_logits_ith(context, -1);
            if (logits == nullptr) throw std::runtime_error("continuation logits unavailable");
        }
        write_tokens(output_dir / "continuation.u32le", continuation);
        write_steps(output_dir / "steps.json", steps);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        std::cout << "REFERENCE_EXTRACTED prompt_tokens=" << prompt_token_count
                  << " continuation_tokens=" << continuation.size() << " output=" << output_dir << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "REFERENCE_EXTRACTION_FAILED " << error.what() << "\n";
        return 51;
    }
}
