#include "llama.h"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
static constexpr llama_token IM_START = 151644;
static constexpr llama_token IM_END = 151645;

static void ordinary(const llama_vocab * vocab, std::string_view text, std::vector<llama_token> & out) {
    if (text.empty()) return;
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) throw std::runtime_error("text too long");
    const int32_t length = static_cast<int32_t>(text.size());
    const int32_t needed = -llama_tokenize(vocab, text.data(), length, nullptr, 0, false, false);
    if (needed <= 0 || out.size() + static_cast<size_t>(needed) > 599) throw std::runtime_error("token bound");
    const size_t at = out.size(); out.resize(at + needed);
    if (llama_tokenize(vocab, text.data(), length, out.data() + at, needed, false, false) != needed) {
        throw std::runtime_error("tokenize failed");
    }
}

static std::vector<llama_token> fresh_prompt(const llama_vocab * vocab, const std::string & user) {
    if (user.size() > 512) throw std::runtime_error("user bound");
    for (unsigned char byte : user) if (byte < 0x20 || byte > 0x7e) throw std::runtime_error("user byte");
    std::vector<llama_token> out;
    out.push_back(IM_START); ordinary(vocab, "system", out); ordinary(vocab, "\n", out);
    ordinary(vocab, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", out);
    out.push_back(IM_END); ordinary(vocab, "\n", out); out.push_back(IM_START);
    ordinary(vocab, "user", out); ordinary(vocab, "\n", out); ordinary(vocab, user, out);
    out.push_back(IM_END); ordinary(vocab, "\n", out); out.push_back(IM_START);
    ordinary(vocab, "assistant", out); ordinary(vocab, "\n", out);
    return out;
}

static std::vector<llama_token> history_prompt(
    const llama_vocab * vocab, const std::vector<llama_token> & history, const std::string & user
) {
    if (history.empty() || history.back() != IM_END) throw std::runtime_error("history state");
    std::vector<llama_token> out = history;
    ordinary(vocab, "\n", out); out.push_back(IM_START); ordinary(vocab, "user", out);
    ordinary(vocab, "\n", out); ordinary(vocab, user, out); out.push_back(IM_END);
    ordinary(vocab, "\n", out); out.push_back(IM_START); ordinary(vocab, "assistant", out);
    ordinary(vocab, "\n", out);
    return out;
}

struct Token { llama_token id; std::string kind; std::string hex; };
struct Turn { std::vector<llama_token> prompt; std::vector<Token> generated; std::string reason; };

static std::string hex(std::string_view value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : value) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

static Turn generate(llama_model * model, const llama_vocab * vocab, std::vector<llama_token> prompt) {
    if (prompt.empty() || prompt.size() > 480) throw std::runtime_error("prompt context");
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 512; params.n_batch = 512; params.n_ubatch = 512; params.n_seq_max = 1;
    params.n_outputs_max = 1; params.n_threads = 1; params.n_threads_batch = 1;
    params.type_k = GGML_TYPE_F32; params.type_v = GGML_TYPE_F32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; params.embeddings = false;
    params.offload_kqv = false; params.op_offload = false; params.no_perf = true;
    llama_context * context = llama_init_from_model(model, params);
    if (!context) throw std::runtime_error("context init");
    if (llama_decode(context, llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()))) != 0) {
        llama_free(context); throw std::runtime_error("prompt decode");
    }
    Turn result{std::move(prompt), {}, "MAX_NEW_TOKENS"};
    const int32_t vocab_count = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(context, -1);
    for (int index = 0; index < 32; ++index) {
        if (!logits) { llama_free(context); throw std::runtime_error("logits"); }
        llama_token selected = 0;
        for (llama_token token = 1; token < vocab_count; ++token) if (logits[token] > logits[selected]) selected = token;
        std::array<char, 128> piece{};
        int32_t length = llama_token_to_piece(vocab, selected, piece.data(), piece.size(), 0, false);
        if (length < 0 || length > static_cast<int32_t>(piece.size())) { llama_free(context); throw std::runtime_error("piece"); }
        const std::string kind = selected == IM_END ? "EOS" : length == 0 ? "SUPPRESSED" : "TEXT";
        result.generated.push_back({selected, kind, hex(std::string_view(piece.data(), length))});
        if (selected == IM_END) { result.reason = "EOS"; break; }
        if (llama_decode(context, llama_batch_get_one(&selected, 1)) != 0) {
            llama_free(context); throw std::runtime_error("token decode");
        }
        logits = llama_get_logits_ith(context, -1);
    }
    llama_free(context);
    return result;
}

static void json_turn(std::ostream & out, const Turn & turn) {
    out << "{\"generated\":[";
    for (size_t at = 0; at < turn.generated.size(); ++at) {
        if (at) out << ',';
        const Token & token = turn.generated[at];
        out << "{\"id\":" << token.id << ",\"kind\":\"" << token.kind
            << "\",\"piece_hex\":\"" << token.hex << "\"}";
    }
    out << "],\"prompt_tokens\":[";
    for (size_t at = 0; at < turn.prompt.size(); ++at) { if (at) out << ','; out << turn.prompt[at]; }
    out << "],\"reason\":\"" << turn.reason << "\"}";
}

int main(int argc, char ** argv) {
    const bool count_only = argc == 4 && std::string_view(argv[1]) == "--token-count";
    if (!count_only && argc != 5 && argc != 6) {
        std::cerr << "REPL_REFERENCE_USAGE [--token-count MODEL USER] | MODEL OUTPUT_JSON CASE USER1 [USER2]\n"; return 50;
    }
    try {
        llama_backend_init();
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 0; params.use_mmap = true; params.use_mlock = false;
        params.check_tensors = true; params.use_extra_bufts = false;
        llama_model * model = llama_model_load_from_file(argv[count_only ? 2 : 1], params);
        if (!model) throw std::runtime_error("model load");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (llama_vocab_n_tokens(vocab) != 151936 || llama_vocab_eos(vocab) != IM_END) throw std::runtime_error("model identity");
        if (count_only) {
            const auto tokens = fresh_prompt(vocab, argv[3]);
            std::cout << "REPL_TOKEN_COUNT tokens=" << tokens.size() << "\n";
            llama_model_free(model); llama_backend_free(); return 0;
        }
        Turn first = generate(model, vocab, fresh_prompt(vocab, argv[4]));
        std::vector<Turn> turns{first};
        if (argc == 6) {
            if (first.reason != "EOS") throw std::runtime_error("turn1 did not commit");
            std::vector<llama_token> history = first.prompt;
            for (const Token & token : first.generated) history.push_back(token.id);
            turns.push_back(generate(model, vocab, history_prompt(vocab, history, argv[5])));
        }
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        output << "{\"case\":\"" << argv[3] << "\",\"llama_revision\":\"571d0d540df04f25298d0e159e520d9fc62ed121\",\"turns\":[";
        for (size_t at = 0; at < turns.size(); ++at) { if (at) output << ','; json_turn(output, turns[at]); }
        output << "]}\n";
        if (!output) throw std::runtime_error("output");
        llama_model_free(model); llama_backend_free();
        std::cout << "REPL_REFERENCE_PASS case=" << argv[3] << " turns=" << turns.size() << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "REPL_REFERENCE_FAILED " << error.what() << "\n"; return 51;
    }
}
