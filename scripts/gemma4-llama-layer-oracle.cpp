// Bounded Gemma 4 layer-state oracle for llama.cpp b10360 (48d22e295).
// This replaces the eval-callback example only in a disposable source checkout.

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static bool capture_layer_outputs = false;

static bool should_capture_layer_outputs(const ggml_tensor * tensor) {
    return capture_layer_outputs && std::strncmp(tensor->name, "l_out-", 6) == 0;
}

static llama_token emit_logits(llama_context * context, const llama_vocab * vocab, int token_index) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(context, -1);
    int32_t top_token = 0;
    int32_t runner_up_token = 0;
    for (int32_t token = 1; token < n_vocab; ++token) {
        if (logits[token] > logits[top_token]) {
            runner_up_token = top_token;
            top_token = token;
        } else if (token != top_token && logits[token] > logits[runner_up_token]) {
            runner_up_token = token;
        }
    }
    std::printf("{\"event\":\"generated_token\",\"token_index\":%d,\"token_id\":%d,\"top_logit\":%.9g,\"runner_up_token_id\":%d,\"runner_up_logit\":%.9g}\n",
        token_index, top_token, logits[top_token], runner_up_token, logits[runner_up_token]);
    std::fflush(stdout);
    return top_token;
}

static bool emit_layer_summary(ggml_tensor * tensor, bool ask, void *) {
    if (ask) return should_capture_layer_outputs(tensor);
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
        std::fprintf(stderr, "unexpected layer output `%s`: type=%s shape=[%lld,%lld]\n",
            tensor->name, ggml_type_name(tensor->type),
            static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]));
        return false;
    }
    const size_t bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> copied;
    const uint8_t * data;
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        data = static_cast<const uint8_t *>(tensor->data);
    } else {
        copied.resize(bytes);
        ggml_backend_tensor_get(tensor, copied.data(), 0, bytes);
        data = copied.data();
    }
    const int64_t last_token = tensor->ne[1] - 1;
    double squared_sum = 0.0;
    double sum = 0.0;
    float max_abs = 0.0f;
    size_t non_finite = 0;
    for (int64_t index = 0; index < tensor->ne[0]; ++index) {
        float value;
        std::memcpy(&value, data + index * tensor->nb[0] + last_token * tensor->nb[1], sizeof(value));
        if (!std::isfinite(value)) {
            ++non_finite;
            continue;
        }
        sum += value;
        squared_sum += static_cast<double>(value) * value;
        max_abs = std::max(max_abs, std::fabs(value));
    }
    std::printf("{\"event\":\"layer_state\",\"name\":\"%s\",\"width\":%lld,\"last_token\":%lld,\"sum\":%.9g,\"l2_norm\":%.9g,\"max_abs\":%.9g,\"non_finite\":%zu}\n",
        tensor->name, static_cast<long long>(tensor->ne[0]), static_cast<long long>(last_token),
        sum, std::sqrt(squared_sum), max_abs, non_finite);
    std::fflush(stdout);
    return true;
}

static bool run(llama_context * context, const common_params & params) {
    // Atlas supplies its raw template and never injects BOS. `true` parses
    // Gemma turn markers as special vocabulary tokens.
    std::vector<llama_token> tokens = common_tokenize(context, params.prompt, false, true);
    if (tokens.empty()) {
        LOG_ERR("%s : prompt tokenizes to no tokens\n", __func__);
        return false;
    }
    std::printf("{\"event\":\"prompt_tokens\",\"token_ids\":[");
    for (size_t index = 0; index < tokens.size(); ++index) {
        std::printf("%s%d", index == 0 ? "" : ",", tokens[index]);
    }
    std::printf("]}\n");
    std::fflush(stdout);
    const char * capture_index_text = std::getenv("ATLAS_LLAMA_ORACLE_CAPTURE_TOKEN_INDEX");
    int capture_index = -1;
    if (capture_index_text != nullptr) {
        capture_index = std::atoi(capture_index_text);
        if (capture_index < 0) {
            LOG_ERR("%s : ATLAS_LLAMA_ORACLE_CAPTURE_TOKEN_INDEX must be non-negative\n", __func__);
            return false;
        }
    }
    capture_layer_outputs = capture_index == 0;
    if (llama_decode(context, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to evaluate prompt\n", __func__);
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(context));
    const int n_predict = params.n_predict;
    if (n_predict == 0) {
        emit_logits(context, vocab, 0);
        return true;
    }
    if (n_predict < 0) {
        LOG_ERR("%s : require an explicit positive --n-predict for raw-token generation\n", __func__);
        return false;
    }
    for (int index = 0; index < n_predict; ++index) {
        const llama_token next = emit_logits(context, vocab, index);
        if (llama_vocab_is_eog(vocab, next)) break;
        if (index + 1 < n_predict) {
            capture_layer_outputs = index + 1 == capture_index;
            if (llama_decode(context, llama_batch_get_one(const_cast<llama_token *>(&next), 1))) {
            LOG_ERR("%s : failed to evaluate generated token\n", __func__);
            return false;
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = emit_layer_summary;
    params.warmup = false;
    auto llama_init = common_init_from_params(params);
    auto * context = llama_init->context();
    if (llama_init->model() == nullptr || context == nullptr) {
        LOG_ERR("%s : failed to initialize llama.cpp\n", __func__);
        return 1;
    }
    const bool ok = run(context, params);
    llama_backend_free();
    return ok ? 0 : 1;
}
