#include <metal_stdlib>
using namespace metal;

kernel void vector_add_f32(
    device const float *lhs [[buffer(0)]],
    device const float *rhs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) {
        output[id] = lhs[id] + rhs[id];
    }
}

kernel void scalar_multiply_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant float &scalar [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) {
        output[id] = input[id] * scalar;
    }
}

kernel void silu_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) {
        float x = input[id];
        output[id] = x / (1.0f + exp(-x));
    }
}

kernel void reduction_sum_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    // Phase 0 favors deterministic validation over parallel reduction speed.
    if (id == 0) {
        float sum = 0.0f;
        for (uint index = 0; index < count; ++index) {
            sum += input[index];
        }
        output[0] = sum;
    }
}

kernel void transpose_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint2 id [[thread_position_in_grid]]) {
    if (id.x < cols && id.y < rows) {
        output[id.x * rows + id.y] = input[id.y * cols + id.x];
    }
}

kernel void vector_multiply_f32(
    device const float *lhs [[buffer(0)]], device const float *rhs [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) { output[id] = lhs[id] * rhs[id]; }
}

kernel void embedding_lookup_f32(
    device const float *table [[buffer(0)]], device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &vocabulary [[buffer(3)]],
    constant uint &hidden [[buffer(4)]], constant uint &tokens [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    uint total = tokens * hidden;
    if (id < total) {
        uint token = token_ids[id / hidden];
        output[id] = token < vocabulary ? table[token * hidden + id % hidden] : 0.0f;
    }
}

kernel void rms_norm_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    constant float &epsilon [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    float squared_sum = 0.0f;
    for (uint column = 0; column < hidden; ++column) { float x = input[row * hidden + column]; squared_sum += x * x; }
    float inverse_rms = rsqrt(squared_sum / float(hidden) + epsilon);
    for (uint column = 0; column < hidden; ++column) { output[row * hidden + column] = input[row * hidden + column] * inverse_rms * weight[column]; }
}

kernel void matvec_f32(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint output_id [[thread_position_in_grid]]) {
    if (output_id < output_width) {
        float sum = 0.0f;
        for (uint column = 0; column < input_width; ++column) { sum += input[column] * weights[output_id * input_width + column]; }
        output[output_id] = sum;
    }
}

// GGUF block-32 quantizers.  The output layout is exactly block_q4_0
// (fp16 scale + 16 packed signed nibbles) or block_q8_0 (fp16 scale + 32 i8).
kernel void quantize_q4_0(
    device const float *input [[buffer(0)]], device uchar *output [[buffer(1)]],
    constant uint &blocks [[buffer(2)]], uint block_id [[thread_position_in_grid]]) {
    if (block_id >= blocks) return;
    float maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) maximum = max(maximum, abs(input[block_id * 32 + i]));
    float scale = maximum == 0.0f ? 0.0f : maximum / 7.0f;
    device half *scale_out = (device half *)(output + block_id * 18);
    *scale_out = half(scale);
    for (uint i = 0; i < 16; ++i) {
        int a = scale == 0.0f ? 0 : int(round(clamp(input[block_id * 32 + i * 2] / scale, -8.0f, 7.0f)));
        int b = scale == 0.0f ? 0 : int(round(clamp(input[block_id * 32 + i * 2 + 1] / scale, -8.0f, 7.0f)));
        output[block_id * 18 + 2 + i] = uchar((a + 8) | ((b + 8) << 4));
    }
}

kernel void quantize_q8_0(
    device const float *input [[buffer(0)]], device uchar *output [[buffer(1)]],
    constant uint &blocks [[buffer(2)]], uint block_id [[thread_position_in_grid]]) {
    if (block_id >= blocks) return;
    float maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) maximum = max(maximum, abs(input[block_id * 32 + i]));
    float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    device half *scale_out = (device half *)(output + block_id * 34);
    *scale_out = half(scale);
    for (uint i = 0; i < 32; ++i) output[block_id * 34 + 2 + i] = uchar(scale == 0.0f ? 0 : int(round(clamp(input[block_id * 32 + i] / scale, -128.0f, 127.0f))));
}

kernel void matvec_q4_0(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= output_width) return;
    float sum = 0.0f;
    uint blocks = input_width / 32;
    for (uint block = 0; block < blocks; ++block) {
        device const uchar *base = weights + (row * blocks + block) * 18;
        float scale = float(*(device const half *)base);
        for (uint i = 0; i < 32; ++i) { uchar nibble = (i & 1) ? base[2 + i / 2] >> 4 : base[2 + i / 2] & 15; sum += input[block * 32 + i] * float(int(nibble) - 8) * scale; }
    }
    output[row] = sum;
}

kernel void matvec_q8_0(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= output_width) return;
    float sum = 0.0f;
    uint blocks = input_width / 32;
    for (uint block = 0; block < blocks; ++block) { device const uchar *base = weights + (row * blocks + block) * 34; float scale = float(*(device const half *)base); for (uint i = 0; i < 32; ++i) { char q = (char)base[2 + i]; sum += input[block * 32 + i] * float(q) * scale; } }
    output[row] = sum;
}

kernel void embedding_lookup_q4_0(
    device const uchar *weights [[buffer(0)]], device const uint *token_ids [[buffer(1)]], device float *output [[buffer(2)]], constant uint &vocabulary [[buffer(3)]], constant uint &hidden [[buffer(4)]], constant uint &tokens [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= tokens * hidden) return; uint token = token_ids[id / hidden]; if (token >= vocabulary) return; uint column = id % hidden; uint block = column / 32; device const uchar *base = weights + (token * (hidden / 32) + block) * 18; float scale = float(*(device const half *)base); uchar nibble = (column & 1) ? base[2 + (column % 32) / 2] >> 4 : base[2 + (column % 32) / 2] & 15; output[id] = float(int(nibble) - 8) * scale;
}

kernel void embedding_lookup_q8_0(
    device const uchar *weights [[buffer(0)]], device const uint *token_ids [[buffer(1)]], device float *output [[buffer(2)]], constant uint &vocabulary [[buffer(3)]], constant uint &hidden [[buffer(4)]], constant uint &tokens [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= tokens * hidden) return; uint token = token_ids[id / hidden]; if (token >= vocabulary) return; uint column = id % hidden; uint block = column / 32; device const uchar *base = weights + (token * (hidden / 32) + block) * 34; float scale = float(*(device const half *)base); output[id] = float((char)base[2 + (column % 32)]) * scale;
}

// One-token decode projection.  The resident command path keeps input,
// weights, and output in Metal buffers; this kernel is deliberately separate
// from the correctness reference so it can be specialized per GPU family
// without changing that oracle.
kernel void matvec_tiled_f32(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint output_id [[thread_position_in_grid]]) {
    if (output_id < output_width) {
        float sum = 0.0f;
        uint column = 0;
        for (; column + 3 < input_width; column += 4) {
            sum += input[column] * weights[output_id * input_width + column];
            sum += input[column + 1] * weights[output_id * input_width + column + 1];
            sum += input[column + 2] * weights[output_id * input_width + column + 2];
            sum += input[column + 3] * weights[output_id * input_width + column + 3];
        }
        for (; column < input_width; ++column) {
            sum += input[column] * weights[output_id * input_width + column];
        }
        output[output_id] = sum;
    }
}

kernel void matmul_f32(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &input_width [[buffer(4)]], constant uint &output_width [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    if (id < rows * output_width) {
        uint row = id / output_width; uint output_column = id % output_width; float sum = 0.0f;
        for (uint column = 0; column < input_width; ++column) { sum += input[row * input_width + column] * weights[output_column * input_width + column]; }
        output[id] = sum;
    }
}

kernel void rope_f32(
    device const float *input [[buffer(0)]], device const float *cosine [[buffer(1)]],
    device const float *sine [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &hidden [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    uint pairs_per_row = hidden / 2; uint row = id / pairs_per_row; uint pair = id % pairs_per_row;
    uint base = row * hidden + pair * 2;
    float x0 = input[base]; float x1 = input[base + 1]; float c = cosine[pair]; float s = sine[pair];
    output[base] = x0 * c - x1 * s;
    output[base + 1] = x0 * s + x1 * c;
}

kernel void masked_softmax_f32(
    device const float *input [[buffer(0)]], device const float *mask [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &columns [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    float maximum = -INFINITY;
    for (uint column = 0; column < columns; ++column) { maximum = max(maximum, input[row * columns + column] + mask[row * columns + column]); }
    float sum = 0.0f;
    for (uint column = 0; column < columns; ++column) { float value = exp(input[row * columns + column] + mask[row * columns + column] - maximum); output[row * columns + column] = value; sum += value; }
    for (uint column = 0; column < columns; ++column) { output[row * columns + column] /= sum; }
}

kernel void attention_scores_f32(
    device const float *queries [[buffer(0)]], device const float *keys [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &key_count [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]], constant float &scale [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    uint query = id / key_count; uint key = id % key_count; float sum = 0.0f;
    for (uint dimension = 0; dimension < head_dim; ++dimension) { sum += queries[query * head_dim + dimension] * keys[key * head_dim + dimension]; }
    output[id] = sum * scale;
}

kernel void attention_values_f32(
    device const float *weights [[buffer(0)]], device const float *values [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &key_count [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    uint query = id / head_dim; uint dimension = id % head_dim; float sum = 0.0f;
    for (uint key = 0; key < key_count; ++key) { sum += weights[query * key_count + key] * values[key * head_dim + dimension]; }
    output[id] = sum;
}

kernel void logits_process_f32(
    device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]],
    device float *output [[buffer(2)]], constant float &temperature [[buffer(3)]],
    constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if (id < count) { output[id] = (logits[id] + bias[id]) / temperature; }
}

kernel void rope_llama_decode_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &heads [[buffer(2)]], constant uint &head_dim [[buffer(3)]],
    device const float *cosine [[buffer(4)]], device const float *sine [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    uint pairs = head_dim / 2;
    uint head = id / pairs;
    uint pair = id % pairs;
    if (head < heads && pair < pairs) {
        uint base = head * head_dim;
        float c = cosine[pair], s = sine[pair];
        float x0 = input[base + pair], x1 = input[base + pair + pairs];
        output[base + pair] = x0 * c - x1 * s;
        output[base + pair + pairs] = x0 * s + x1 * c;
    }
}

kernel void rope_half_to_interleaved_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &heads [[buffer(2)]], constant uint &head_dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    uint pairs = head_dim / 2, head = id / pairs, pair = id % pairs;
    if (head < heads && pair < pairs) {
        uint base = head * head_dim;
        output[base + pair * 2] = input[base + pair];
        output[base + pair * 2 + 1] = input[base + pair + pairs];
    }
}

kernel void rope_interleaved_to_half_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &heads [[buffer(2)]], constant uint &head_dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    uint pairs = head_dim / 2, head = id / pairs, pair = id % pairs;
    if (head < heads && pair < pairs) {
        uint base = head * head_dim;
        output[base + pair] = input[base + pair * 2];
        output[base + pair + pairs] = input[base + pair * 2 + 1];
    }
}

// KV layout: [K|V][position][kv_head][dimension].
kernel void kv_append_decode_f32(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device float *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], constant uint &position [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    if (id < kv_width && position < capacity) {
        cache[position * kv_width + id] = key[id];
        cache[capacity * kv_width + position * kv_width + id] = value[id];
    }
}

// One thread produces one attention output dimension.  This favors resident
// single-token correctness and eliminates CPU head gathering/readbacks.
kernel void attention_decode_f32(
    device const float *query [[buffer(0)]], device const float *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &position [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    uint head = id / head_dim, dim = id % head_dim;
    if (head >= heads || dim >= head_dim) return;
    uint group = heads / kv_heads, kv_head = head / group;
    float maximum = -INFINITY;
    for (uint pos = 0; pos <= position; ++pos) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d)
            score += query[head * head_dim + d] * cache[pos * kv_heads * head_dim + kv_head * head_dim + d];
        maximum = max(maximum, score * rsqrt(float(head_dim)));
    }
    float denominator = 0.0f, value = 0.0f;
    uint value_base = capacity * kv_heads * head_dim;
    for (uint pos = 0; pos <= position; ++pos) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d)
            score += query[head * head_dim + d] * cache[pos * kv_heads * head_dim + kv_head * head_dim + d];
        float weight = exp(score * rsqrt(float(head_dim)) - maximum);
        denominator += weight;
        value += weight * cache[value_base + pos * kv_heads * head_dim + kv_head * head_dim + dim];
    }
    output[id] = value / denominator;
}

// Resident-layout equivalents of the reference score -> softmax -> value
// pipeline. Persistent intermediate buffers intentionally preserve the same
// FP32 rounding boundaries as the reference kernels.
kernel void attention_scores_resident_f32(
    device const float *query [[buffer(0)]], device const float *cache [[buffer(1)]],
    device float *scores [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    uint head = id / key_count, position = id % key_count;
    if (head >= heads || position >= key_count) return;
    uint kv_head = head / (heads / kv_heads), base = position * kv_heads * head_dim + kv_head * head_dim;
    float sum = 0.0f;
    for (uint d = 0; d < head_dim; ++d) sum += query[head * head_dim + d] * cache[base + d];
    scores[head * capacity + position] = sum * rsqrt(float(head_dim));
}

kernel void masked_softmax_resident_f32(
    device const float *scores [[buffer(0)]], device float *weights [[buffer(1)]],
    constant uint &heads [[buffer(2)]], constant uint &capacity [[buffer(3)]],
    constant uint &key_count [[buffer(4)]], uint head [[thread_position_in_grid]]) {
    if (head >= heads) return;
    uint base = head * capacity;
    float maximum = -INFINITY;
    for (uint key = 0; key < key_count; ++key) maximum = max(maximum, scores[base + key]);
    float sum = 0.0f;
    for (uint key = 0; key < key_count; ++key) { float value = exp(scores[base + key] - maximum); weights[base + key] = value; sum += value; }
    for (uint key = 0; key < key_count; ++key) weights[base + key] /= sum;
}

kernel void attention_values_resident_f32(
    device const float *weights [[buffer(0)]], device const float *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    uint head = id / head_dim, dim = id % head_dim;
    if (head >= heads || dim >= head_dim) return;
    uint kv_head = head / (heads / kv_heads), value_base = capacity * kv_heads * head_dim;
    float sum = 0.0f;
    for (uint key = 0; key < key_count; ++key)
        sum += weights[head * capacity + key] * cache[value_base + key * kv_heads * head_dim + kv_head * head_dim + dim];
    output[id] = sum;
}

kernel void argmax_f32(
    device const float *values [[buffer(0)]], device uint *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id == 0) {
        uint best = 0;
        for (uint i = 1; i < count; ++i) if (values[i] >= values[best]) best = i;
        output[0] = best;
    }
}
