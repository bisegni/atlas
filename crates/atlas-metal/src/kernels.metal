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
