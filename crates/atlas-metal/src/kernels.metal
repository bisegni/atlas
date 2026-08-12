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

kernel void vector_multiply_offset_f32(
    device const float *lhs [[buffer(0)]], device const float *rhs [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rhs_offset [[buffer(3)]],
    constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = lhs[id] * rhs[rhs_offset + id];
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

// Batched prefill counterpart to rms_norm_decode_f32.  Each prompt row owns
// one SIMD group, preserving the known-good decode reduction and per-lane
// arithmetic order while still normalizing all rows in one dispatch.
kernel void rms_norm_decode_batch_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    constant float &epsilon [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint base = row * hidden;
    float squared_sum = 0.0f;
    for (uint column = lane; column < hidden; column += 32) {
        float value = input[base + column];
        squared_sum += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(squared_sum) / float(hidden) + epsilon);
    for (uint column = lane; column < hidden; column += 32)
        output[base + column] = input[base + column] * inverse_rms * weight[column];
}

// Decode normalizes one hidden-state row at a time.  A single scalar thread
// made this a serial bubble between resident projections; keep the same FP32
// reduction/order per lane but spread the row across one Apple SIMD-group.
kernel void rms_norm_decode_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    constant float &epsilon [[buffer(4)]], uint lane [[thread_index_in_threadgroup]]) {
    float squared_sum = 0.0f;
    for (uint column = lane; column < hidden; column += 32) {
        float x = input[column];
        squared_sum += x * x;
    }
    float inverse_rms = rsqrt(simd_sum(squared_sum) / float(hidden) + epsilon);
    for (uint column = lane; column < hidden; column += 32) {
        output[column] = input[column] * inverse_rms * weight[column];
    }
}

// Gemma's decode hidden width is 2304, so each of the 32 lanes can process
// eighteen aligned float4 values. This is the production Resident decode
// path: it retains the one-SIMD-group reduction while reducing scalar
// load/store instructions.
kernel void rms_norm_decode_f32_vec4(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    constant float &epsilon [[buffer(4)]], uint lane [[thread_index_in_threadgroup]]) {
    float squared_sum = 0.0f;
    uint vector_tiles = hidden / 128;
    for (uint tile = 0; tile < vector_tiles; ++tile) {
        uint offset = tile * 128 + lane * 4;
        float4 x = *(device const float4 *)(input + offset);
        squared_sum += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    }
    float inverse_rms = rsqrt(simd_sum(squared_sum) / float(hidden) + epsilon);
    for (uint tile = 0; tile < vector_tiles; ++tile) {
        uint offset = tile * 128 + lane * 4;
        float4 x = *(device const float4 *)(input + offset);
        float4 w = *(device const float4 *)(weight + offset);
        *(device float4 *)(output + offset) = x * inverse_rms * w;
    }
}

// Experimental Gemma decode epilogue.  This retains the 32-lane reduction
// and explicit intermediate arithmetic of rms_norm_decode_f32_vec4 (the
// production decode norm), then writes the following residual in the same
// dispatch.  It is intentionally opt-in: its value is fewer dispatch
// boundaries, not a changed numerical contract.  Per-element values are
// bitwise identical to rms_norm_decode_f32_vec4 followed by vector_add_f32.
kernel void gemma4_rms_residual_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    // Keep the normalized vector as a volatile device-memory intermediate.
    // The baseline writes this value in rms_norm_decode_f32_vec4, then
    // reloads it in vector_add_f32; retaining that rounding boundary is
    // required for greedy-token parity.
    device volatile float *normalized [[buffer(3)]], device float *output [[buffer(4)]],
    constant uint &hidden [[buffer(5)]], constant float &epsilon [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint vector_tiles = hidden / 128;
    float squared_sum = 0.0f;
    for (uint tile = 0; tile < vector_tiles; ++tile) {
        uint offset = tile * 128 + lane * 4;
        float4 x = *(device const float4 *)(input + offset);
        squared_sum += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    }
    float inverse_rms = rsqrt(simd_sum(squared_sum) / float(hidden) + epsilon);
    for (uint tile = 0; tile < vector_tiles; ++tile) {
        uint offset = tile * 128 + lane * 4;
        float4 x = *(device const float4 *)(input + offset);
        float4 w = *(device const float4 *)(weight + offset);
        *(device float4 *)(normalized + offset) = x * inverse_rms * w;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint tile = 0; tile < vector_tiles; ++tile) {
        uint offset = tile * 128 + lane * 4;
        float4 n = *(device const float4 *)(normalized + offset);
        float4 r = *(device const float4 *)(residual + offset);
        *(device float4 *)(output + offset) = r + n;
    }
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
        for (uint i = 0; i < 32; ++i) { uchar packed = base[2 + (i & 15)]; uchar nibble = i < 16 ? packed & 15 : packed >> 4; sum += input[block * 32 + i] * float(int(nibble) - 8) * scale; }
    }
    output[row] = sum;
}

// One SIMD-group owns one output row.  Each lane consumes the same lane of
// every Q4 block, so packed GGUF weights stay resident and adjacent lanes read
// adjacent nibbles instead of one scalar thread walking the entire row.
kernel void matvec_q4_0_blocked(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= output_width) return;
    float sum = 0.0f;
    uint blocks = input_width / 32;
    for (uint block = 0; block < blocks; ++block) {
        device const uchar *base = weights + (row * blocks + block) * 18;
        uchar packed = base[2 + (lane & 15)];
        uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
        sum += input[block * 32 + lane] * float(int(nibble) - 8) * float(*(device const half *)base);
    }
    float total = simd_sum(sum);
    if (lane == 0) output[row] = total;
}










inline float atlas_tanh_f32(float value);







// Batched prefill building block.  Inputs and outputs are row-major
// [batch, width] matrices; weights remain in their GGUF Q4_0 packing and each
// output dot product still accumulates in FP32.  The executor can therefore
// move a whole prompt layer at a time without creating a dequantized cache.
kernel void matmul_q4_0_batch_16row(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], constant uint &batch [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint rows_per_batch = (output_width + 15) / 16;
    uint token = group / rows_per_batch;
    uint group_row = group % rows_per_batch;
    uint row = group_row * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (token < batch && row < output_width) {
        uint blocks = input_width / 32;
        device const float *token_input = input + token * input_width;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += token_input[block * 32 + column] * float(int(packed0 & 15) - 8) * scale;
            sum += token_input[block * 32 + column + 8] * float(int(packed1 & 15) - 8) * scale;
            sum += token_input[block * 32 + column + 16] * float(int(packed0 >> 4) - 8) * scale;
            sum += token_input[block * 32 + column + 24] * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && token < batch && row < output_width)
        output[token * output_width + row] = sum;
}





kernel void matmul_f16_batch(
    device const float *input [[buffer(0)]], device const half *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], constant uint &batch [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    uint token = id / output_width;
    uint row = id % output_width;
    if (token >= batch || row >= output_width) return;
    float sum = 0.0f;
    device const float *token_input = input + token * input_width;
    for (uint column = 0; column < input_width; ++column)
        sum += token_input[column] * float(weights[row * input_width + column]);
    output[token * output_width + row] = sum;
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
    if (id >= tokens * hidden) return; uint token = token_ids[id / hidden]; if (token >= vocabulary) return; uint column = id % hidden; uint block = column / 32; uint within = column % 32; device const uchar *base = weights + (token * (hidden / 32) + block) * 18; float scale = float(*(device const half *)base); uchar packed = base[2 + (within & 15)]; uchar nibble = within < 16 ? packed & 15 : packed >> 4; output[id] = float(int(nibble) - 8) * scale;
}

kernel void embedding_lookup_q8_0(
    device const uchar *weights [[buffer(0)]], device const uint *token_ids [[buffer(1)]], device float *output [[buffer(2)]], constant uint &vocabulary [[buffer(3)]], constant uint &hidden [[buffer(4)]], constant uint &tokens [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= tokens * hidden) return; uint token = token_ids[id / hidden]; if (token >= vocabulary) return; uint column = id % hidden; uint block = column / 32; device const uchar *base = weights + (token * (hidden / 32) + block) * 34; float scale = float(*(device const half *)base); output[id] = float((char)base[2 + (column % 32)]) * scale;
}

// GGML block_q6_K: 128 low nibbles, 64 packed high-bit pairs, sixteen signed
// per-group scales, then the f16 block scale. Gemma 4 E2B's two embedding
// tables use this format while its projections remain Q4_0.
inline float q6_k_value(device const uchar *base, uint index) {
    uint chunk = index / 128;
    uint within = index % 128;
    uint stream = within / 32;
    uint lane = within % 32;
    uchar packed = base[chunk * 64 + lane + ((stream & 1) ? 32 : 0)];
    uchar low = stream >= 2 ? packed >> 4 : packed & 15;
    uchar high = (base[128 + chunk * 32 + lane] >> (stream * 2)) & 3;
    int group_scale = int((char) base[192 + chunk * 8 + lane / 16 + stream * 2]);
    int quantized = int((high << 4) | low) - 32;
    return float(quantized * group_scale) * float(*(device const half *)(base + 208));
}

kernel void embedding_lookup_q6_k(
    device const uchar *weights [[buffer(0)]], device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &vocabulary [[buffer(3)]],
    constant uint &hidden [[buffer(4)]], constant uint &tokens [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= tokens * hidden) return;
    uint token = token_ids[id / hidden];
    if (token >= vocabulary) return;
    uint column = id % hidden;
    uint block = column / 256;
    uint index = column % 256;
    device const uchar *base = weights + (token * (hidden / 256) + block) * 210;
    output[id] = q6_k_value(base, index);
}




// Q6_K counterpart of the batched Q4 prefill projection.  It is principally
// used for Gemma's tied vocabulary projection when a batched diagnostic needs
// logits, and deliberately shares q6_k_value with the decode kernel.
kernel void matmul_q6_k_batch_8row(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], constant uint &batch [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 16;
    uint column = lane % 16;
    uint rows_per_batch = (output_width + 7) / 8;
    uint token = group / rows_per_batch;
    uint group_row = group % rows_per_batch;
    uint row = group_row * 8 + simdgroup * 2 + row_in_simd;
    float sum = 0.0f;
    if (token < batch && row < output_width) {
        uint blocks = input_width / 256;
        device const float *token_input = input + token * input_width;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + (row * blocks + block) * 210;
            for (uint index = column; index < 256; index += 16)
                sum += token_input[block * 256 + index] * q6_k_value(base, index);
        }
    }
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && token < batch && row < output_width)
        output[token * output_width + row] = sum;
}

kernel void matvec_f16(
    device const float *input [[buffer(0)]], device const half *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= output_width) return;
    float sum = 0.0f;
    for (uint column = 0; column < input_width; ++column)
        sum += input[column] * float(weights[row * input_width + column]);
    output[row] = sum;
}

inline float atlas_tanh_f32(float value) {
    // Metal's generic tanh path produced NaN for ordinary finite scalar input
    // on the resident device.  Saturate outside the range where tanh differs
    // from +/-1 at FP32 precision, then use its stable exponential identity.
    if (value >= 10.0f) return 1.0f;
    if (value <= -10.0f) return -1.0f;
    float exponent = exp(2.0f * value);
    return (exponent - 1.0f) / (exponent + 1.0f);
}

kernel void gelu_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        float x = input[id];
        // The tanh GELU polynomial has the correct limits x and 0.  Preserve
        // those limits explicitly when the cubic intermediate overflows so a
        // finite large negative activation cannot become (-finite * 0) NaN.
        float argument = 0.7978845608f * (x + 0.044715f * x * x * x);
        if (isinf(argument)) {
            output[id] = argument > 0.0f ? x : 0.0f;
        } else {
            output[id] = 0.5f * x * (1.0f + atlas_tanh_f32(argument));
        }
    }
}

// Decode-only FFN activation.  Replaces the gelu_f32 + vector_multiply_f32
// pair with one elementwise pass computing gelu(gate) * up; the operations
// are exactly the reference's GELU followed by its multiply, so no
// reassociation is possible and the result is bitwise identical.
kernel void gelu_multiply_f32(
    device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) {
        float x = gate[id];
        float argument = 0.7978845608f * (x + 0.044715f * x * x * x);
        float gelu = isinf(argument) ? (argument > 0.0f ? x : 0.0f)
                                      : 0.5f * x * (1.0f + atlas_tanh_f32(argument));
        output[id] = gelu * up[id];
    }
}

// Decode-only PLE composition.  The baseline writes GELU(gate) back to the
// gate buffer, then launches vector_multiply_offset_f32 to select the current
// layer's PLE slice.  This candidate keeps the same per-element GELU and
// multiply operations but removes that intermediate write/read pair.
kernel void ple_gelu_multiply_offset_f32(
    device const float *gate [[buffer(0)]], device const float *ple [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &ple_offset [[buffer(3)]],
    constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        float x = gate[id];
        float argument = 0.7978845608f * (x + 0.044715f * x * x * x);
        float gelu = isinf(argument) ? (argument > 0.0f ? x : 0.0f)
                                      : 0.5f * x * (1.0f + atlas_tanh_f32(argument));
        output[id] = gelu * ple[ple_offset + id];
    }
}



kernel void copy_u32(
    device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = input[id];
}



// Gemma's resident buffers intentionally reuse storage between stages.  A
// parallel elementwise RMS implementation races when input and output alias,
// so this decode-oriented variant owns an entire group in one thread.
kernel void rms_norm_groups_in_place_f32(
    device float *values [[buffer(0)]], device const float *weight [[buffer(1)]],
    constant uint &width [[buffer(2)]], constant uint &groups [[buffer(3)]],
    constant float &epsilon [[buffer(4)]], uint group [[thread_position_in_grid]]) {
    if (group >= groups) return;
    uint base = group * width;
    float squared_sum = 0.0f;
    for (uint index = 0; index < width; ++index) squared_sum += values[base + index] * values[base + index];
    float inv_rms = rsqrt(squared_sum / float(width) + epsilon);
    for (uint index = 0; index < width; ++index) values[base + index] = values[base + index] * inv_rms * weight[index];
}

// PLE projection can inherit the QAT embedding's very large dynamic range.
// Scale the reduction by each group's finite maximum to avoid x*x overflow
// while preserving RMSNorm's mathematical result.
kernel void rms_norm_groups_in_place_stable_f32(
    device float *values [[buffer(0)]], device const float *weight [[buffer(1)]],
    constant uint &width [[buffer(2)]], constant uint &groups [[buffer(3)]],
    constant float &epsilon [[buffer(4)]], uint group [[thread_position_in_grid]]) {
    if (group >= groups) return;
    uint base = group * width;
    float maximum = 0.0f;
    for (uint index = 0; index < width; ++index) maximum = max(maximum, abs(values[base + index]));
    if (maximum == 0.0f) {
        for (uint index = 0; index < width; ++index) values[base + index] = 0.0f;
        return;
    }
    float squared_sum = 0.0f;
    for (uint index = 0; index < width; ++index) {
        float scaled = values[base + index] / maximum;
        squared_sum += scaled * scaled;
    }
    float inverse_rms = rsqrt(squared_sum / float(width) + epsilon / (maximum * maximum)) / maximum;
    for (uint index = 0; index < width; ++index)
        values[base + index] = values[base + index] * inverse_rms * weight[index];
}

kernel void rms_norm_groups_in_place_unweighted_f32(
    device float *values [[buffer(0)]], constant uint &width [[buffer(1)]],
    constant uint &groups [[buffer(2)]], constant float &epsilon [[buffer(3)]],
    uint group [[thread_position_in_grid]]) {
    if (group >= groups) return;
    uint base = group * width;
    float squared_sum = 0.0f;
    for (uint index = 0; index < width; ++index) squared_sum += values[base + index] * values[base + index];
    float inv_rms = rsqrt(squared_sum / float(width) + epsilon);
    for (uint index = 0; index < width; ++index) values[base + index] *= inv_rms;
}

kernel void softcap_f32(
    device float *values [[buffer(0)]], constant float &cap [[buffer(1)]], constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) values[id] = cap * tanh(values[id] / cap);
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

// Gemma applies weighted RMS normalization to every Q/K head immediately
// before RoPE.  Decode previously materialized an interleaved scratch layout
// and used three dispatches per tensor (normalize, convert, rotate, convert
// back).  Q and a provider K are independent, so map the threadgroup range
// over Q heads followed by the optional one K head and write their half-split
// RoPE result directly.  Each group deliberately keeps the scalar reduction
// order of rms_norm_groups_in_place_f32 for greedy-token parity.
kernel void gemma4_qk_norm_rope_fused_f32(
    device const float *q_input [[buffer(0)]],
    device const float *k_input [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    device const float *cosine [[buffer(4)]],
    device const float *sine [[buffer(5)]],
    device float *q_output [[buffer(6)]],
    device float *k_output [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &q_heads [[buffer(9)]],
    constant uint &has_key [[buffer(10)]],
    constant float &epsilon [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0 || group >= q_heads + has_key) return;
    bool key = group >= q_heads;
    uint head = key ? 0 : group;
    device const float *input = key ? k_input : q_input;
    device const float *weight = key ? k_weight : q_weight;
    device float *output = key ? k_output : q_output;
    uint base = head * head_dim;
    float squared_sum = 0.0f;
    for (uint index = 0; index < head_dim; ++index) {
        float value = input[base + index];
        squared_sum += value * value;
    }
    float inv_rms = rsqrt(squared_sum / float(head_dim) + epsilon);
    uint pairs = head_dim / 2;
    for (uint pair = 0; pair < pairs; ++pair) {
        float x0 = input[base + pair] * inv_rms * weight[pair];
        float x1 = input[base + pair + pairs] * inv_rms * weight[pair + pairs];
        float c = cosine[pair];
        float s = sine[pair];
        output[base + pair] = x0 * c - x1 * s;
        output[base + pair + pairs] = x0 * s + x1 * c;
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

// Packed KV layouts are [K blocks | V blocks][position][block].  They are
// intentionally separate from model-weight packing: K/V are transient FP32
// activations, quantized only at their cache boundary.
kernel void kv_append_decode_q8_0(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device uchar *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], constant uint &position [[buffer(5)]],
    uint block [[thread_position_in_grid]]) {
    uint blocks = kv_width / 32;
    if (block >= blocks || position >= capacity) return;
    uint base = block * 32;
    float maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) maximum = max(maximum, abs(key[base + i]));
    float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    device uchar *out = cache + (position * blocks + block) * 34;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 32; ++i)
        out[2 + i] = uchar(char(scale == 0.0f ? 0 : int(round(clamp(key[base + i] / scale, -127.0f, 127.0f)))));
    maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) maximum = max(maximum, abs(value[base + i]));
    scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    out = cache + (capacity * blocks + position * blocks + block) * 34;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 32; ++i)
        out[2 + i] = uchar(char(scale == 0.0f ? 0 : int(round(clamp(value[base + i] / scale, -127.0f, 127.0f)))));
}

kernel void kv_append_decode_q4_0(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device uchar *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], constant uint &position [[buffer(5)]],
    uint block [[thread_position_in_grid]]) {
    uint blocks = kv_width / 32;
    if (block >= blocks || position >= capacity) return;
    uint base = block * 32;
    float maximum = 0.0f, signed_maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) if (abs(key[base + i]) > maximum) { maximum = abs(key[base + i]); signed_maximum = key[base + i]; }
    float scale = maximum == 0.0f ? 0.0f : signed_maximum / -8.0f;
    device uchar *out = cache + (position * blocks + block) * 18;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 16; ++i) {
        int low = scale == 0.0f ? 0 : clamp(int(key[base + i] / scale + 8.5f), 0, 15);
        int high = scale == 0.0f ? 0 : clamp(int(key[base + i + 16] / scale + 8.5f), 0, 15);
        out[2 + i] = uchar(low | (high << 4));
    }
    maximum = 0.0f; signed_maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) if (abs(value[base + i]) > maximum) { maximum = abs(value[base + i]); signed_maximum = value[base + i]; }
    scale = maximum == 0.0f ? 0.0f : signed_maximum / -8.0f;
    out = cache + (capacity * blocks + position * blocks + block) * 18;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 16; ++i) {
        int low = scale == 0.0f ? 0 : clamp(int(value[base + i] / scale + 8.5f), 0, 15);
        int high = scale == 0.0f ? 0 : clamp(int(value[base + i + 16] / scale + 8.5f), 0, 15);
        out[2 + i] = uchar(low | (high << 4));
    }
}

// Decode-only KV append with the provider V group RMS folded in.  The
// unfused path launches rms_norm_groups_in_place_unweighted_f32 over the
// whole V vector (one group), then kv_append_decode_* over the 32-wide
// blocks.  This candidate removes that dispatch boundary: every block thread
// redundantly computes the single group's sumsq in the reference's exact
// sequential order, so inv_rms and the quantized V blocks are bitwise
// identical while the raw V buffer is left untouched.
kernel void kv_append_decode_f32_vnorm(
    device const float *key [[buffer(0)]], device float *value [[buffer(1)]],
    device float *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], constant uint &position [[buffer(5)]],
    constant float &epsilon [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    if (id < kv_width && position < capacity) {
        cache[position * kv_width + id] = key[id];
    }
    if (id == 0) {
        float squared_sum = 0.0f;
        for (uint index = 0; index < kv_width; ++index)
            squared_sum += value[index] * value[index];
        float inverse_rms = rsqrt(squared_sum / float(kv_width) + epsilon);
        for (uint index = 0; index < kv_width; ++index)
            cache[capacity * kv_width + position * kv_width + index] = value[index] * inverse_rms;
    }
}

kernel void kv_append_decode_q8_0_vnorm(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device uchar *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], constant uint &position [[buffer(5)]],
    constant float &epsilon [[buffer(6)]], uint block [[thread_position_in_grid]]) {
    uint blocks = kv_width / 32;
    if (block >= blocks || position >= capacity) return;
    uint base = block * 32;
    float squared_sum = 0.0f;
    for (uint index = 0; index < kv_width; ++index)
        squared_sum += value[index] * value[index];
    float inverse_rms = rsqrt(squared_sum / float(kv_width) + epsilon);
    float maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) maximum = max(maximum, abs(key[base + i]));
    float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    device uchar *out = cache + (position * blocks + block) * 34;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 32; ++i)
        out[2 + i] = uchar(char(scale == 0.0f ? 0 : int(round(clamp(key[base + i] / scale, -127.0f, 127.0f)))));
    maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) maximum = max(maximum, abs(value[base + i] * inverse_rms));
    scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    out = cache + (capacity * blocks + position * blocks + block) * 34;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 32; ++i)
        out[2 + i] = uchar(char(scale == 0.0f ? 0 : int(round(clamp(value[base + i] * inverse_rms / scale, -127.0f, 127.0f)))));
}

kernel void kv_append_decode_q4_0_vnorm(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device uchar *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], constant uint &position [[buffer(5)]],
    constant float &epsilon [[buffer(6)]], uint block [[thread_position_in_grid]]) {
    uint blocks = kv_width / 32;
    if (block >= blocks || position >= capacity) return;
    uint base = block * 32;
    float squared_sum = 0.0f;
    for (uint index = 0; index < kv_width; ++index)
        squared_sum += value[index] * value[index];
    float inverse_rms = rsqrt(squared_sum / float(kv_width) + epsilon);
    float maximum = 0.0f, signed_maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) if (abs(key[base + i]) > maximum) { maximum = abs(key[base + i]); signed_maximum = key[base + i]; }
    float scale = maximum == 0.0f ? 0.0f : signed_maximum / -8.0f;
    device uchar *out = cache + (position * blocks + block) * 18;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 16; ++i) {
        int low = scale == 0.0f ? 0 : clamp(int(key[base + i] / scale + 8.5f), 0, 15);
        int high = scale == 0.0f ? 0 : clamp(int(key[base + i + 16] / scale + 8.5f), 0, 15);
        out[2 + i] = uchar(low | (high << 4));
    }
    maximum = 0.0f; signed_maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) {
        float normalized = value[base + i] * inverse_rms;
        if (abs(normalized) > maximum) { maximum = abs(normalized); signed_maximum = normalized; }
    }
    scale = maximum == 0.0f ? 0.0f : signed_maximum / -8.0f;
    out = cache + (capacity * blocks + position * blocks + block) * 18;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 16; ++i) {
        float normalized_low = value[base + i] * inverse_rms;
        float normalized_high = value[base + i + 16] * inverse_rms;
        int low = scale == 0.0f ? 0 : clamp(int(normalized_low / scale + 8.5f), 0, 15);
        int high = scale == 0.0f ? 0 : clamp(int(normalized_high / scale + 8.5f), 0, 15);
        out[2 + i] = uchar(low | (high << 4));
    }
}

inline float kv_q8_0_value(device const uchar *cache, uint index) {
    return float(*(device const half *)cache) * float((char)cache[2 + index]);
}

inline float kv_q4_0_value(device const uchar *cache, uint index) {
    uchar packed = cache[2 + (index & 15)];
    uchar nibble = index < 16 ? packed & 15 : packed >> 4;
    return float(*(device const half *)cache) * float(int(nibble) - 8);
}


// One workgroup owns one query head.  It streams keys in order and maintains
// the stable online-softmax state, avoiding resident score/weight buffers and
// their three dependent dispatches.  The fixed 128-thread launch is supplied
// by the resident encoder; each thread may own multiple output dimensions.
kernel void attention_decode_fused_f32(
    device const float *query [[buffer(0)]], device const float *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    if (head >= heads) return;
    uint kv_head = head / (heads / kv_heads);
    uint value_base = capacity * kv_heads * head_dim;
    threadgroup float reductions[128];
    threadgroup float maximum;
    threadgroup float denominator;
    threadgroup float rescale;
    threadgroup float weight;
    maximum = -INFINITY;
    denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = 0; key < key_count; ++key) {
        float partial = 0.0f;
        uint key_base = key * kv_heads * head_dim + kv_head * head_dim;
        for (uint d = tid; d < head_dim; d += threads)
            partial += query[head * head_dim + d] * cache[key_base + d];
        reductions[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reductions[tid] += reductions[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            float score = reductions[0] * rsqrt(float(head_dim));
            if (score > maximum) {
                rescale = exp(maximum - score);
                weight = 1.0f;
                maximum = score;
                denominator = denominator * rescale + weight;
            } else {
                rescale = 1.0f;
                weight = exp(score - maximum);
                denominator += weight;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint value_offset = value_base + key_base;
        for (uint d = tid; d < head_dim; d += threads)
            output[head * head_dim + d] = output[head * head_dim + d] * rescale
                + weight * cache[value_offset + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = tid; d < head_dim; d += threads)
        output[head * head_dim + d] /= denominator;
}


// Same online-softmax and KV layout as the baseline Gemma fused kernel. This
// keeps dot-product partials in SIMD-group registers, then combines the four
// group totals, avoiding the repeated 128-lane shared-memory reduction.
kernel void attention_decode_fused_gemma4_simd_f32(
    device const float *query [[buffer(0)]], device const float *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_control [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (head >= heads) return;
    uint key_start = key_control >> 16;
    uint key_count = key_control & 0xffffu;
    uint kv_head = head / (heads / kv_heads);
    uint value_base = capacity * kv_heads * head_dim;
    threadgroup float simd_sums[4];
    threadgroup float maximum;
    threadgroup float denominator;
    threadgroup float rescale;
    threadgroup float weight;
    maximum = -INFINITY;
    denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key_offset = 0; key_offset < key_count; ++key_offset) {
        uint key = key_start + key_offset;
        float partial = 0.0f;
        uint key_base = key * kv_heads * head_dim + kv_head * head_dim;
        for (uint d = tid; d < head_dim; d += threads)
            partial += query[head * head_dim + d] * cache[key_base + d];
        float simd_total = simd_sum(partial);
        if (lane == 0) simd_sums[simd_group] = simd_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float score = simd_sums[0] + simd_sums[1] + simd_sums[2] + simd_sums[3];
            if (score > maximum) {
                rescale = exp(maximum - score);
                weight = 1.0f;
                maximum = score;
                denominator = denominator * rescale + weight;
            } else {
                rescale = 1.0f;
                weight = exp(score - maximum);
                denominator += weight;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint value_offset = value_base + key_base;
        for (uint d = tid; d < head_dim; d += threads)
            output[head * head_dim + d] = output[head * head_dim + d] * rescale
                + weight * cache[value_offset + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = tid; d < head_dim; d += threads)
        output[head * head_dim + d] /= denominator;
}

// Q8/Q4 cache attention preserves the F32 query, online softmax, and output
// contract. Only cached K/V blocks are decoded on demand.
kernel void attention_decode_fused_gemma4_simd_q8_0(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_control [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (head >= heads) return;
    uint key_start = key_control >> 16;
    uint key_count = key_control & 0xffffu;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key_offset = 0; key_offset < key_count; ++key_offset) {
        uint key = key_start + key_offset;
        float partial = 0.0f;
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim;
        for (uint d = tid; d < head_dim; d += threads) {
            uint index = key_element + d;
            partial += query[head * head_dim + d] * kv_q8_0_value(cache + (index / 32) * 34, index % 32);
        }
        float simd_total = simd_sum(partial);
        if (lane == 0) simd_sums[simd_group] = simd_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float score = simd_sums[0] + simd_sums[1] + simd_sums[2] + simd_sums[3]; if (score > maximum) { rescale = exp(maximum - score); weight = 1.0f; maximum = score; denominator = denominator * rescale + weight; } else { rescale = 1.0f; weight = exp(score - maximum); denominator += weight; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < head_dim; d += threads) { uint index = key_element + d; output[head * head_dim + d] = output[head * head_dim + d] * rescale + weight * kv_q8_0_value(cache + (value_base + index / 32) * 34, index % 32); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] /= denominator;
}

kernel void attention_decode_fused_gemma4_simd_q4_0(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_control [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (head >= heads) return;
    uint key_start = key_control >> 16;
    uint key_count = key_control & 0xffffu;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key_offset = 0; key_offset < key_count; ++key_offset) {
        uint key = key_start + key_offset;
        float partial = 0.0f;
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim;
        for (uint d = tid; d < head_dim; d += threads) { uint index = key_element + d; partial += query[head * head_dim + d] * kv_q4_0_value(cache + (index / 32) * 18, index % 32); }
        float simd_total = simd_sum(partial);
        if (lane == 0) simd_sums[simd_group] = simd_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float score = simd_sums[0] + simd_sums[1] + simd_sums[2] + simd_sums[3]; if (score > maximum) { rescale = exp(maximum - score); weight = 1.0f; maximum = score; denominator = denominator * rescale + weight; } else { rescale = 1.0f; weight = exp(score - maximum); denominator += weight; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < head_dim; d += threads) { uint index = key_element + d; output[head * head_dim + d] = output[head * head_dim + d] * rescale + weight * kv_q4_0_value(cache + (value_base + index / 32) * 18, index % 32); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] /= denominator;
}

// Exact-compatible Flash16 replacement. The previous flash16_uw kernels
// partitioned keys into independently normalized slices and merged them. That
// is mathematically sound but changes FP32 rounding enough to alter Gemma's
// greedy stream. Keep the same four-SIMD reduction, runtime Q·K accumulation,
// and key-ordered online softmax sequence as the established Resident Q4
// kernel. In particular, do not make the head width compile-time constant:
// Metal may otherwise unroll or reassociate the partial dot product and alter
// a later greedy token despite mathematically equivalent attention.
#define FLASH16_VALUE_BARRIER threadgroup_barrier(mem_flags::mem_threadgroup);
#define FLASH16_NO_VALUE_BARRIER
#define DEFINE_FLASH16_EXACT(NAME, HEAD_DIM, VALUE_BARRIER) \
kernel void NAME( \
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], constant uint &key_control [[buffer(7)]], \
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint threads [[threads_per_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    if (head >= heads) return; \
    uint key_start = key_control >> 16; \
    uint key_count = key_control & 0xffffu; \
    uint kv_head = head / (heads / kv_heads); \
    uint blocks_per_position = (kv_heads * head_dim) / 32; \
    uint value_base = capacity * blocks_per_position; \
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight; \
    maximum = -INFINITY; denominator = 0.0f; \
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] = 0.0f; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint key_offset = 0; key_offset < key_count; ++key_offset) { \
        uint key = key_start + key_offset; \
        float partial = 0.0f; \
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim; \
        for (uint d = tid; d < head_dim; d += threads) { \
            uint index = key_element + d; \
            partial += query[head * head_dim + d] * kv_q4_0_value(cache + (index / 32) * 18, index % 32); \
        } \
        float simd_total = simd_sum(partial); \
        if (lane == 0) simd_sums[simd_group] = simd_total; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (tid == 0) { \
            float score = simd_sums[0] + simd_sums[1] + simd_sums[2] + simd_sums[3]; \
            if (score > maximum) { \
                rescale = exp(maximum - score); weight = 1.0f; maximum = score; denominator = denominator * rescale + weight; \
            } else { \
                rescale = 1.0f; weight = exp(score - maximum); denominator += weight; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint d = tid; d < head_dim; d += threads) { \
            uint index = key_element + d; \
            output[head * head_dim + d] = output[head * head_dim + d] * rescale \
                + weight * kv_q4_0_value(cache + (value_base + index / 32) * 18, index % 32); \
        } \
        VALUE_BARRIER \
    } \
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] /= denominator; \
}

DEFINE_FLASH16_EXACT(attention_decode_gemma4_simd_q4_0_flash16_exact, 512, FLASH16_VALUE_BARRIER)
DEFINE_FLASH16_EXACT(attention_decode_gemma4_simd_q4_0_flash16_swa_exact, 256, FLASH16_VALUE_BARRIER)
// Keep these separately named exports while validating the runtime-loop
// implementation. Generation metrics must identify the binary that ran.
DEFINE_FLASH16_EXACT(attention_decode_gemma4_simd_q4_0_flash16_exact_runtime, 512, FLASH16_VALUE_BARRIER)
DEFINE_FLASH16_EXACT(attention_decode_gemma4_simd_q4_0_flash16_swa_exact_runtime, 256, FLASH16_VALUE_BARRIER)
DEFINE_FLASH16_EXACT(attention_decode_gemma4_simd_q4_0_flash16_exact_nb, 512, FLASH16_NO_VALUE_BARRIER)
DEFINE_FLASH16_EXACT(attention_decode_gemma4_simd_q4_0_flash16_swa_exact_nb, 256, FLASH16_NO_VALUE_BARRIER)















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
    constant uint &count [[buffer(2)]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float candidates[256];
    threadgroup uint candidate_ids[256];
    float best_value = -INFINITY;
    uint best_id = 0;
    for (uint i = lane; i < count; i += 256) {
        float value = values[i];
        // Match the old greedy tie rule: the later (higher) token ID wins.
        if (value >= best_value) {
            best_value = value;
            best_id = i;
        }
    }
    candidates[lane] = best_value;
    candidate_ids[lane] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            float other_value = candidates[lane + stride];
            uint other_id = candidate_ids[lane + stride];
            if (other_value > candidates[lane]
                || (other_value == candidates[lane] && other_id >= candidate_ids[lane])) {
                candidates[lane] = other_value;
                candidate_ids[lane] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[0] = candidate_ids[0];
    }
}

// Single-dispatch flash attention over the packed-Q4 cache (llama.cpp
// flash-attention structure).  Multiple SIMD groups scan disjoint key slices
// with per-simdgroup register softmax state (running maximum, denominator,
// and value accumulators) and per-key online rescale, then merge the slice
// states inside the threadgroup.  There are no per-key threadgroup barriers,
// no partial/max/sum buffers, and no combine dispatch: the KV cache is read
// once and the normalized result is written directly to the attention
// output.  Specialized to the Gemma4 head geometries (512-wide full context
// with 8 slices, 256-wide sliding window with 16 slices).
#define FLASH_ACC_DECL(B) float acc##B = 0.0f;
#define FLASH_ACC_UPDATE(B) \
    else if (b == B) acc##B = acc##B * rescale + weight * value;
#define FLASH_ACC_STORE(HD, NB, B) \
    if (B < NB) merg_out[simd_group * HD + B * 32 + lane] = acc##B;
#define FLASH_ACC_DECLS \
    FLASH_ACC_DECL(0) FLASH_ACC_DECL(1) FLASH_ACC_DECL(2) FLASH_ACC_DECL(3) \
    FLASH_ACC_DECL(4) FLASH_ACC_DECL(5) FLASH_ACC_DECL(6) FLASH_ACC_DECL(7) \
    FLASH_ACC_DECL(8) FLASH_ACC_DECL(9) FLASH_ACC_DECL(10) FLASH_ACC_DECL(11) \
    FLASH_ACC_DECL(12) FLASH_ACC_DECL(13) FLASH_ACC_DECL(14) FLASH_ACC_DECL(15)
#define FLASH_ACC_UPDATES \
    FLASH_ACC_UPDATE(1) FLASH_ACC_UPDATE(2) FLASH_ACC_UPDATE(3) \
    FLASH_ACC_UPDATE(4) FLASH_ACC_UPDATE(5) FLASH_ACC_UPDATE(6) \
    FLASH_ACC_UPDATE(7) FLASH_ACC_UPDATE(8) FLASH_ACC_UPDATE(9) \
    FLASH_ACC_UPDATE(10) FLASH_ACC_UPDATE(11) FLASH_ACC_UPDATE(12) \
    FLASH_ACC_UPDATE(13) FLASH_ACC_UPDATE(14) FLASH_ACC_UPDATE(15)
#define FLASH_ACC_STORES(HD, NB) \
    FLASH_ACC_STORE(HD, NB, 0) FLASH_ACC_STORE(HD, NB, 1) FLASH_ACC_STORE(HD, NB, 2) \
    FLASH_ACC_STORE(HD, NB, 3) FLASH_ACC_STORE(HD, NB, 4) FLASH_ACC_STORE(HD, NB, 5) \
    FLASH_ACC_STORE(HD, NB, 6) FLASH_ACC_STORE(HD, NB, 7) FLASH_ACC_STORE(HD, NB, 8) \
    FLASH_ACC_STORE(HD, NB, 9) FLASH_ACC_STORE(HD, NB, 10) FLASH_ACC_STORE(HD, NB, 11) \
    FLASH_ACC_STORE(HD, NB, 12) FLASH_ACC_STORE(HD, NB, 13) FLASH_ACC_STORE(HD, NB, 14) \
    FLASH_ACC_STORE(HD, NB, 15)



// Second-generation flash16 variants.  The block loops are force-unrolled so
// the per-block accumulator update chain (if (b == B) accB = ...) resolves at
// compile time instead of evaluating sixteen runtime branches per value
// block, and the query values are cached in registers across the serial key
// scan instead of being re-read for every key.  Both changes are
// semantics-preserving, so the tolerance parity test applies unchanged.  The
// "_uw" variants additionally widen the slice count (more SIMD groups per
// head) within the 32 KiB threadgroup-memory limit for the merge buffers.
#define FLASH_QUERY_CACHE(QVAR, HEAD_DIM, BLOCKS) \
    float QVAR[BLOCKS]; \
    for (uint qb = 0; qb < BLOCKS; ++qb) { \
        uint qdim = 32 * qb + lane; \
        QVAR[qb] = query[head * HEAD_DIM + qdim]; \
    }

#define DEFINE_FLASH_ATTENTION_V2(NAME, HEAD_DIM, BLOCKS, SLICES) \
kernel void NAME( \
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]], \
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    if (head >= heads) return; \
    if (head_dim != HEAD_DIM) return; \
    if (simd_group >= SLICES) return; \
    uint kv_head = head / (heads / kv_heads); \
    uint blocks_per_position = kv_heads * BLOCKS; \
    uint value_base = capacity * blocks_per_position; \
    if (key_count == 0) { \
        for (uint b = 0; b < BLOCKS; ++b) { \
            uint dim = 32 * b + lane; \
            if (dim < HEAD_DIM) output[head * HEAD_DIM + dim] = 0.0f; \
        } \
        return; \
    } \
    uint start = simd_group * key_count / SLICES; \
    uint end = (simd_group + 1) * key_count / SLICES; \
    float maximum = -INFINITY; \
    float denominator = 0.0f; \
    FLASH_ACC_DECLS \
    FLASH_QUERY_CACHE(q_cache, HEAD_DIM, BLOCKS) \
    for (uint key = start; key < end; ++key) { \
        uint key_element = key * kv_heads * HEAD_DIM + kv_head * HEAD_DIM; \
        uint key_block_base = key_element / 32; \
        float partial = 0.0f; \
        _Pragma("unroll") \
        for (uint b = 0; b < BLOCKS; ++b) { \
            device const uchar *base = cache + (key_block_base + b) * 18; \
            float scale = simd_broadcast(float(*(device const half *)base), 0); \
            uchar packed = base[2 + (lane & 15)]; \
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4; \
            partial += q_cache[b] * scale * float(int(nibble) - 8); \
        } \
        float score = simd_sum(partial); \
        float rescale = 1.0f; \
        float weight; \
        if (score > maximum) { \
            rescale = exp(maximum - score); \
            weight = 1.0f; \
            maximum = score; \
            denominator = denominator * rescale + weight; \
        } else { \
            weight = exp(score - maximum); \
            denominator += weight; \
        } \
        uint value_block_base = value_base + key_block_base; \
        _Pragma("unroll") \
        for (uint b = 0; b < BLOCKS; ++b) { \
            device const uchar *base = cache + (value_block_base + b) * 18; \
            float scale = simd_broadcast(float(*(device const half *)base), 0); \
            uchar packed = base[2 + (lane & 15)]; \
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4; \
            float value = scale * float(int(nibble) - 8); \
            if (b == 0) acc0 = acc0 * rescale + weight * value; \
            FLASH_ACC_UPDATES \
        } \
    } \
    threadgroup float merg_max[SLICES]; \
    threadgroup float merg_sum[SLICES]; \
    threadgroup float merg_out[SLICES * HEAD_DIM]; \
    if (lane == 0) { \
        merg_max[simd_group] = maximum; \
        merg_sum[simd_group] = denominator; \
    } \
    FLASH_ACC_STORES(HEAD_DIM, BLOCKS) \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    threadgroup float global_max, global_sum; \
    if (tid == 0) { \
        global_max = -INFINITY; \
        for (uint g = 0; g < SLICES; ++g) global_max = max(global_max, merg_max[g]); \
        global_sum = 0.0f; \
        for (uint g = 0; g < SLICES; ++g) global_sum += merg_sum[g] * exp(merg_max[g] - global_max); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint b = 0; b < BLOCKS; ++b) { \
        uint dim = 32 * b + lane; \
        float value = 0.0f; \
        for (uint g = 0; g < SLICES; ++g) { \
            value += merg_out[g * HEAD_DIM + dim] * exp(merg_max[g] - global_max); \
        } \
        output[head * HEAD_DIM + dim] = value / global_sum; \
    } \
}

DEFINE_FLASH_ATTENTION_V2(attention_decode_gemma4_simd_q4_0_flash16_uw, 512, 16, 12)
DEFINE_FLASH_ATTENTION_V2(attention_decode_gemma4_simd_q4_0_flash16_swa_uw, 256, 8, 24)




// RMS-input counterpart of matmul_q4_0_qkv_32row_mv: the raw hidden state is
// normalized in-kernel (per-SIMD-group sum-of-squares reduction folded into
// the existing input loads) so the attention-input RMS dispatch and its
// device-memory round trip disappear.  Same tolerance contract as
// matvec_q4_0_32row_mv_rms.
kernel void matmul_q4_0_qkv_32row_mv_rms(
    device const float *input [[buffer(0)]],
    device const uchar *q_weights [[buffer(1)]],
    device const uchar *k_weights [[buffer(2)]],
    device const uchar *v_weights [[buffer(3)]],
    device float *q_output [[buffer(4)]],
    device float *k_output [[buffer(5)]],
    device float *v_output [[buffer(6)]],
    constant uint &input_width [[buffer(7)]],
    constant uint &q_width [[buffer(8)]],
    constant uint &kv_width [[buffer(9)]],
    device const float *rms_weight [[buffer(10)]],
    constant float &epsilon [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint q_groups = (q_width + 31) / 32;
    uint kv_groups = (kv_width + 31) / 32;
    uint projection = group < q_groups ? 0 : (group < q_groups + kv_groups ? 1 : 2);
    uint local_group = projection == 0 ? group :
        (projection == 1 ? group - q_groups : group - q_groups - kv_groups);
    uint output_width = projection == 0 ? q_width : kv_width;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = local_group * 32 + simdgroup * 8;
    bool active = row < output_width;
    uint blocks = input_width / 32;
    uint ix = lane / 2;
    uint il = (lane % 2) * 8;
    float sumf[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq = 0.0f;
    device const uchar *weights = projection == 0 ? q_weights :
        (projection == 1 ? k_weights : v_weights);
    device const uchar *ax[8];
    for (uint r = 0; r < 8; ++r) {
        uint safe_row = min(row + r, output_width - 1);
        ax[r] = weights + safe_row * blocks * 18;
    }
    float yl[16];
    device const float *yb = input + ix * 32 + il;
    device const float *wb = rms_weight + ix * 32 + il;
    for (uint ib = ix; ib < blocks; ib += 16) {
        float sumy0 = 0.0f;
        float sumy1 = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8; i += 2) {
            float y0 = yb[i + 0] * wb[i + 0];
            float y1 = yb[i + 1] * wb[i + 1];
            sumy0 += y0 + y1;
            yl[i + 0] = y0;
            yl[i + 1] = y1 * (1.0f / 256.0f);
            sum_sq += yb[i + 0] * yb[i + 0] + yb[i + 1] * yb[i + 1];
            float y2 = yb[i + 16] * wb[i + 16];
            float y3 = yb[i + 17] * wb[i + 17];
            sumy1 += y2 + y3;
            yl[i + 8] = y2 * (1.0f / 16.0f);
            yl[i + 9] = y3 * (1.0f / 4096.0f);
            sum_sq += yb[i + 16] * yb[i + 16] + yb[i + 17] * yb[i + 17];
        }
        float sumy = sumy0 + sumy1;
        if (active) {
            #pragma unroll
            for (uint r = 0; r < 8; ++r) {
                device const uchar *base = ax[r] + ib * 18;
                float scale = float(*(device const half *)base);
                device const ushort *qs = (device const ushort *)(base + 2 + il);
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 8; i += 2) {
                    ushort q = qs[i / 2];
                    acc0 += yl[i + 0] * float(q & 0x000F);
                    acc1 += yl[i + 1] * float(q & 0x0F00);
                    acc2 += yl[i + 8] * float(q & 0x00F0);
                    acc3 += yl[i + 9] * float(q & 0xF000);
                }
                sumf[r] += scale * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
            }
        }
        yb += 512;
        wb += 512;
    }
    float inverse_rms = rsqrt(simd_sum(sum_sq) / float(input_width) + epsilon);
    for (uint r = 0; r < 8; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 8; ++r) {
            uint out_row = row + r;
            if (out_row >= output_width) continue;
            float value = sumf[r] * inverse_rms;
            if (projection == 0) q_output[out_row] = value;
            else if (projection == 1) k_output[out_row] = value;
            else v_output[out_row] = value;
        }
    }
}


kernel void matmul_q4_0_gate_up_32row_mv_rms(
    device const float *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device float *gate_output [[buffer(3)]],
    device float *up_output [[buffer(4)]],
    constant uint &input_width [[buffer(5)]],
    constant uint &output_width [[buffer(6)]],
    device const float *rms_weight [[buffer(7)]],
    constant float &epsilon [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint groups = (output_width + 31) / 32;
    uint projection = group < groups ? 0 : 1;
    uint local_group = projection == 0 ? group : group - groups;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = local_group * 32 + simdgroup * 8;
    bool active = row < output_width;
    uint blocks = input_width / 32;
    uint ix = lane / 2;
    uint il = (lane % 2) * 8;
    float sumf[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq = 0.0f;
    device const uchar *weights = projection == 0 ? gate_weights : up_weights;
    device const uchar *ax[8];
    for (uint r = 0; r < 8; ++r) {
        uint safe_row = min(row + r, output_width - 1);
        ax[r] = weights + safe_row * blocks * 18;
    }
    float yl[16];
    device const float *yb = input + ix * 32 + il;
    device const float *wb = rms_weight + ix * 32 + il;
    for (uint ib = ix; ib < blocks; ib += 16) {
        float sumy0 = 0.0f;
        float sumy1 = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8; i += 2) {
            float y0 = yb[i + 0] * wb[i + 0];
            float y1 = yb[i + 1] * wb[i + 1];
            sumy0 += y0 + y1;
            yl[i + 0] = y0;
            yl[i + 1] = y1 * (1.0f / 256.0f);
            sum_sq += yb[i + 0] * yb[i + 0] + yb[i + 1] * yb[i + 1];
            float y2 = yb[i + 16] * wb[i + 16];
            float y3 = yb[i + 17] * wb[i + 17];
            sumy1 += y2 + y3;
            yl[i + 8] = y2 * (1.0f / 16.0f);
            yl[i + 9] = y3 * (1.0f / 4096.0f);
            sum_sq += yb[i + 16] * yb[i + 16] + yb[i + 17] * yb[i + 17];
        }
        float sumy = sumy0 + sumy1;
        if (active) {
            #pragma unroll
            for (uint r = 0; r < 8; ++r) {
                device const uchar *base = ax[r] + ib * 18;
                float scale = float(*(device const half *)base);
                device const ushort *qs = (device const ushort *)(base + 2 + il);
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 8; i += 2) {
                    ushort q = qs[i / 2];
                    acc0 += yl[i + 0] * float(q & 0x000F);
                    acc1 += yl[i + 1] * float(q & 0x0F00);
                    acc2 += yl[i + 8] * float(q & 0x00F0);
                    acc3 += yl[i + 9] * float(q & 0xF000);
                }
                sumf[r] += scale * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
            }
        }
        yb += 512;
        wb += 512;
    }
    float inverse_rms = rsqrt(simd_sum(sum_sq) / float(input_width) + epsilon);
    for (uint r = 0; r < 8; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 8; ++r) {
            uint out_row = row + r;
            if (out_row < output_width) {
                float value = sumf[r] * inverse_rms;
                if (projection == 0) gate_output[out_row] = value;
                else up_output[out_row] = value;
            }
        }
    }
}



// 64-row-per-threadgroup variants (phase 13.0 P4) of the mv_ext matvec
// family.  Each threadgroup covers 64 output rows via 8 SIMD groups of 8
// rows and is dispatched with 256 threads.  The per-lane accumulation order
// is byte-for-byte the 32-row kernel's, so the tolerance parity contract
// (max-abs < 1e-3 vs the 16-row/8-row production kernels) is unchanged.
kernel void matvec_q4_0_64row_mv(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = group * 64 + simdgroup * 8;
    bool active = row < output_width;
    uint blocks = input_width / 32;
    uint ix = lane / 2;
    uint il = (lane % 2) * 8;
    float sumf[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    device const uchar *ax[8];
    for (uint r = 0; r < 8; ++r) {
        uint safe_row = min(row + r, output_width - 1);
        ax[r] = weights + safe_row * blocks * 18;
    }
    float yl[16];
    device const float *yb = input + ix * 32 + il;
    for (uint ib = ix; ib < blocks; ib += 16) {
        float sumy0 = 0.0f;
        float sumy1 = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8; i += 2) {
            sumy0 += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
            yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
        }
        float sumy = sumy0 + sumy1;
        if (active) {
            #pragma unroll
            for (uint r = 0; r < 8; ++r) {
                device const uchar *base = ax[r] + ib * 18;
                float scale = float(*(device const half *)base);
                device const ushort *qs = (device const ushort *)(base + 2 + il);
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 8; i += 2) {
                    ushort q = qs[i / 2];
                    acc0 += yl[i + 0] * float(q & 0x000F);
                    acc1 += yl[i + 1] * float(q & 0x0F00);
                    acc2 += yl[i + 8] * float(q & 0x00F0);
                    acc3 += yl[i + 9] * float(q & 0xF000);
                }
                sumf[r] += scale * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
            }
        }
        yb += 512;
    }
    for (uint r = 0; r < 8; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 8; ++r) {
            uint out_row = row + r;
            if (out_row < output_width) output[out_row] = sumf[r];
        }
    }
}

// RMS-input counterpart of matvec_q4_0_64row_mv: the standalone rms-norm
// dispatch is folded into the projection exactly as in the 32-row _rms
// kernel, with the same per-lane sum-of-squares and yl-scale order.
kernel void matvec_q4_0_64row_mv_rms(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]],
    device const float *rms_weight [[buffer(5)]],
    constant float &epsilon [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = group * 64 + simdgroup * 8;
    bool active = row < output_width;
    uint blocks = input_width / 32;
    uint ix = lane / 2;
    uint il = (lane % 2) * 8;
    float sumf[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq = 0.0f;
    device const uchar *ax[8];
    for (uint r = 0; r < 8; ++r) {
        uint safe_row = min(row + r, output_width - 1);
        ax[r] = weights + safe_row * blocks * 18;
    }
    float yl[16];
    device const float *yb = input + ix * 32 + il;
    device const float *wb = rms_weight + ix * 32 + il;
    for (uint ib = ix; ib < blocks; ib += 16) {
        float sumy0 = 0.0f;
        float sumy1 = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8; i += 2) {
            float y0 = yb[i + 0] * wb[i + 0];
            float y1 = yb[i + 1] * wb[i + 1];
            sumy0 += y0 + y1;
            yl[i + 0] = y0;
            yl[i + 1] = y1 * (1.0f / 256.0f);
            sum_sq += yb[i + 0] * yb[i + 0] + yb[i + 1] * yb[i + 1];
            float y2 = yb[i + 16] * wb[i + 16];
            float y3 = yb[i + 17] * wb[i + 17];
            sumy1 += y2 + y3;
            yl[i + 8] = y2 * (1.0f / 16.0f);
            yl[i + 9] = y3 * (1.0f / 4096.0f);
            sum_sq += yb[i + 16] * yb[i + 16] + yb[i + 17] * yb[i + 17];
        }
        float sumy = sumy0 + sumy1;
        if (active) {
            #pragma unroll
            for (uint r = 0; r < 8; ++r) {
                device const uchar *base = ax[r] + ib * 18;
                float scale = float(*(device const half *)base);
                device const ushort *qs = (device const ushort *)(base + 2 + il);
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 8; i += 2) {
                    ushort q = qs[i / 2];
                    acc0 += yl[i + 0] * float(q & 0x000F);
                    acc1 += yl[i + 1] * float(q & 0x0F00);
                    acc2 += yl[i + 8] * float(q & 0x00F0);
                    acc3 += yl[i + 9] * float(q & 0xF000);
                }
                sumf[r] += scale * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
            }
        }
        yb += 512;
        wb += 512;
    }
    float inverse_rms = rsqrt(simd_sum(sum_sq) / float(input_width) + epsilon);
    for (uint r = 0; r < 8; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 8; ++r) {
            uint out_row = row + r;
            if (out_row < output_width) output[out_row] = sumf[r] * inverse_rms;
        }
    }
}

// 64-row-per-threadgroup counterpart of matvec_q6_k_32row_mv (8 SIMD groups
// of 8 rows per threadgroup, 256 threads per dispatch).
kernel void matvec_q6_k_64row_mv(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = group * 64 + simdgroup * 8;
    uint blocks = input_width / 256;
    uint tid_l = lane / 2;
    uint ix = lane % 2;
    uint ip = tid_l / 8;
    uint il = tid_l % 8;
    uint l0 = 4 * il;
    uint is = 8 * ip + l0 / 16;
    uint y_offset = 128 * ip + l0;
    uint q_offset_l = 64 * ip + l0;
    uint q_offset_h = 32 * ip + l0;
    float sumf[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float yl[16];
    uint safe_row = min(row, output_width - 1);
    bool active = row < output_width;
    for (uint ib = ix; ib < blocks; ib += 2) {
        device const uchar *base = weights + (safe_row * blocks + ib) * 210;
        device const float *y = input + ib * 256 + y_offset;
        #pragma unroll
        for (uint l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }
        if (active) {
            #pragma unroll
            for (uint r = 0; r < 8; ++r) {
                device const uchar *base_r = base + r * blocks * 210;
                device const uchar *q1 = base_r + q_offset_l;
                device const uchar *q2 = q1 + 32;
                device const uchar *qh = base_r + 128 + q_offset_h;
                device const int8_t *sc = (device const int8_t *)(base_r + 192) + is;
                float dh_r = float(*(device const half *)(base_r + 208));
                float sums0 = 0.0f;
                float sums1 = 0.0f;
                float sums2 = 0.0f;
                float sums3 = 0.0f;
                #pragma unroll
                for (uint l = 0; l < 4; ++l) {
                    sums0 += yl[4 * l + 0] * float(int((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                    sums1 += yl[4 * l + 1] * float(int((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                    sums2 += yl[4 * l + 2] * float(int((q1[l] >> 4) | ((qh[l] & 0x30) << 0)) - 32);
                    sums3 += yl[4 * l + 3] * float(int((q2[l] >> 4) | ((qh[l] & 0xC0) >> 2)) - 32);
                }
                sumf[r] += dh_r * (sums0 * float(sc[0]) + sums1 * float(sc[2])
                    + sums2 * float(sc[4]) + sums3 * float(sc[6]));
            }
        }
    }
    for (uint r = 0; r < 8; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 8; ++r) {
            uint out_row = row + r;
            if (out_row < output_width) output[out_row] = sumf[r];
        }
    }
}

// RMS-input counterpart of matvec_q6_k_64row_mv for the vocabulary output.
kernel void matvec_q6_k_64row_mv_rms(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]],
    device const float *rms_weight [[buffer(5)]],
    constant float &epsilon [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = group * 64 + simdgroup * 8;
    uint blocks = input_width / 256;
    uint tid_l = lane / 2;
    uint ix = lane % 2;
    uint ip = tid_l / 8;
    uint il = tid_l % 8;
    uint l0 = 4 * il;
    uint is = 8 * ip + l0 / 16;
    uint y_offset = 128 * ip + l0;
    uint q_offset_l = 64 * ip + l0;
    uint q_offset_h = 32 * ip + l0;
    float sumf[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq = 0.0f;
    float yl[16];
    uint safe_row = min(row, output_width - 1);
    bool active = row < output_width;
    for (uint ib = ix; ib < blocks; ib += 2) {
        device const uchar *base = weights + (safe_row * blocks + ib) * 210;
        device const float *y = input + ib * 256 + y_offset;
        device const float *w = rms_weight + ib * 256 + y_offset;
        #pragma unroll
        for (uint l = 0; l < 4; ++l) {
            sum_sq += y[l + 0] * y[l + 0] + y[l + 32] * y[l + 32]
                + y[l + 64] * y[l + 64] + y[l + 96] * y[l + 96];
            yl[4 * l + 0] = y[l + 0] * w[l + 0];
            yl[4 * l + 1] = y[l + 32] * w[l + 32];
            yl[4 * l + 2] = y[l + 64] * w[l + 64];
            yl[4 * l + 3] = y[l + 96] * w[l + 96];
        }
        if (active) {
            #pragma unroll
            for (uint r = 0; r < 8; ++r) {
                device const uchar *base_r = base + r * blocks * 210;
                device const uchar *q1 = base_r + q_offset_l;
                device const uchar *q2 = q1 + 32;
                device const uchar *qh = base_r + 128 + q_offset_h;
                device const int8_t *sc = (device const int8_t *)(base_r + 192) + is;
                float dh_r = float(*(device const half *)(base_r + 208));
                float sums0 = 0.0f;
                float sums1 = 0.0f;
                float sums2 = 0.0f;
                float sums3 = 0.0f;
                #pragma unroll
                for (uint l = 0; l < 4; ++l) {
                    sums0 += yl[4 * l + 0] * float(int((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                    sums1 += yl[4 * l + 1] * float(int((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                    sums2 += yl[4 * l + 2] * float(int((q1[l] >> 4) | ((qh[l] & 0x30) << 0)) - 32);
                    sums3 += yl[4 * l + 3] * float(int((q2[l] >> 4) | ((qh[l] & 0xC0) >> 2)) - 32);
                }
                sumf[r] += dh_r * (sums0 * float(sc[0]) + sums1 * float(sc[2])
                    + sums2 * float(sc[4]) + sums3 * float(sc[6]));
            }
        }
    }
    float inverse_rms = rsqrt(simd_sum(sum_sq) / float(input_width) + epsilon);
    for (uint r = 0; r < 8; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 8; ++r) {
            uint out_row = row + r;
            if (out_row < output_width) output[out_row] = sumf[r] * inverse_rms;
        }
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
