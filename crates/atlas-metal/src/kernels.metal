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
// and explicit intermediate arithmetic of rms_norm_decode_f32, then writes
// the following residual in the same dispatch.  It is intentionally opt-in:
// its value is fewer dispatch boundaries, not a changed numerical contract.
kernel void gemma4_rms_residual_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    // Keep the normalized vector as a volatile device-memory intermediate.
    // The baseline writes this value in rms_norm_decode_f32, then reloads it
    // in vector_add_f32; retaining that rounding boundary is required for
    // greedy-token parity.
    device volatile float *normalized [[buffer(3)]], device float *output [[buffer(4)]],
    constant uint &hidden [[buffer(5)]], constant float &epsilon [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]]) {
    float squared_sum = 0.0f;
    for (uint column = lane; column < hidden; column += 32) {
        float value = input[column];
        squared_sum += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(squared_sum) / float(hidden) + epsilon);
    for (uint column = lane; column < hidden; column += 32) {
        normalized[column] = input[column] * inverse_rms * weight[column];
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint column = lane; column < hidden; column += 32) {
        output[column] = residual[column] + normalized[column];
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

// GGUF block-32 quantizers.  The output layout is exactly block_q4_0
// (fp16 scale + 16 packed signed nibbles) or block_q8_0 (fp16 scale + 32 i8).
kernel void quantize_q4_0(
    device const float *input [[buffer(0)]], device uchar *output [[buffer(1)]],
    constant uint &blocks [[buffer(2)]], uint block_id [[thread_position_in_grid]]) {
    if (block_id >= blocks) return;
    float maximum = 0.0f;
    float signed_maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) {
        float value = input[block_id * 32 + i];
        if (abs(value) > maximum) { maximum = abs(value); signed_maximum = value; }
    }
    float scale = maximum == 0.0f ? 0.0f : signed_maximum / -8.0f;
    device half *scale_out = (device half *)(output + block_id * 18);
    *scale_out = half(scale);
    for (uint i = 0; i < 16; ++i) {
        int a = scale == 0.0f ? 0 : int(round(clamp(input[block_id * 32 + i] / scale, -8.0f, 7.0f)));
        int b = scale == 0.0f ? 0 : int(round(clamp(input[block_id * 32 + i + 16] / scale, -8.0f, 7.0f)));
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

// Four rows per SIMD-group, sixteen rows per threadgroup. Eight lanes own a
// row; each lane consumes four input values and two packed bytes per block.
kernel void matvec_q4_0_16row(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input[block * 32 + column] * float(int(packed0 & 15) - 8) * scale;
            sum += input[block * 32 + column + 8] * float(int(packed1 & 15) - 8) * scale;
            sum += input[block * 32 + column + 16] * float(int(packed0 >> 4) - 8) * scale;
            sum += input[block * 32 + column + 24] * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) output[row] = sum;
}

// FFN-down-only companion for the resident [16-row tile][block][row] Q4_0
// layout. It retains the proven 128-thread arithmetic and reduction order.
kernel void matvec_q4_0_16row_ffn_down_interleaved(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        uint tile_rows = min(16u, output_width - group * 16);
        uint tile_base = group * 16 * blocks * 18;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + tile_base
                + (block * tile_rows + simdgroup * 4 + row_in_simd) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input[block * 32 + column] * float(int(packed0 & 15) - 8) * scale;
            sum += input[block * 32 + column + 8] * float(int(packed1 & 15) - 8) * scale;
            sum += input[block * 32 + column + 16] * float(int(packed0 >> 4) - 8) * scale;
            sum += input[block * 32 + column + 24] * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) output[row] = sum;
}

// General resident packed-16 companion.  Its byte layout is identical to the
// FFN-down experiment above, but it is named separately so PLE/FFN composition
// telemetry can prove which opt-in layout was selected.
kernel void matvec_q4_0_16row_packed16(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        uint tile_rows = min(16u, output_width - group * 16);
        uint tile_base = group * 16 * blocks * 18;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + tile_base
                + (block * tile_rows + simdgroup * 4 + row_in_simd) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input[block * 32 + column] * float(int(packed0 & 15) - 8) * scale;
            sum += input[block * 32 + column + 8] * float(int(packed1 & 15) - 8) * scale;
            sum += input[block * 32 + column + 16] * float(int(packed0 >> 4) - 8) * scale;
            sum += input[block * 32 + column + 24] * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) output[row] = sum;
}

// Same 16-row Q4_0 tile as matvec_q4_0_16row, but each SIMD group loads a
// 32-value activation block once.  The four rows in the group then obtain the
// values needed by their eight-lane partial reductions through SIMD shuffles.
// This preserves the packed-weight layout and per-lane accumulation order.
kernel void matvec_q4_0_16row_shared_input(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        for (uint block = 0; block < blocks; ++block) {
            uint input_base = block * 32;
            float input_lane = input[input_base + lane];
            float input0 = simd_shuffle(input_lane, ushort(column));
            float input8 = simd_shuffle(input_lane, ushort(column + 8));
            float input16 = simd_shuffle(input_lane, ushort(column + 16));
            float input24 = simd_shuffle(input_lane, ushort(column + 24));
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input0 * float(int(packed0 & 15) - 8) * scale;
            sum += input8 * float(int(packed1 & 15) - 8) * scale;
            sum += input16 * float(int(packed0 >> 4) - 8) * scale;
            sum += input24 * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) output[row] = sum;
}

// Whole-path SIMD-group candidate for Resident decode.  Each 32-wide input
// block is loaded once per SIMD-group and broadcast to the four rows that the
// group owns.  It retains the baseline's F32 products, per-lane order, and
// eight-lane reduction, so it is safe to use in exact-token A/Bs.
kernel void matvec_q4_0_16row_simdgroup_tiled(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        for (uint block = 0; block < blocks; ++block) {
            float input_lane = input[block * 32 + lane];
            float input0 = simd_shuffle(input_lane, ushort(column));
            float input8 = simd_shuffle(input_lane, ushort(column + 8));
            float input16 = simd_shuffle(input_lane, ushort(column + 16));
            float input24 = simd_shuffle(input_lane, ushort(column + 24));
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input0 * float(int(packed0 & 15) - 8) * scale;
            sum += input8 * float(int(packed1 & 15) - 8) * scale;
            sum += input16 * float(int(packed0 >> 4) - 8) * scale;
            sum += input24 * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) output[row] = sum;
}

// Fused Q/K/V projection dispatch for Gemma provider layers.  The dispatch
// keeps the established 16-row Q4 tile and accumulation order, but maps the
// group range across the Q, K, and V matrices so one command encoder launch
// replaces three independent projection launches.
kernel void matmul_q4_0_qkv_16row(
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
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint q_groups = (q_width + 15) / 16;
    uint kv_groups = (kv_width + 15) / 16;
    uint projection = group < q_groups ? 0 : (group < q_groups + kv_groups ? 1 : 2);
    uint local_group = projection == 0 ? group :
        (projection == 1 ? group - q_groups : group - q_groups - kv_groups);
    uint output_width = projection == 0 ? q_width : kv_width;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = local_group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        device const uchar *weights = projection == 0 ? q_weights :
            (projection == 1 ? k_weights : v_weights);
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input[block * 32 + column] * float(int(packed0 & 15) - 8) * scale;
            sum += input[block * 32 + column + 8] * float(int(packed1 & 15) - 8) * scale;
            sum += input[block * 32 + column + 16] * float(int(packed0 >> 4) - 8) * scale;
            sum += input[block * 32 + column + 24] * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) {
        if (projection == 0) q_output[row] = sum;
        else if (projection == 1) k_output[row] = sum;
        else v_output[row] = sum;
    }
}

// SIMD-group tiled counterpart of the fused Q/K/V projection.  This keeps
// the existing fused dispatch boundary while removing redundant activation
// loads for the four rows owned by each SIMD-group.
kernel void matmul_q4_0_qkv_16row_simdgroup_tiled(
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
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint q_groups = (q_width + 15) / 16;
    uint kv_groups = (kv_width + 15) / 16;
    uint projection = group < q_groups ? 0 : (group < q_groups + kv_groups ? 1 : 2);
    uint local_group = projection == 0 ? group :
        (projection == 1 ? group - q_groups : group - q_groups - kv_groups);
    uint output_width = projection == 0 ? q_width : kv_width;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = local_group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        device const uchar *weights = projection == 0 ? q_weights :
            (projection == 1 ? k_weights : v_weights);
        for (uint block = 0; block < blocks; ++block) {
            float input_lane = input[block * 32 + lane];
            float input0 = simd_shuffle(input_lane, ushort(column));
            float input8 = simd_shuffle(input_lane, ushort(column + 8));
            float input16 = simd_shuffle(input_lane, ushort(column + 16));
            float input24 = simd_shuffle(input_lane, ushort(column + 24));
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input0 * float(int(packed0 & 15) - 8) * scale;
            sum += input8 * float(int(packed1 & 15) - 8) * scale;
            sum += input16 * float(int(packed0 >> 4) - 8) * scale;
            sum += input24 * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) {
        if (projection == 0) q_output[row] = sum;
        else if (projection == 1) k_output[row] = sum;
        else v_output[row] = sum;
    }
}

// Fused FFN gate/up projection dispatch. Both matrices consume the same
// normalized token vector, so this follows the Q/K/V fusion shape: retain the
// proven 16-row Q4 tile while replacing two command-encoder launches with one.
// The gate and up outputs remain independent FP32 buffers and use the exact
// same per-row accumulation order as matvec_q4_0_16row.
kernel void matmul_q4_0_gate_up_16row(
    device const float *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device float *gate_output [[buffer(3)]],
    device float *up_output [[buffer(4)]],
    constant uint &input_width [[buffer(5)]],
    constant uint &output_width [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint groups = (output_width + 15) / 16;
    uint projection = group < groups ? 0 : 1;
    uint local_group = projection == 0 ? group : group - groups;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = local_group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        device const uchar *weights = projection == 0 ? gate_weights : up_weights;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input[block * 32 + column] * float(int(packed0 & 15) - 8) * scale;
            sum += input[block * 32 + column + 8] * float(int(packed1 & 15) - 8) * scale;
            sum += input[block * 32 + column + 16] * float(int(packed0 >> 4) - 8) * scale;
            sum += input[block * 32 + column + 24] * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) {
        if (projection == 0) gate_output[row] = sum;
        else up_output[row] = sum;
    }
}

kernel void matmul_q4_0_gate_up_16row_simdgroup_tiled(
    device const float *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device float *gate_output [[buffer(3)]],
    device float *up_output [[buffer(4)]],
    constant uint &input_width [[buffer(5)]],
    constant uint &output_width [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint groups = (output_width + 15) / 16;
    uint projection = group < groups ? 0 : 1;
    uint local_group = projection == 0 ? group : group - groups;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = local_group * 16 + simdgroup * 4 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        device const uchar *weights = projection == 0 ? gate_weights : up_weights;
        for (uint block = 0; block < blocks; ++block) {
            float input_lane = input[block * 32 + lane];
            float input0 = simd_shuffle(input_lane, ushort(column));
            float input8 = simd_shuffle(input_lane, ushort(column + 8));
            float input16 = simd_shuffle(input_lane, ushort(column + 16));
            float input24 = simd_shuffle(input_lane, ushort(column + 24));
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input0 * float(int(packed0 & 15) - 8) * scale;
            sum += input8 * float(int(packed1 & 15) - 8) * scale;
            sum += input16 * float(int(packed0 >> 4) - 8) * scale;
            sum += input24 * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) {
        if (projection == 0) gate_output[row] = sum;
        else up_output[row] = sum;
    }
}

inline float atlas_tanh_f32(float value);

// Decode-only Gate/Up epilogue. Each eight-lane unit preserves the Q4_0
// accumulation and reduction order used by matmul_q4_0_gate_up_16row, but
// keeps the paired row sums in registers and writes only the FFN-down input.
kernel void matmul_q4_0_gate_up_gelu_16row(
    device const float *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &input_width [[buffer(4)]],
    constant uint &output_width [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint row = group * 16 + simdgroup * 4 + row_in_simd;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 32;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *gate_base = gate_weights + (row * blocks + block) * 18;
            device const uchar *up_base = up_weights + (row * blocks + block) * 18;
            float gate_scale = float(*(device const half *)gate_base);
            float up_scale = float(*(device const half *)up_base);
            uchar gate0 = gate_base[2 + column];
            uchar gate1 = gate_base[2 + column + 8];
            uchar up0 = up_base[2 + column];
            uchar up1 = up_base[2 + column + 8];
            float x0 = input[block * 32 + column];
            float x8 = input[block * 32 + column + 8];
            float x16 = input[block * 32 + column + 16];
            float x24 = input[block * 32 + column + 24];
            gate_sum += x0 * float(int(gate0 & 15) - 8) * gate_scale;
            gate_sum += x8 * float(int(gate1 & 15) - 8) * gate_scale;
            gate_sum += x16 * float(int(gate0 >> 4) - 8) * gate_scale;
            gate_sum += x24 * float(int(gate1 >> 4) - 8) * gate_scale;
            up_sum += x0 * float(int(up0 & 15) - 8) * up_scale;
            up_sum += x8 * float(int(up1 & 15) - 8) * up_scale;
            up_sum += x16 * float(int(up0 >> 4) - 8) * up_scale;
            up_sum += x24 * float(int(up1 >> 4) - 8) * up_scale;
        }
    }
    gate_sum += simd_shuffle_xor(gate_sum, 4);
    gate_sum += simd_shuffle_xor(gate_sum, 2);
    gate_sum += simd_shuffle_xor(gate_sum, 1);
    up_sum += simd_shuffle_xor(up_sum, 4);
    up_sum += simd_shuffle_xor(up_sum, 2);
    up_sum += simd_shuffle_xor(up_sum, 1);
    if (column == 0 && row < output_width) {
        float argument = 0.7978845608f * (gate_sum + 0.044715f * gate_sum * gate_sum * gate_sum);
        float gelu = isinf(argument) ? (argument > 0.0f ? gate_sum : 0.0f)
                                   : 0.5f * gate_sum * (1.0f + atlas_tanh_f32(argument));
        output[row] = gelu * up_sum;
    }
}

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

kernel void matmul_q4_0_batch_16row_simdgroup_tiled(
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
            float input_lane = token_input[block * 32 + lane];
            float input0 = simd_shuffle(input_lane, ushort(column));
            float input8 = simd_shuffle(input_lane, ushort(column + 8));
            float input16 = simd_shuffle(input_lane, ushort(column + 16));
            float input24 = simd_shuffle(input_lane, ushort(column + 24));
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            sum += input0 * float(int(packed0 & 15) - 8) * scale;
            sum += input8 * float(int(packed1 & 15) - 8) * scale;
            sum += input16 * float(int(packed0 >> 4) - 8) * scale;
            sum += input24 * float(int(packed1 >> 4) - 8) * scale;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && token < batch && row < output_width)
        output[token * output_width + row] = sum;
}

// Batched prefill companion for matvec_q4_0_16row_ffn_down_interleaved.
kernel void matmul_q4_0_batch_16row_ffn_down_interleaved(
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
        uint tile_rows = min(16u, output_width - group_row * 16);
        uint tile_base = group_row * 16 * blocks * 18;
        device const float *token_input = input + token * input_width;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + tile_base
                + (block * tile_rows + simdgroup * 4 + row_in_simd) * 18;
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
    if (column == 0 && token < batch && row < output_width) output[token * output_width + row] = sum;
}

// Batched prefill companion for the general packed-16 resident layout.
kernel void matmul_q4_0_batch_16row_packed16(
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
        uint tile_rows = min(16u, output_width - group_row * 16);
        uint tile_base = group_row * 16 * blocks * 18;
        device const float *token_input = input + token * input_width;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + tile_base
                + (block * tile_rows + simdgroup * 4 + row_in_simd) * 18;
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
    if (column == 0 && token < batch && row < output_width) output[token * output_width + row] = sum;
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

// Q6_K matrix-vector multiplication shares the exact block layout used by the
// Gemma E2B embedding tables.  Keeping it packed avoids a hidden FP32 output
// projection cache for Gemma's tied vocabulary matrix.
kernel void matvec_q6_k(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= output_width) return;
    float sum = 0.0f;
    for (uint block = 0; block < input_width / 256; ++block) {
        device const uchar *base = weights + (row * (input_width / 256) + block) * 210;
        for (uint index = 0; index < 256; ++index) {
            sum += input[block * 256 + index] * q6_k_value(base, index);
        }
    }
    output[row] = sum;
}

// Q6_K vocabulary projection with eight rows per threadgroup. A half-SIMD
// owns one row and evaluates sixteen values per 256-value block.
kernel void matvec_q6_k_8row(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 16;
    uint column = lane % 16;
    uint row = group * 8 + simdgroup * 2 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 256;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + (row * blocks + block) * 210;
            for (uint index = column; index < 256; index += 16)
                sum += input[block * 256 + index] * q6_k_value(base, index);
        }
    }
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) output[row] = sum;
}

// Opt-in LM-head variant: share the Q6_K super-block scale across the two
// half-SIMD rows while retaining the canonical packed-bit and group-scale math.
kernel void matvec_q6_k_8row_cacheopt(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint simdgroup = tid / 32;
    uint row_in_simd = lane / 16;
    uint column = lane % 16;
    uint row = group * 8 + simdgroup * 2 + row_in_simd;
    float sum = 0.0f;
    if (row < output_width) {
        uint blocks = input_width / 256;
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + (row * blocks + block) * 210;
            float block_scale = simd_broadcast(float(*(device const half *)(base + 208)), row_in_simd * 16);
            for (uint index = column; index < 256; index += 16) {
                uint chunk = index / 128;
                uint within = index % 128;
                uint stream = within / 32;
                uint value_lane = within % 32;
                uchar packed = base[chunk * 64 + value_lane + ((stream & 1) ? 32 : 0)];
                uchar low = stream >= 2 ? packed >> 4 : packed & 15;
                uchar high = (base[128 + chunk * 32 + value_lane] >> (stream * 2)) & 3;
                int group_scale = int((char) base[192 + chunk * 8 + value_lane / 16 + stream * 2]);
                int quantized = int((high << 4) | low) - 32;
                sum += input[block * 256 + index] * float(quantized * group_scale) * block_scale;
            }
        }
    }
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (column == 0 && row < output_width) output[row] = sum;
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

// Trace-only GELU variant. Its intermediate buffers make a finite-input NaN
// attributable to a specific arithmetic operation rather than a later reuse
// of the activation arena.
kernel void gelu_trace_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    device float *cubic_output [[buffer(2)]], device float *argument_output [[buffer(3)]],
    device float *tanh_output [[buffer(4)]], constant uint &count [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) {
        float x = input[id];
        float cubic = 0.044715f * x * x * x;
        float argument = 0.7978845608f * (x + cubic);
        float tanh_value = atlas_tanh_f32(argument);
        cubic_output[id] = cubic;
        argument_output[id] = argument;
        tanh_output[id] = tanh_value;
        output[id] = isinf(argument) ? (argument > 0.0f ? x : 0.0f)
                                   : 0.5f * x * (1.0f + tanh_value);
    }
}

kernel void copy_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = input[id];
}

kernel void copy_u32(
    device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = input[id];
}

kernel void rms_norm_groups_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &width [[buffer(3)]],
    constant uint &groups [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    uint group = id / width, column = id % width;
    if (group >= groups) return;
    uint base = group * width;
    float squared_sum = 0.0f;
    for (uint index = 0; index < width; ++index) squared_sum += input[base + index] * input[base + index];
    output[base + column] = input[base + column] * rsqrt(squared_sum / float(width) + epsilon) * weight[column];
}

kernel void rms_norm_groups_unweighted_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &width [[buffer(2)]], constant uint &groups [[buffer(3)]], constant float &epsilon [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    uint group = id / width, column = id % width;
    if (group >= groups) return;
    uint base = group * width;
    float squared_sum = 0.0f;
    for (uint index = 0; index < width; ++index) squared_sum += input[base + index] * input[base + index];
    output[base + column] = input[base + column] * rsqrt(squared_sum / float(width) + epsilon);
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

kernel void first_nonfinite_f32(
    device const float *values [[buffer(0)]], device atomic_uint *result [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant uint &slot [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id != 0) return;
    uint first = count;
    for (uint index = 0; index < count; ++index) {
        if (!isfinite(values[index])) { first = index; break; }
    }
    if (first < count) atomic_fetch_min_explicit(&result[0], (slot << 16) | first, memory_order_relaxed);
}

kernel void max_abs_f32(
    device const float *values [[buffer(0)]], device float *result [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant uint &slot [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id != 0) return;
    float maximum = 0.0f;
    for (uint index = 0; index < count; ++index) maximum = max(maximum, abs(values[index]));
    result[slot] = maximum;
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
        int low = scale == 0.0f ? 0 : int(round(clamp(key[base + i] / scale, -8.0f, 7.0f)));
        int high = scale == 0.0f ? 0 : int(round(clamp(key[base + i + 16] / scale, -8.0f, 7.0f)));
        out[2 + i] = uchar((low + 8) | ((high + 8) << 4));
    }
    maximum = 0.0f; signed_maximum = 0.0f;
    for (uint i = 0; i < 32; ++i) if (abs(value[base + i]) > maximum) { maximum = abs(value[base + i]); signed_maximum = value[base + i]; }
    scale = maximum == 0.0f ? 0.0f : signed_maximum / -8.0f;
    out = cache + (capacity * blocks + position * blocks + block) * 18;
    *((device half *)out) = half(scale);
    for (uint i = 0; i < 16; ++i) {
        int low = scale == 0.0f ? 0 : int(round(clamp(value[base + i] / scale, -8.0f, 7.0f)));
        int high = scale == 0.0f ? 0 : int(round(clamp(value[base + i + 16] / scale, -8.0f, 7.0f)));
        out[2 + i] = uchar((low + 8) | ((high + 8) << 4));
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

// Gemma 4 normalizes Q and K per head and explicitly uses attention scale
// 1.0. Keep this separate from the Llama fused kernel, whose contract is
// 1/sqrt(head_dim), so changing Gemma cannot regress existing executors.
kernel void attention_decode_fused_gemma4_f32(
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
            float score = reductions[0];
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
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (head >= heads) return;
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
    for (uint key = 0; key < key_count; ++key) {
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
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (head >= heads) return;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = 0; key < key_count; ++key) {
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
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (head >= heads) return;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = 0; key < key_count; ++key) {
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

// MLX-style split attention for longer Gemma Q4 decode contexts.  The first
// pass owns a contiguous range of KV positions per (query head, block), keeps
// online-softmax state local, and writes only one FP32 partial vector plus its
// max/sum-exp pair.  It never materializes a score matrix or dequantized KV
// cache.  The second pass combines those four stable partials per query head.
kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_1(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *partials [[buffer(2)]], device float *maxima [[buffer(3)]],
    device float *sums [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant uint &key_count [[buffer(9)]],
    constant uint &blocks [[buffer(10)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    uint head = group / blocks;
    uint block = group % blocks;
    if (head >= heads) return;
    uint start = block * key_count / blocks;
    uint end = (block + 1) * key_count / blocks;
    uint slot = block * heads + head;
    uint partial_base = slot * head_dim;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) partials[partial_base + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = start; key < end; ++key) {
        float partial = 0.0f;
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim;
        for (uint d = tid; d < head_dim; d += threads) {
            uint index = key_element + d;
            partial += query[head * head_dim + d]
                * kv_q4_0_value(cache + (index / 32) * 18, index % 32);
        }
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
        for (uint d = tid; d < head_dim; d += threads) {
            uint index = key_element + d;
            partials[partial_base + d] = partials[partial_base + d] * rescale
                + weight * kv_q4_0_value(cache + (value_base + index / 32) * 18, index % 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        maxima[slot] = maximum;
        sums[slot] = denominator;
    }
}

// Exact arithmetic variant of the accepted four-way split scan.  Each thread
// owns disjoint partial-value elements, and the next key reads only query/KV
// data plus the score scratch.  The trailing post-value barrier in the
// production kernel therefore has no consumer; omitting it lets threads start
// the next score while preserving the per-thread value accumulation order.
kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_1_no_value_barrier(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *partials [[buffer(2)]], device float *maxima [[buffer(3)]],
    device float *sums [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant uint &key_count [[buffer(9)]],
    constant uint &blocks [[buffer(10)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    uint head = group / blocks;
    uint block = group % blocks;
    if (head >= heads) return;
    uint start = block * key_count / blocks;
    uint end = (block + 1) * key_count / blocks;
    uint slot = block * heads + head;
    uint partial_base = slot * head_dim;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) partials[partial_base + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = start; key < end; ++key) {
        float partial = 0.0f;
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim;
        for (uint d = tid; d < head_dim; d += threads) {
            uint index = key_element + d;
            partial += query[head * head_dim + d]
                * kv_q4_0_value(cache + (index / 32) * 18, index % 32);
        }
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
        for (uint d = tid; d < head_dim; d += threads) {
            uint index = key_element + d;
            partials[partial_base + d] = partials[partial_base + d] * rescale
                + weight * kv_q4_0_value(cache + (value_base + index / 32) * 18, index % 32);
        }
    }
    if (tid == 0) {
        maxima[slot] = maximum;
        sums[slot] = denominator;
    }
}

// Candidate that preserves the production four-way, 128-thread geometry and
// exact online-softmax/value update order. Its two-key inner loop is explicitly
// unrolled so the compiler can optimize sequential KV address arithmetic.
kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_1_unroll2_no_value_barrier(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *partials [[buffer(2)]], device float *maxima [[buffer(3)]],
    device float *sums [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant uint &key_count [[buffer(9)]],
    constant uint &blocks [[buffer(10)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    uint head = group / blocks;
    uint block = group % blocks;
    if (head >= heads) return;
    uint start = block * key_count / blocks;
    uint end = (block + 1) * key_count / blocks;
    uint slot = block * heads + head;
    uint partial_base = slot * head_dim;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) partials[partial_base + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pair_start = start; pair_start < end; pair_start += 2) {
        uint pair_count = min(2u, end - pair_start);
#pragma unroll
        for (uint pair_offset = 0; pair_offset < 2; ++pair_offset) {
            if (pair_offset >= pair_count) break;
            uint key = pair_start + pair_offset;
            float partial = 0.0f;
            uint key_element = key * kv_heads * head_dim + kv_head * head_dim;
            for (uint d = tid; d < head_dim; d += threads) {
                uint index = key_element + d;
                partial += query[head * head_dim + d]
                    * kv_q4_0_value(cache + (index / 32) * 18, index % 32);
            }
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
            for (uint d = tid; d < head_dim; d += threads) {
                uint index = key_element + d;
                partials[partial_base + d] = partials[partial_base + d] * rescale
                    + weight * kv_q4_0_value(cache + (value_base + index / 32) * 18, index % 32);
            }
        }
    }
    if (tid == 0) {
        maxima[slot] = maximum;
        sums[slot] = denominator;
    }
}

// One-SIMD-group variant of the split first pass. Each lane calculates the
// four interleaved partial sums owned by the production kernel's four SIMD
// groups, then reduces each partial independently in the same order. This
// preserves the score arithmetic while replacing the hot per-key
// threadgroup barriers with simdgroup operations.
kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_1_simd(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *partials [[buffer(2)]], device float *maxima [[buffer(3)]],
    device float *sums [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant uint &key_count [[buffer(9)]],
    constant uint &blocks [[buffer(10)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = group / blocks;
    uint block = group % blocks;
    if (head >= heads) return;
    uint start = block * key_count / blocks;
    uint end = (block + 1) * key_count / blocks;
    uint slot = block * heads + head;
    uint partial_base = slot * head_dim;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    float maximum = -INFINITY;
    float denominator = 0.0f;

    for (uint simd = 0; simd < 4; ++simd) {
        uint first = simd * 32 + lane;
        for (uint d = first; d < head_dim; d += 128) {
            partials[partial_base + d] = 0.0f;
        }
    }

    for (uint key = start; key < end; ++key) {
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim;
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        for (uint d = lane; d < head_dim; d += 128) {
            partial0 += query[head * head_dim + d]
                * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32);
        }
        for (uint d = 32 + lane; d < head_dim; d += 128) {
            partial1 += query[head * head_dim + d]
                * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32);
        }
        for (uint d = 64 + lane; d < head_dim; d += 128) {
            partial2 += query[head * head_dim + d]
                * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32);
        }
        for (uint d = 96 + lane; d < head_dim; d += 128) {
            partial3 += query[head * head_dim + d]
                * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32);
        }
        float score = simd_sum(partial0) + simd_sum(partial1)
            + simd_sum(partial2) + simd_sum(partial3);
        float rescale = 1.0f;
        float weight = 0.0f;
        if (lane == 0) {
            if (score > maximum) {
                rescale = exp(maximum - score);
                weight = 1.0f;
                maximum = score;
                denominator = denominator * rescale + weight;
            } else {
                weight = exp(score - maximum);
                denominator += weight;
            }
        }
        rescale = simd_broadcast(rescale, 0);
        weight = simd_broadcast(weight, 0);
        for (uint simd = 0; simd < 4; ++simd) {
            uint first = simd * 32 + lane;
            for (uint d = first; d < head_dim; d += 128) {
                uint index = key_element + d;
                partials[partial_base + d] = partials[partial_base + d] * rescale
                    + weight * kv_q4_0_value(cache + (value_base + index / 32) * 18, index % 32);
            }
        }
    }
    if (lane == 0) {
        maxima[slot] = maximum;
        sums[slot] = denominator;
    }
}

// Packed-Q4 variant of the first split-attention pass. It deliberately keeps
// the production four-way KV ranges, 128-thread geometry, and online-softmax
// arithmetic. Each SIMD group owns complete Q4 blocks, so its half scale is
// loaded once and broadcast to its lanes instead of reloaded per dimension.
kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_1_cacheopt(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *partials [[buffer(2)]], device float *maxima [[buffer(3)]],
    device float *sums [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant uint &key_count [[buffer(9)]],
    constant uint &blocks [[buffer(10)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    uint head = group / blocks;
    uint block = group % blocks;
    if (head >= heads) return;
    uint start = block * key_count / blocks;
    uint end = (block + 1) * key_count / blocks;
    uint slot = block * heads + head;
    uint partial_base = slot * head_dim;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_head = head_dim / 32;
    uint blocks_per_position = kv_heads * blocks_per_head;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) partials[partial_base + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = start; key < end; ++key) {
        float partial = 0.0f;
        uint key_block_base = key * blocks_per_position + kv_head * blocks_per_head;
        for (uint q4_block = simd_group; q4_block < blocks_per_head; q4_block += 4) {
            device const uchar *base = cache + (key_block_base + q4_block) * 18;
            float scale = simd_broadcast(float(*(device const half *)base), 0);
            uchar packed = base[2 + (lane & 15)];
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
            float value = scale * float(int(nibble) - 8);
            uint d = q4_block * 32 + lane;
            partial += query[head * head_dim + d] * value;
        }
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
        uint value_block_base = value_base + key_block_base;
        for (uint q4_block = simd_group; q4_block < blocks_per_head; q4_block += 4) {
            device const uchar *base = cache + (value_block_base + q4_block) * 18;
            float scale = simd_broadcast(float(*(device const half *)base), 0);
            uchar packed = base[2 + (lane & 15)];
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
            float value = scale * float(int(nibble) - 8);
            uint d = q4_block * 32 + lane;
            partials[partial_base + d] = partials[partial_base + d] * rescale + weight * value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        maxima[slot] = maximum;
        sums[slot] = denominator;
    }
}

// Hybrid candidate: retain cacheopt's one-scale-per-Q4-block dequantization
// while removing the post-value barrier proven unnecessary by the production
// no-value-barrier scan. Each thread still owns a distinct partial element,
// and the next key only reads query/KV plus score scratch.
kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_1_cacheopt_no_value_barrier(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *partials [[buffer(2)]], device float *maxima [[buffer(3)]],
    device float *sums [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant uint &key_count [[buffer(9)]],
    constant uint &blocks [[buffer(10)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    uint head = group / blocks;
    uint block = group % blocks;
    if (head >= heads) return;
    uint start = block * key_count / blocks;
    uint end = (block + 1) * key_count / blocks;
    uint slot = block * heads + head;
    uint partial_base = slot * head_dim;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_head = head_dim / 32;
    uint blocks_per_position = kv_heads * blocks_per_head;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY; denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += threads) partials[partial_base + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = start; key < end; ++key) {
        float partial = 0.0f;
        uint key_block_base = key * blocks_per_position + kv_head * blocks_per_head;
        for (uint q4_block = simd_group; q4_block < blocks_per_head; q4_block += 4) {
            device const uchar *base = cache + (key_block_base + q4_block) * 18;
            float scale = simd_broadcast(float(*(device const half *)base), 0);
            uchar packed = base[2 + (lane & 15)];
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
            float value = scale * float(int(nibble) - 8);
            uint d = q4_block * 32 + lane;
            partial += query[head * head_dim + d] * value;
        }
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
        uint value_block_base = value_base + key_block_base;
        for (uint q4_block = simd_group; q4_block < blocks_per_head; q4_block += 4) {
            device const uchar *base = cache + (value_block_base + q4_block) * 18;
            float scale = simd_broadcast(float(*(device const half *)base), 0);
            uchar packed = base[2 + (lane & 15)];
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
            float value = scale * float(int(nibble) - 8);
            uint d = q4_block * 32 + lane;
            partials[partial_base + d] = partials[partial_base + d] * rescale + weight * value;
        }
    }
    if (tid == 0) {
        maxima[slot] = maximum;
        sums[slot] = denominator;
    }
}

// Gemma E2B has eight query heads sharing one 512-wide global KV head. This
// opt-in scan keeps
// the production four-way context split and per-head online-softmax order, but
// stages each Q4 key/value block once in threadgroup memory before reusing it
// for all eight query heads. It is deliberately shape-specific: the host only
// selects it for the full-attention 8x512 shape and one shared KV head. The
// 8x256 sliding-window shape remains on the baseline scan.
kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_1_gqa(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *partials [[buffer(2)]], device float *maxima [[buffer(3)]],
    device float *sums [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant uint &key_count [[buffer(9)]],
    constant uint &blocks [[buffer(10)]], uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint query_heads = 8;
    constexpr uint q4_blocks_per_head = 16;
    constexpr uint q4_bytes_per_block = 18;
    constexpr uint q4_packed_bytes_per_block = 16;
    if (heads != query_heads || kv_heads != 1 || head_dim != 512 || block >= blocks) return;
    uint start = block * key_count / blocks;
    uint end = (block + 1) * key_count / blocks;
    uint blocks_per_position = q4_blocks_per_head;
    uint value_base = capacity * blocks_per_position;
    threadgroup half key_scales[q4_blocks_per_head], value_scales[q4_blocks_per_head];
    threadgroup uchar key_packed[q4_blocks_per_head * q4_packed_bytes_per_block];
    threadgroup uchar value_packed[q4_blocks_per_head * q4_packed_bytes_per_block];
    threadgroup float simd_sums[4];
    threadgroup float maximum[query_heads], denominator[query_heads];
    threadgroup float rescale[query_heads], weight[query_heads];

    for (uint head = 0; head < query_heads; ++head) {
        uint partial_base = (block * query_heads + head) * head_dim;
        for (uint d = tid; d < head_dim; d += 128) partials[partial_base + d] = 0.0f;
    }
    if (tid < query_heads) {
        maximum[tid] = -INFINITY;
        denominator[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint key = start; key < end; ++key) {
        uint key_block_base = key * blocks_per_position;
        if (tid < q4_blocks_per_head)
            key_scales[tid] = *(device const half *)(cache + (key_block_base + tid) * q4_bytes_per_block);
        for (uint packed_index = tid;
             packed_index < q4_blocks_per_head * q4_packed_bytes_per_block;
             packed_index += 128) {
            uint q4_block = packed_index / q4_packed_bytes_per_block;
            uint byte = packed_index % q4_packed_bytes_per_block;
            key_packed[packed_index] = cache[(key_block_base + q4_block) * q4_bytes_per_block + 2 + byte];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Finish each head's score in the same four-SIMD-group order as the
        // production kernel before calculating its online-softmax state.
        for (uint head = 0; head < query_heads; ++head) {
            float partial = 0.0f;
            for (uint q4_block = simd_group; q4_block < q4_blocks_per_head; q4_block += 4) {
                uchar packed = key_packed[q4_block * q4_packed_bytes_per_block + (lane & 15)];
                uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
                uint d = q4_block * 32 + lane;
                partial += query[head * head_dim + d]
                    * float(key_scales[q4_block]) * float(int(nibble) - 8);
            }
            float simd_total = simd_sum(partial);
            if (lane == 0) simd_sums[simd_group] = simd_total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float score = simd_sums[0] + simd_sums[1] + simd_sums[2] + simd_sums[3];
                if (score > maximum[head]) {
                    rescale[head] = exp(maximum[head] - score);
                    weight[head] = 1.0f;
                    maximum[head] = score;
                    denominator[head] = denominator[head] * rescale[head] + weight[head];
                } else {
                    rescale[head] = 1.0f;
                    weight[head] = exp(score - maximum[head]);
                    denominator[head] += weight[head];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint value_block_base = value_base + key_block_base;
        if (tid < q4_blocks_per_head)
            value_scales[tid] = *(device const half *)(cache + (value_block_base + tid) * q4_bytes_per_block);
        for (uint packed_index = tid;
             packed_index < q4_blocks_per_head * q4_packed_bytes_per_block;
             packed_index += 128) {
            uint q4_block = packed_index / q4_packed_bytes_per_block;
            uint byte = packed_index % q4_packed_bytes_per_block;
            value_packed[packed_index] = cache[(value_block_base + q4_block) * q4_bytes_per_block + 2 + byte];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint head = 0; head < query_heads; ++head) {
            uint partial_base = (block * query_heads + head) * head_dim;
            for (uint q4_block = simd_group; q4_block < q4_blocks_per_head; q4_block += 4) {
                uchar packed = value_packed[q4_block * q4_packed_bytes_per_block + (lane & 15)];
                uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
                uint d = q4_block * 32 + lane;
                float value = float(value_scales[q4_block]) * float(int(nibble) - 8);
                partials[partial_base + d] = partials[partial_base + d] * rescale[head]
                    + weight[head] * value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < query_heads) {
        uint slot = block * query_heads + tid;
        maxima[slot] = maximum[tid];
        sums[slot] = denominator[tid];
    }
}

// Parallel shared-KV candidate for Gemma's 8-query-head, 1-KV-head global
// attention. Unlike the older _gqa experiment, every query head owns one
// SIMD group: the group cooperatively stages each Q4 key/value position once,
// then all eight heads consume it concurrently. This is intentionally limited
// to the 8x512 shape selected by the Resident executor.
kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_1_mqa_tiled(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *partials [[buffer(2)]], device float *maxima [[buffer(3)]],
    device float *sums [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant uint &key_count [[buffer(9)]],
    constant uint &blocks [[buffer(10)]], uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint query_heads = 8;
    constexpr uint q4_blocks_per_head = 16;
    constexpr uint q4_bytes_per_block = 18;
    constexpr uint q4_packed_bytes_per_block = 16;
    constexpr uint head_width = 512;
    if (heads != query_heads || kv_heads != 1 || head_dim != head_width || block >= blocks
        || simd_group >= query_heads) return;

    uint start = block * key_count / blocks;
    uint end = (block + 1) * key_count / blocks;
    uint value_base = capacity * q4_blocks_per_head;
    threadgroup half key_scales[q4_blocks_per_head], value_scales[q4_blocks_per_head];
    threadgroup uchar key_packed[q4_blocks_per_head * q4_packed_bytes_per_block];
    threadgroup uchar value_packed[q4_blocks_per_head * q4_packed_bytes_per_block];
    threadgroup float maximum[query_heads], denominator[query_heads];
    threadgroup float rescale[query_heads], weight[query_heads];

    for (uint index = tid; index < query_heads * head_width; index += 256) {
        uint head = index / head_width;
        uint d = index % head_width;
        partials[(block * query_heads + head) * head_width + d] = 0.0f;
    }
    if (tid < query_heads) {
        maximum[tid] = -INFINITY;
        denominator[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint head = simd_group;
    uint partial_base = (block * query_heads + head) * head_width;
    for (uint key = start; key < end; ++key) {
        uint key_block_base = key * q4_blocks_per_head;
        if (tid < q4_blocks_per_head)
            key_scales[tid] = *(device const half *)(cache + (key_block_base + tid) * q4_bytes_per_block);
        if (tid < q4_blocks_per_head * q4_packed_bytes_per_block) {
            uint q4_block = tid / q4_packed_bytes_per_block;
            uint byte = tid % q4_packed_bytes_per_block;
            key_packed[tid] = cache[(key_block_base + q4_block) * q4_bytes_per_block + 2 + byte];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score_partial = 0.0f;
        for (uint q4_block = 0; q4_block < q4_blocks_per_head; ++q4_block) {
            uchar packed = key_packed[q4_block * q4_packed_bytes_per_block + (lane & 15)];
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
            uint d = q4_block * 32 + lane;
            score_partial += query[head * head_width + d]
                * float(key_scales[q4_block]) * float(int(nibble) - 8);
        }
        float score = simd_sum(score_partial);
        if (lane == 0) {
            if (score > maximum[head]) {
                rescale[head] = exp(maximum[head] - score);
                weight[head] = 1.0f;
                maximum[head] = score;
                denominator[head] = denominator[head] * rescale[head] + weight[head];
            } else {
                rescale[head] = 1.0f;
                weight[head] = exp(score - maximum[head]);
                denominator[head] += weight[head];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint value_block_base = value_base + key_block_base;
        if (tid < q4_blocks_per_head)
            value_scales[tid] = *(device const half *)(cache + (value_block_base + tid) * q4_bytes_per_block);
        if (tid < q4_blocks_per_head * q4_packed_bytes_per_block) {
            uint q4_block = tid / q4_packed_bytes_per_block;
            uint byte = tid % q4_packed_bytes_per_block;
            value_packed[tid] = cache[(value_block_base + q4_block) * q4_bytes_per_block + 2 + byte];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint q4_block = 0; q4_block < q4_blocks_per_head; ++q4_block) {
            uchar packed = value_packed[q4_block * q4_packed_bytes_per_block + (lane & 15)];
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
            uint d = q4_block * 32 + lane;
            float value = float(value_scales[q4_block]) * float(int(nibble) - 8);
            partials[partial_base + d] = partials[partial_base + d] * rescale[head]
                + weight[head] * value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        uint slot = block * query_heads + head;
        maxima[slot] = maximum[head];
        sums[slot] = denominator[head];
    }
}

kernel void attention_decode_fused_gemma4_simd_q4_0_2pass_2(
    device const float *partials [[buffer(0)]], device const float *maxima [[buffer(1)]],
    device const float *sums [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &blocks [[buffer(6)]], uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    if (head >= heads) return;
    threadgroup float maximum, denominator;
    if (tid == 0) {
        maximum = -INFINITY;
        for (uint block = 0; block < blocks; ++block) {
            maximum = max(maximum, maxima[block * heads + head]);
        }
        denominator = 0.0f;
        for (uint block = 0; block < blocks; ++block) {
            uint slot = block * heads + head;
            denominator += sums[slot] * exp(maxima[slot] - maximum);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < head_dim; d += threads) {
        float value = 0.0f;
        for (uint block = 0; block < blocks; ++block) {
            uint slot = block * heads + head;
            value += partials[slot * head_dim + d] * exp(maxima[slot] - maximum);
        }
        output[head * head_dim + d] = value / denominator;
    }
}

// Q4 cache-dequantization experiment.  This retains the production
// 128-thread/four-SIMD-group geometry and accumulation order, but each SIMD
// group handles complete 32-value Q4 blocks.  The block scale is loaded once
// by lane zero and broadcast to the group instead of being loaded once per
// value lane.
kernel void attention_decode_fused_gemma4_simd_q4_0_cacheopt(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (head >= heads) return;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_head = head_dim / 32;
    uint blocks_per_position = kv_heads * blocks_per_head;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight;
    maximum = -INFINITY;
    denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += 128) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = 0; key < key_count; ++key) {
        float partial = 0.0f;
        uint key_block_base = (key * blocks_per_position) + kv_head * blocks_per_head;
        for (uint block = simd_group; block < blocks_per_head; block += 4) {
            uint block_index = key_block_base + block;
            device const uchar *base = cache + block_index * 18;
            float scale = simd_broadcast(float(*(device const half *)base), 0);
            uint index = block * 32 + lane;
            uchar packed = base[2 + (lane & 15)];
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
            partial += query[head * head_dim + index]
                * scale * float(int(nibble) - 8);
        }
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
        uint value_block_base = capacity * blocks_per_position + key_block_base;
        for (uint block = simd_group; block < blocks_per_head; block += 4) {
            uint block_index = value_block_base + block;
            device const uchar *base = cache + block_index * 18;
            float scale = simd_broadcast(float(*(device const half *)base), 0);
            uint index = block * 32 + lane;
            uchar packed = base[2 + (lane & 15)];
            uchar nibble = lane < 16 ? packed & 15 : packed >> 4;
            uint output_index = head * head_dim + index;
            output[output_index] = output[output_index] * rescale
                + weight * scale * float(int(nibble) - 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = tid; d < head_dim; d += 128)
        output[head * head_dim + d] /= denominator;
}

// Q4 cache attention variant for one SIMD group per query head.  The regular
// Q4 kernel uses four SIMD groups and synchronizes them for every key because
// it launches 128 threads per head.  A Gemma head is small enough for one
// SIMD group to cover its dimensions with a strided loop, eliminating the
// cross-SIMD reduction and reducing the per-key barrier count while retaining
// the same online-softmax and packed-cache contract.
kernel void attention_decode_fused_gemma4_simd_q4_0_32(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    if (head >= heads) return;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float maximum, denominator, rescale, weight;
    maximum = -INFINITY;
    denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += 32) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = 0; key < key_count; ++key) {
        float partial = 0.0f;
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim;
        for (uint d = tid; d < head_dim; d += 32) {
            uint index = key_element + d;
            partial += query[head * head_dim + d]
                * kv_q4_0_value(cache + (index / 32) * 18, index % 32);
        }
        float score = simd_sum(partial);
        if (tid == 0) {
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
        uint value_offset = value_base + key_element;
        for (uint d = tid; d < head_dim; d += 32) {
            uint index = key_element + d;
            output[head * head_dim + d] = output[head * head_dim + d] * rescale
                + weight * kv_q4_0_value(cache + (value_offset + d) / 32 * 18, d % 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = tid; d < head_dim; d += 32)
        output[head * head_dim + d] /= denominator;
}

// Two-SIMD-group Q4 attention experiment.  This keeps more parallelism than
// the 32-thread variant while reducing the four-group reduction used by the
// production 128-thread kernel.
kernel void attention_decode_fused_gemma4_simd_q4_0_64(
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &key_count [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (head >= heads) return;
    uint kv_head = head / (heads / kv_heads);
    uint blocks_per_position = (kv_heads * head_dim) / 32;
    uint value_base = capacity * blocks_per_position;
    threadgroup float simd_sums[2], maximum, denominator, rescale, weight;
    maximum = -INFINITY;
    denominator = 0.0f;
    for (uint d = tid; d < head_dim; d += 64) output[head * head_dim + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = 0; key < key_count; ++key) {
        float partial = 0.0f;
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim;
        for (uint d = tid; d < head_dim; d += 64) {
            uint index = key_element + d;
            partial += query[head * head_dim + d]
                * kv_q4_0_value(cache + (index / 32) * 18, index % 32);
        }
        float simd_total = simd_sum(partial);
        if (lane == 0) simd_sums[simd_group] = simd_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float score = simd_sums[0] + simd_sums[1];
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
        uint value_offset = value_base + key_element;
        for (uint d = tid; d < head_dim; d += 64) {
            uint index = key_element + d;
            output[head * head_dim + d] = output[head * head_dim + d] * rescale
                + weight * kv_q4_0_value(cache + (value_offset + d) / 32 * 18, d % 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = tid; d < head_dim; d += 64)
        output[head * head_dim + d] /= denominator;
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
