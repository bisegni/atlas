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

// Token-batched PLE composition.  lhs is the current layer's gate slice
// `[token][0..ple_size]`, rhs is the resident `[token][layer][dim]` PLE table,
// and the output slice is `[token][0..ple_size]`.  One dispatch per layer
// covers the whole token chunk (replacing the previous per-token loop), so the
// per-element multiply is bitwise identical.
kernel void vector_multiply_offset_batch_f32(
    device const float *lhs [[buffer(0)]], device const float *rhs [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rhs_offset [[buffer(3)]],
    constant uint &layers [[buffer(4)]], constant uint &ple_size [[buffer(5)]],
    constant uint &batch [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    uint token = id / ple_size;
    uint lane = id % ple_size;
    if (token >= batch) return;
    output[token * ple_size + lane] = lhs[token * ple_size + lane] * rhs[token * layers * ple_size + rhs_offset + lane];
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

// Decode PLE-per-layer epilogue fusion (phase-13.9, dispatch reduction).
// The current per-layer PLE tail is three elementwise dispatches over the
// hidden row:
//   rms_norm_decode_f32_vec4(work, post_norm)          // -> normalized
//   vector_add_f32(state, normalized)                  // -> state
//   scalar_multiply_f32(state, layer_output_scale)     // -> state
// This kernel fuses them into one dispatch, saving 2 launches per layer
// (~70 of the ~547 decode dispatches/token).  It retains the exact
// rms_norm_decode_f32_vec4 reduction and the volatile `normalized`
// rounding boundary (same as gemma4_rms_residual_f32), then computes
//   state = (state + normalized) * layer_output_scale
// elementwise, which is bitwise-identical to the three-kernel baseline.
kernel void gemma4_ple_rms_add_scale_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    // state doubles as the residual source and the scaled output.
    device volatile float *state [[buffer(2)]],
    device volatile float *normalized [[buffer(3)]],
    constant uint &hidden [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    device const float *scale [[buffer(6)]],
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
    float s = *scale;
    for (uint tile = 0; tile < vector_tiles; ++tile) {
        uint offset = tile * 128 + lane * 4;
        float4 n = *(device const float4 *)(normalized + offset);
        float4 r = *(device const float4 *)(state + offset);
        *(device float4 *)(state + offset) = (r + n) * s;
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

// Batched prefill GEMM with a weight-stationary batch tile (phase 13.6 /
// phase 13.8 step 1).  The reference (matmul_q4_0_batch_16row) reads each
// weight row once per token, so a pp512 prompt reads the whole weight matrix
// 512 times.  This kernel instead has each threadgroup compute
// GEMMA4_BATCH_TILE_TOKENS tokens for its 32 output rows, reading the shared
// weight block once and reusing it in registers across those tokens, with one
// independent accumulator chain per token for ILP.  Each token's dot product
// accumulates block-sequentially with the identical `in * q * scale`
// expression and shuffle_xor(4,2,1) butterfly; under the Path B tolerance
// contract the interleaved chains are covered by batch_matmul_parity.rs.
// Out-of-range token pointers are clamped so reads stay in bounds; their
// results are discarded by the per-token write guard.
#define GEMMA4_BATCH_TILE_TOKENS 8
#define GEMMA4_BATCH_DECL_ACC(n) float sum##n = 0.0f;
#define GEMMA4_BATCH_DECL_TIN(n) \
    device const float *tin##n = input + min(base_token + n, last) * input_width;
#define GEMMA4_BATCH_FMA(n) \
    sum##n += tin##n[off] * float(int(packed0 & 15) - 8) * scale; \
    sum##n += tin##n[off + 8] * float(int(packed1 & 15) - 8) * scale; \
    sum##n += tin##n[off + 16] * float(int(packed0 >> 4) - 8) * scale; \
    sum##n += tin##n[off + 24] * float(int(packed1 >> 4) - 8) * scale;
#define GEMMA4_BATCH_BUTTERFLY(n) \
    sum##n += simd_shuffle_xor(sum##n, 4); \
    sum##n += simd_shuffle_xor(sum##n, 2); \
    sum##n += simd_shuffle_xor(sum##n, 1);
#define GEMMA4_BATCH_WRITE(n) \
    if (base_token + n < batch) output[(base_token + n) * output_width + row] = sum##n;
#define GEMMA4_BATCH_SLOTS(M) \
    M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
kernel void matmul_q4_0_batch_32row(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], constant uint &batch [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_in_simd = lane / 8;
    uint column = lane % 8;
    uint rows_per_batch = (output_width + 31) / 32;
    uint token_group = group / rows_per_batch;
    uint group_row = group % rows_per_batch;
    uint row = group_row * 32 + simdgroup * 4 + row_in_simd;
    uint base_token = token_group * GEMMA4_BATCH_TILE_TOKENS;
    uint last = batch - 1;
    GEMMA4_BATCH_SLOTS(GEMMA4_BATCH_DECL_ACC)
    if (row < output_width) {
        uint blocks = input_width / 32;
        GEMMA4_BATCH_SLOTS(GEMMA4_BATCH_DECL_TIN)
        for (uint block = 0; block < blocks; ++block) {
            device const uchar *base = weights + (row * blocks + block) * 18;
            float scale = float(*(device const half *)base);
            uchar packed0 = base[2 + column];
            uchar packed1 = base[2 + column + 8];
            uint off = block * 32 + column;
            GEMMA4_BATCH_SLOTS(GEMMA4_BATCH_FMA)
        }
    }
    GEMMA4_BATCH_SLOTS(GEMMA4_BATCH_BUTTERFLY)
    if (column == 0 && row < output_width) {
        GEMMA4_BATCH_SLOTS(GEMMA4_BATCH_WRITE)
    }
}





// Phase B (Path B extension): tuned 32-token x 64-row fp16 matrix-unit GEMM.
// Each threadgroup stages one 64-dim K-chunk of q4_0-dequantized weights (fp16)
// and the 32-token input slice (cast to fp16) in threadgroup memory (one
// barrier per chunk), then loops 4 token-sub-tiles x 8 k-sub-chunks with
// simdgroup_multiply_accumulate (fp16 inputs, fp32 accumulate).  The dequant is
// amortized over 32 tokens and weight DRAM traffic is read once per threadgroup
// (8x less than the 8-token scalar tile).  NOT within the max-abs 1e-3 contract
// (fp16 cast); Phase B accepts ~1e-2 for prefill speed.  Requires batch % 32
// == 0, output_width % 64 == 0, input_width % 64 == 0 (executor falls back to
// matmul_q4_0_batch_32row otherwise).  Correct simdgroup combo (pinned down
// 2026-08-17 after the original load/store flags + per-chunk accumulators
// were found to compute a garbled last-chunk-only result): a-load
// transpose=false, b-load transpose=true, fp32 accumulators persist across
// K-chunks, single transpose=false store per sub-tile.  Semantics details in
// docs/plan-close-prefill-gap.md.
inline float simd_q4_0_dequant(device const uchar *weights, uint row, uint blocks, uint dim) {
    uint block = dim / 32;
    uint within = dim % 32;
    device const uchar *base = weights + (row * blocks + block) * 18;
    float scale = float(*(device const half *)base);
    uchar packed = base[2 + (within & 15)];
    uchar nibble = within < 16 ? packed & 15 : packed >> 4;
    return scale * float(int(nibble) - 8);
}
kernel void matmul_q4_0_batch_mm64(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], constant uint &batch [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint TOKENS = 32;
    const uint ROWS = 64;
    const uint CHUNK = 64;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row_tiles = output_width / ROWS;
    uint token_tile = group / row_tiles;
    uint row_tile = group % row_tiles;
    uint base_token = token_tile * TOKENS;
    uint base_row = row_tile * ROWS;
    uint blocks = input_width / 32;
    threadgroup half wb[ROWS * CHUNK];
    threadgroup half ib[TOKENS * CHUNK];
    // Persistent per-token-subtile accumulators: every K-chunk adds into them
    // and the output is written once after the full reduction.  (An earlier
    // revision reset and stored these inside the chunk loop, silently keeping
    // only the last chunk's contribution.)  The loops over `d` must stay
    // fully unrolled so every simdgroup-matrix index is a compile-time
    // constant; dynamic indexing into a simdgroup-matrix array does not
    // lower to valid fragment storage.
    simdgroup_float8x8 d[TOKENS / 8];
    _Pragma("clang loop unroll(full)") for (uint tt = 0; tt < TOKENS / 8; ++tt) {
        d[tt] = simdgroup_float8x8(0.0f);
    }
    for (uint kc = 0; kc < input_width; kc += CHUNK) {
        for (uint e = 0; e < ROWS * CHUNK / 256; ++e) {
            uint idx = tid * (ROWS * CHUNK / 256) + e;
            uint r = idx / CHUNK;
            uint k = idx % CHUNK;
            uint row = base_row + r;
            wb[idx] = (row < output_width) ? half(simd_q4_0_dequant(weights, row, blocks, kc + k)) : 0.0f;
        }
        for (uint e = 0; e < TOKENS * CHUNK / 256; ++e) {
            uint idx = tid * (TOKENS * CHUNK / 256) + e;
            uint t = idx / CHUNK;
            uint k = idx % CHUNK;
            uint token = base_token + t;
            ib[idx] = (token < batch) ? half(input[token * input_width + kc + k]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)") for (uint tt = 0; tt < TOKENS / 8; ++tt) {
            for (uint sub = 0; sub < CHUNK / 8; ++sub) {
                uint k_off = sub * 8;
                simdgroup_half8x8 a, b;
                // Probe-verified contraction (artifacts/mm64-layout + impulse
                // test): both blocks staged row-major, load flags (a=false,
                // b=true) give d[r][c] = sum_k A[r][k]*B[c][k]; a
                // transpose=false store then writes it row-major.  Here A is
                // the token slice (r=token, k=dim) and B the weight slice
                // (c=row), so the memory cell is output[token][row].  The
                // original (true, false) loads with a transpose=true store
                // computed a garbled, transposed contraction.
                simdgroup_load(a, ib + tt * 8 * CHUNK + k_off, CHUNK, ulong2(0, 0), false);
                simdgroup_load(b, wb + simdgroup * 8 * CHUNK + k_off, CHUNK, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(d[tt], a, b, d[tt]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    _Pragma("clang loop unroll(full)") for (uint tt = 0; tt < TOKENS / 8; ++tt) {
        simdgroup_store(
            d[tt],
            output + (base_token + tt * 8) * output_width + base_row + simdgroup * 8,
            output_width,
            ulong2(0, 0),
            false);
    }
}

// llama.cpp-style fp16 matrix-unit mul_mm (Path B opt-in, `ATLAS_GEMMA4_MUL_MM`).
// The two prep passes turn the two GPU-hostile inputs into fp16 fragments that
// this GEMM can feed straight to the matrix units with NO threadgroup weight
// staging and NO per-K-chunk dequant barrier:
//   - gemma4_q4_0_to_f16_batch  dequantizes the resident row-major GGUF q4_0
//     tensor into a contiguous fp16 layer buffer `[out][in]`;
//   - gemma4_cast_f32_to_f16_batch casts the fp32 activation slice to fp16.
// matmul_q4_0_batch_f16 then uses llama.cpp's persistent-accumulator structure:
// each threadgroup covers GEMMA4_MUL_MM_TOKENS=32 tokens x 32 output rows and
// keeps GEMMA4_MUL_MM_TOKENS/8 independent fp32 simdgroup accumulators across
// the whole K reduction, loading each 8-row weight sub-block into the matrix
// units once and reusing it across the 32 tokens.  A 512-token prompt therefore
// reads the weight matrix 512/32 = 16 times (vs 64 for an 8-token tile) while
// charging no per-chunk dequant/barrier -- the contraction (a-load
// transpose=false, b-load transpose=true, transpose=false store) is the
// probe-verified one from `matmul_q4_0_batch_mm64` (docs/plan-close-prefill-gap.md).
// This is tolerance-level (fp16 inputs => ~1e-4 relative error vs fp32),
// llama.cpp's own accuracy level, so the executor keeps it opt-in (`ATLAS_GEMMA4_MUL_MM`)
// and leaves the fp32 scalar tile as the max-abs < 1e-3 default.
#define GEMMA4_MUL_MM_ROWS 32   // output rows per threadgroup (4 simdgroups x 8)
#define GEMMA4_MUL_MM_TOKENS 32 // tokens per threadgroup (4 x 8-token subtiles)
kernel void gemma4_q4_0_to_f16_batch(
    device const uchar *q4 [[buffer(0)]],
    device half *f16 [[buffer(1)]],
    constant uint &input_width [[buffer(2)]],
    constant uint &output_width [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    // One thread per output row; each thread dequantizes all of that row's
    // q4_0 blocks into fp16.  Layout matches `dequantize_block` in atlas-core.
    if (id >= output_width) return;
    uint blocks = input_width / 32;
    device half *dst = f16 + id * input_width;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar *base = q4 + (id * blocks + b) * 18;
        half scale = *(device const half *)base;
        for (uint i = 0; i < 16; ++i) {
            uchar packed = base[2 + i];
            dst[b * 32 + i]      = half(float(int(packed & 15) - 8) * float(scale));
            dst[b * 32 + i + 16] = half(float(int(packed >> 4) - 8) * float(scale));
        }
    }
}

kernel void gemma4_cast_f32_to_f16_batch(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &input_width [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    uint count = batch * input_width;
    if (id >= count) return;
    output[id] = half(input[id]);
}

kernel void matmul_q4_0_batch_f16(
    device const half *act_f16 [[buffer(0)]],
    device const half *weights_f16 [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]],
    constant uint &batch [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint grid_row_tiles = (output_width + GEMMA4_MUL_MM_ROWS - 1) / GEMMA4_MUL_MM_ROWS;
    const uint token_tile = group / grid_row_tiles;
    const uint row_tile = group % grid_row_tiles;
    const uint base_token = token_tile * GEMMA4_MUL_MM_TOKENS;
    const uint base_row = row_tile * GEMMA4_MUL_MM_ROWS;
    const uint simdgroup = tid / 32;
    const uint sg_row = base_row + simdgroup * 8;
    // Executor gate guarantees batch % GEMMA4_MUL_MM_TOKENS == 0,
    // output_width % GEMMA4_MUL_MM_ROWS == 0 and input_width % 8 == 0, so every
    // load/store stays in bounds.  d[] must stay fully unrolled so every
    // fragment index is a compile-time constant (dynamic indexing into a
    // simdgroup-matrix array does not lower to valid fragment storage).
    simdgroup_float8x8 d[GEMMA4_MUL_MM_TOKENS / 8];
    _Pragma("clang loop unroll(full)") for (uint tt = 0; tt < GEMMA4_MUL_MM_TOKENS / 8; ++tt) {
        d[tt] = simdgroup_float8x8(0.0f);
    }
    for (uint k0 = 0; k0 + 8 <= input_width; k0 += 8) {
        simdgroup_half8x8 b;
        simdgroup_load(b, weights_f16 + sg_row * input_width + k0, input_width, ulong2(0, 0), true);
        _Pragma("clang loop unroll(full)") for (uint tt = 0; tt < GEMMA4_MUL_MM_TOKENS / 8; ++tt) {
            simdgroup_half8x8 a;
            simdgroup_load(a, act_f16 + (base_token + tt * 8) * input_width + k0, input_width, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(d[tt], a, b, d[tt]);
        }
    }
    _Pragma("clang loop unroll(full)") for (uint tt = 0; tt < GEMMA4_MUL_MM_TOKENS / 8; ++tt) {
        simdgroup_store(
            d[tt],
            output + (base_token + tt * 8) * output_width + sg_row,
            output_width,
            ulong2(0, 0),
            false);
    }
}

kernel void matmul_f16_batch(    device const float *input [[buffer(0)]], device const half *weights [[buffer(1)]],
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

// Token-batched prefill variant of gemma4_qk_norm_rope_fused_f32.  The grid is
// `batch * (q_heads + has_key)`; one threadgroup per (token, head) keeps the
// exact per-token scalar reduction order, so every output byte is identical to
// the per-token dispatches it replaces.  Per-token strides differ per buffer
// (q_width vs kv width vs rope_pairs), so the kernel derives its own offsets
// from the threadgroup index instead of per-buffer dispatch offsets.
kernel void gemma4_qk_norm_rope_fused_batch_f32(
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
    constant uint &batch [[buffer(12)]],
    constant uint &rope_pairs [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint total = q_heads + has_key;
    if (tid != 0 || group >= batch * total) return;
    uint token = group / total;
    uint local = group % total;
    bool key = local >= q_heads;
    uint head = key ? 0 : local;
    uint token_stride = key ? head_dim : q_heads * head_dim;
    device const float *input = key ? k_input : q_input;
    device const float *weight = key ? k_weight : q_weight;
    device float *output = key ? k_output : q_output;
    uint base = token * token_stride + head * head_dim;
    uint rope_base = token * rope_pairs;
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
        float c = cosine[rope_base + pair];
        float s = sine[rope_base + pair];
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

// Token-batched KV append for prefill.  The grid covers `batch * blocks`; each
// thread derives its token from the flat index and reads that token's absolute
// position from the contiguous positions table, so the packed cache bytes are
// identical to the per-token dispatches it replaces.
kernel void kv_append_decode_f32_batch(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device float *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], device const uint *positions [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    uint blocks = kv_width;
    uint token = id / blocks;
    uint slot = id % blocks;
    uint base = token * kv_width;
    uint position = positions[token];
    if (slot < kv_width && position < capacity) {
        cache[position * kv_width + slot] = key[base + slot];
        cache[capacity * kv_width + position * kv_width + slot] = value[base + slot];
    }
}

kernel void kv_append_decode_q8_0_batch(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device uchar *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], device const uint *positions [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    uint blocks = kv_width / 32;
    uint token = id / blocks;
    uint block = id % blocks;
    uint base = token * kv_width + block * 32;
    uint position = positions[token];
    if (block >= blocks || position >= capacity) return;
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

kernel void kv_append_decode_q4_0_batch(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device uchar *cache [[buffer(2)]], constant uint &kv_width [[buffer(3)]],
    constant uint &capacity [[buffer(4)]], device const uint *positions [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    uint blocks = kv_width / 32;
    uint token = id / blocks;
    uint block = id % blocks;
    uint base = token * kv_width + block * 32;
    uint position = positions[token];
    if (block >= blocks || position >= capacity) return;
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

// Token-batched prefill variants of the Resident SIMD attention kernels.  The
// grid is `batch * heads`; each threadgroup derives its token from the flat
// index and reads that token's key control from the contiguous per-token table
// (buffer passed at the layer byte offset, stride `layers`), so each
// (token, head) threadgroup performs the identical runtime Q·K accumulation,
// four-SIMD reduction, and key-ordered online softmax as the per-token
// dispatches it replaces.
#define BATCH_KV_Q8_READ(cache, index) kv_q8_0_value(cache + ((index) / 32) * 34, (index) % 32)
#define BATCH_KV_Q4_READ(cache, index) kv_q4_0_value(cache + ((index) / 32) * 18, (index) % 32)
#define DEFINE_SIMD_ATTENTION_BATCH(NAME, KV_READ) \
kernel void NAME( \
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], device const uint *key_control [[buffer(7)]], \
    constant uint &layers [[buffer(8)]], \
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    uint token = group / heads; \
    uint head = group % heads; \
    if (head >= heads) return; \
    uint control = key_control[token * layers]; \
    uint key_start = control >> 16; \
    uint key_count = control & 0xffffu; \
    uint kv_head = head / (heads / kv_heads); \
    uint blocks_per_position = (kv_heads * head_dim) / 32; \
    uint value_base = capacity * blocks_per_position; \
    uint base = token * heads * head_dim; \
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight; \
    maximum = -INFINITY; denominator = 0.0f; \
    for (uint d = tid; d < head_dim; d += threads) output[base + head * head_dim + d] = 0.0f; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint key_offset = 0; key_offset < key_count; ++key_offset) { \
        uint key = key_start + key_offset; \
        float partial = 0.0f; \
        uint key_element = key * kv_heads * head_dim + kv_head * head_dim; \
        for (uint d = tid; d < head_dim; d += threads) { \
            uint index = key_element + d; \
            partial += query[base + head * head_dim + d] * KV_READ(cache, index); \
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
            output[base + head * head_dim + d] = output[base + head * head_dim + d] * rescale \
                + weight * KV_READ(cache, value_base * 32 + index); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (uint d = tid; d < head_dim; d += threads) output[base + head * head_dim + d] /= denominator; \
}

#define DEFINE_F32_SIMD_ATTENTION_BATCH(NAME) \
kernel void NAME( \
    device const float *query [[buffer(0)]], device const float *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], device const uint *key_control [[buffer(7)]], \
    constant uint &layers [[buffer(8)]], \
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    uint token = group / heads; \
    uint head = group % heads; \
    if (head >= heads) return; \
    uint control = key_control[token * layers]; \
    uint key_start = control >> 16; \
    uint key_count = control & 0xffffu; \
    uint kv_head = head / (heads / kv_heads); \
    uint value_base = capacity * kv_heads * head_dim; \
    uint base = token * heads * head_dim; \
    threadgroup float simd_sums[4], maximum, denominator, rescale, weight; \
    maximum = -INFINITY; denominator = 0.0f; \
    for (uint d = tid; d < head_dim; d += threads) output[base + head * head_dim + d] = 0.0f; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint key_offset = 0; key_offset < key_count; ++key_offset) { \
        uint key = key_start + key_offset; \
        float partial = 0.0f; \
        uint key_base = key * kv_heads * head_dim + kv_head * head_dim; \
        for (uint d = tid; d < head_dim; d += threads) \
            partial += query[base + head * head_dim + d] * cache[key_base + d]; \
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
        uint value_offset = value_base + key_base; \
        for (uint d = tid; d < head_dim; d += threads) \
            output[base + head * head_dim + d] = output[base + head * head_dim + d] * rescale \
                + weight * cache[value_offset + d]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (uint d = tid; d < head_dim; d += threads) output[base + head * head_dim + d] /= denominator; \
}

DEFINE_F32_SIMD_ATTENTION_BATCH(attention_decode_fused_gemma4_simd_f32_batch)
DEFINE_SIMD_ATTENTION_BATCH(attention_decode_fused_gemma4_simd_q8_0_batch, BATCH_KV_Q8_READ)
DEFINE_SIMD_ATTENTION_BATCH(attention_decode_fused_gemma4_simd_q4_0_batch, BATCH_KV_Q4_READ)

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

// Flash16 v3 (gap-analysis D2): staged, chunked, exact-ordered KV scan.  The
// reference kernels scan keys serially with two threadgroup barriers per key
// and a device-memory output round trip per key.  This version keeps the exact
// FP32 arithmetic -- per-thread dims d = t128 + k*128 within each 128-thread
// score group, the same simd_sum, the same p0+p1+p2+p3 fold, the same
// key-ordered online softmax, and the same per-dim value chain -- but stages
// the per-key SIMD partials and the online-softmax rescale/weight sequence
// through threadgroup memory so the score reduction covers the full key range
// under a single running maximum/denominator (no slice merging, no rounding
// change).  A chunk of FLASH16_V3_CHUNK keys is scanned in three
// barrier-separated passes: (A) all threads compute raw per-key partials with
// `threads / 128` keys in parallel per iteration (the query is cached in
// registers), (B) thread 0 folds and runs the online softmax, (C) all threads
// apply the register-resident value chain spread over the full threadgroup
// width.  Per-key barriers drop from ~2 to ~3 per chunk, the value accumulator
// stays in registers, and the wide threadgroup gives several independent key
// chains in flight, so the output is byte-identical to the reference while
// memory latency is hidden across keys and threads.  Requires threads to be a
// positive multiple of 128 (128/256/512 per the resident dispatch); head_dim
// is runtime so Metal cannot unroll or reassociate the partial dot products.
#define FLASH16_V3_CHUNK 128
#define DEFINE_FLASH16_EXACT_V3(NAME) \
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
    if (key_count == 0) { \
        for (uint d = tid; d < head_dim; d += threads) output[head * head_dim + d] = 0.0f; \
        return; \
    } \
    threadgroup float partials[FLASH16_V3_CHUNK][4]; \
    threadgroup float rescale[FLASH16_V3_CHUNK]; \
    threadgroup float weight[FLASH16_V3_CHUNK]; \
    threadgroup float maximum, denominator; \
    maximum = -INFINITY; denominator = 0.0f; \
    const uint wide = threads / 128; \
    const uint t_key = tid / 128; \
    const uint t128 = tid % 128; \
    uint head_base = head * head_dim; \
    float q0 = query[head_base + t128]; \
    float q1 = (t128 + 128 < head_dim) ? query[head_base + t128 + 128] : 0.0f; \
    float q2 = (t128 + 256 < head_dim) ? query[head_base + t128 + 256] : 0.0f; \
    float q3 = (t128 + 384 < head_dim) ? query[head_base + t128 + 384] : 0.0f; \
    float q4 = (t128 + 512 < head_dim) ? query[head_base + t128 + 512] : 0.0f; \
    float q5 = (t128 + 640 < head_dim) ? query[head_base + t128 + 640] : 0.0f; \
    float q6 = (t128 + 768 < head_dim) ? query[head_base + t128 + 768] : 0.0f; \
    float q7 = (t128 + 896 < head_dim) ? query[head_base + t128 + 896] : 0.0f; \
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f; \
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f; \
    uint key_offset = 0; \
    while (key_offset < key_count) { \
        uint chunk = (key_count - key_offset) < FLASH16_V3_CHUNK ? (key_count - key_offset) : FLASH16_V3_CHUNK; \
        for (uint c = 0; c < chunk; c += wide) { \
            uint batch = (chunk - c) < wide ? (chunk - c) : wide; \
            if (t_key < batch) { \
                uint key = key_start + key_offset + c + t_key; \
                float partial = 0.0f; \
                uint key_element = key * kv_heads * head_dim + kv_head * head_dim; \
                uint d = t128; \
                partial += q0 * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32); \
                d += 128; \
                if (d < head_dim) partial += q1 * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32); \
                d += 128; \
                if (d < head_dim) partial += q2 * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32); \
                d += 128; \
                if (d < head_dim) partial += q3 * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32); \
                d += 128; \
                if (d < head_dim) partial += q4 * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32); \
                d += 128; \
                if (d < head_dim) partial += q5 * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32); \
                d += 128; \
                if (d < head_dim) partial += q6 * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32); \
                d += 128; \
                if (d < head_dim) partial += q7 * kv_q4_0_value(cache + ((key_element + d) / 32) * 18, (key_element + d) % 32); \
                float simd_total = simd_sum(partial); \
                if (lane == 0) partials[c + t_key][simd_group % 4] = simd_total; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (tid == 0) { \
            for (uint c = 0; c < chunk; ++c) { \
                float score = partials[c][0] + partials[c][1] + partials[c][2] + partials[c][3]; \
                float r, w; \
                if (score > maximum) { \
                    r = exp(maximum - score); w = 1.0f; maximum = score; denominator = denominator * r + w; \
                } else { \
                    r = 1.0f; w = exp(score - maximum); denominator += w; \
                } \
                rescale[c] = r; weight[c] = w; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint c = 0; c < chunk; ++c) { \
            uint key = key_start + key_offset + c; \
            float r = rescale[c]; \
            float w = weight[c]; \
            uint key_element = key * kv_heads * head_dim + kv_head * head_dim; \
            uint d = tid; \
            acc0 = acc0 * r + w * kv_q4_0_value(cache + (value_base + (key_element + d) / 32) * 18, (key_element + d) % 32); \
            d += threads; \
            if (d < head_dim) acc1 = acc1 * r + w * kv_q4_0_value(cache + (value_base + (key_element + d) / 32) * 18, (key_element + d) % 32); \
            d += threads; \
            if (d < head_dim) acc2 = acc2 * r + w * kv_q4_0_value(cache + (value_base + (key_element + d) / 32) * 18, (key_element + d) % 32); \
            d += threads; \
            if (d < head_dim) acc3 = acc3 * r + w * kv_q4_0_value(cache + (value_base + (key_element + d) / 32) * 18, (key_element + d) % 32); \
            d += threads; \
            if (d < head_dim) acc4 = acc4 * r + w * kv_q4_0_value(cache + (value_base + (key_element + d) / 32) * 18, (key_element + d) % 32); \
            d += threads; \
            if (d < head_dim) acc5 = acc5 * r + w * kv_q4_0_value(cache + (value_base + (key_element + d) / 32) * 18, (key_element + d) % 32); \
            d += threads; \
            if (d < head_dim) acc6 = acc6 * r + w * kv_q4_0_value(cache + (value_base + (key_element + d) / 32) * 18, (key_element + d) % 32); \
            d += threads; \
            if (d < head_dim) acc7 = acc7 * r + w * kv_q4_0_value(cache + (value_base + (key_element + d) / 32) * 18, (key_element + d) % 32); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        key_offset += chunk; \
    } \
    { \
        uint d = tid; \
        if (d < head_dim) output[head_base + d] = acc0 / denominator; \
        d += threads; \
        if (d < head_dim) output[head_base + d] = acc1 / denominator; \
        d += threads; \
        if (d < head_dim) output[head_base + d] = acc2 / denominator; \
        d += threads; \
        if (d < head_dim) output[head_base + d] = acc3 / denominator; \
        d += threads; \
        if (d < head_dim) output[head_base + d] = acc4 / denominator; \
        d += threads; \
        if (d < head_dim) output[head_base + d] = acc5 / denominator; \
        d += threads; \
        if (d < head_dim) output[head_base + d] = acc6 / denominator; \
        d += threads; \
        if (d < head_dim) output[head_base + d] = acc7 / denominator; \
    } \
}

DEFINE_FLASH16_EXACT_V3(attention_decode_gemma4_simd_q4_0_flash16_exact_v3)
DEFINE_FLASH16_EXACT_V3(attention_decode_gemma4_simd_q4_0_flash16_swa_exact_v3)















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

// Flash16 v4 (Path B): merged-slice flash attention wired for the Resident
// key_control contract.  The "_uw" design (SLICES disjoint key ranges scanned
// with per-simdgroup register online-softmax and value accumulators, then
// merged in threadgroup memory) has no per-key barriers and is several times
// faster than the exact-ordered v3 scan, but it is NOT bitwise: the slice
// split and merge change the FP32 reduction order.  Under the Path B tolerance
// contract (max-abs < 1e-3) this is the intended decode default.  Unlike the
// "_uw" kernels it reads the packed `key_control` (key_start << 16 | key_count)
// so sliding-window heads scan the correct absolute key range.
#define DEFINE_FLASH_ATTENTION_V4(NAME, HEAD_DIM, BLOCKS, SLICES) \
kernel void NAME( \
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], constant uint &key_control [[buffer(7)]], \
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    if (head >= heads) return; \
    if (head_dim != HEAD_DIM) return; \
    if (simd_group >= SLICES) return; \
    uint key_start = key_control >> 16; \
    uint key_count = key_control & 0xffffu; \
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
    uint start = key_start + simd_group * key_count / SLICES; \
    uint end = key_start + (simd_group + 1) * key_count / SLICES; \
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
        for (uint g = 0; g < SLICES; ++g) value += merg_out[g * HEAD_DIM + dim] * exp(merg_max[g] - global_max); \
        output[head * HEAD_DIM + dim] = value / global_sum; \
    } \
}

DEFINE_FLASH_ATTENTION_V4(attention_decode_gemma4_simd_q4_0_flash16_v4, 512, 16, 12)

// Batched prefill variant of Flash16 v4 (phase-13.11): identical merged-slice
// design, but the grid is batch*heads (one threadgroup per (token, head)) and
// each threadgroup reads its token's packed key_control entry
// (key_start<<16 | key_count) from the contiguous per-token table (stride
// `layers`).  This replaces the serial per-key, per-key-barrier prefill scan
// (attention_decode_fused_gemma4_simd_q4_0_batch) with the barrier-free
// merged-slice flash scan for every prompt token in one dispatch.  Like the
// decode v4 kernel it is tolerance-level (the slice split + merge changes the
// FP32 reduction order), not bitwise.
#define DEFINE_FLASH_ATTENTION_V4_BATCH(NAME, HEAD_DIM, BLOCKS, SLICES) \
kernel void NAME( \
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], device const uint *key_control [[buffer(7)]], \
    constant uint &layers [[buffer(8)]], \
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    uint token = group / heads; \
    uint head = group % heads; \
    if (head >= heads) return; \
    if (head_dim != HEAD_DIM) return; \
    if (simd_group >= SLICES) return; \
    uint control = key_control[token * layers]; \
    uint key_start = control >> 16; \
    uint key_count = control & 0xffffu; \
    uint kv_head = head / (heads / kv_heads); \
    uint blocks_per_position = kv_heads * BLOCKS; \
    uint value_base = capacity * blocks_per_position; \
    uint qbase = token * heads * HEAD_DIM + head * HEAD_DIM; \
    if (key_count == 0) { \
        for (uint b = 0; b < BLOCKS; ++b) { \
            uint dim = 32 * b + lane; \
            if (dim < HEAD_DIM) output[qbase + dim] = 0.0f; \
        } \
        return; \
    } \
    uint start = key_start + simd_group * key_count / SLICES; \
    uint end = key_start + (simd_group + 1) * key_count / SLICES; \
    float maximum = -INFINITY; \
    float denominator = 0.0f; \
    FLASH_ACC_DECLS \
    float q_cache[BLOCKS]; \
    for (uint qb = 0; qb < BLOCKS; ++qb) { \
        uint qdim = 32 * qb + lane; \
        q_cache[qb] = query[qbase + qdim]; \
    } \
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
        for (uint g = 0; g < SLICES; ++g) value += merg_out[g * HEAD_DIM + dim] * exp(merg_max[g] - global_max); \
        output[qbase + dim] = value / global_sum; \
    } \
}

DEFINE_FLASH_ATTENTION_V4_BATCH(attention_prefill_gemma4_simd_q4_0_flash16_v4, 512, 16, 12)
DEFINE_FLASH_ATTENTION_V4_BATCH(attention_prefill_gemma4_simd_q4_0_flash16_swa_v4, 256, 8, 24)

// Third-generation batched prefill attention. The v4 kernels dispatch one
// threadgroup per (token, head) and, because Gemma 4 E2B uses a single shared
// KV head (kv_heads == 1), every one of the `heads` threadgroups for a token
// re-dequantizes the SAME key/value cache. This variant dispatches one
// threadgroup per token with `heads` SIMD groups (one per head); the K/V q4_0
// dequantization for each key is done ONCE cooperatively into threadgroup
// memory and shared across all heads, removing the per-head dequant redundancy.
// Requires kv_heads == 1 (all heads share one KV head) and heads == 8.
#define FLASH_ACC_STORE_V5(NB, B) \
    if (B < NB) { uint dim = B * 32 + lane; output[qbase + dim] = acc##B / denominator; }
#define FLASH_ACC_STORES_V5(NB) \
    FLASH_ACC_STORE_V5(NB, 0) FLASH_ACC_STORE_V5(NB, 1) FLASH_ACC_STORE_V5(NB, 2) FLASH_ACC_STORE_V5(NB, 3) \
    FLASH_ACC_STORE_V5(NB, 4) FLASH_ACC_STORE_V5(NB, 5) FLASH_ACC_STORE_V5(NB, 6) FLASH_ACC_STORE_V5(NB, 7) \
    FLASH_ACC_STORE_V5(NB, 8) FLASH_ACC_STORE_V5(NB, 9) FLASH_ACC_STORE_V5(NB, 10) FLASH_ACC_STORE_V5(NB, 11) \
    FLASH_ACC_STORE_V5(NB, 12) FLASH_ACC_STORE_V5(NB, 13) FLASH_ACC_STORE_V5(NB, 14) FLASH_ACC_STORE_V5(NB, 15)

#define DEFINE_FLASH_ATTENTION_V5_BATCH(NAME, HEAD_DIM, BLOCKS) \
kernel void NAME( \
    device const float *query [[buffer(0)]], device const uchar *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], device const uint *key_control [[buffer(7)]], \
    constant uint &layers [[buffer(8)]], \
    uint token [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    if (head_dim != HEAD_DIM) return; \
    uint head = simd_group; \
    if (head >= heads) return; \
    uint control = key_control[token * layers]; \
    uint key_start = control >> 16; \
    uint key_count = control & 0xffffu; \
    uint kv_head = head / (heads / kv_heads); \
    uint blocks_per_position = kv_heads * BLOCKS; \
    uint value_base = capacity * blocks_per_position; \
    uint qbase = token * heads * HEAD_DIM + head * HEAD_DIM; \
    threadgroup float k_shared[HEAD_DIM]; \
    threadgroup float v_shared[HEAD_DIM]; \
    float maximum = -INFINITY; \
    float denominator = 0.0f; \
    FLASH_ACC_DECLS \
    float q_cache[BLOCKS]; \
    for (uint qb = 0; qb < BLOCKS; ++qb) { \
        uint qdim = 32 * qb + lane; \
        q_cache[qb] = query[qbase + qdim]; \
    } \
    if (key_count == 0) { \
        for (uint b = 0; b < BLOCKS; ++b) { \
            uint dim = 32 * b + lane; \
            output[qbase + dim] = 0.0f; \
        } \
        return; \
    } \
    for (uint key = key_start; key < key_start + key_count; ++key) { \
        uint key_block_base = key * blocks_per_position + kv_head * BLOCKS; \
        uint value_block_base = value_base + key_block_base; \
        for (uint i = tid; i < HEAD_DIM; i += 256) { \
            uint block = i / 32; \
            uint within = i % 32; \
            device const uchar *base = cache + (key_block_base + block) * 18; \
            float scale = float(*(device const half *)base); \
            uchar packed = base[2 + (within & 15)]; \
            uchar nibble = within < 16 ? packed & 15 : packed >> 4; \
            k_shared[i] = scale * float(int(nibble) - 8); \
        } \
        for (uint i = tid; i < HEAD_DIM; i += 256) { \
            uint block = i / 32; \
            uint within = i % 32; \
            device const uchar *base = cache + (value_block_base + block) * 18; \
            float scale = float(*(device const half *)base); \
            uchar packed = base[2 + (within & 15)]; \
            uchar nibble = within < 16 ? packed & 15 : packed >> 4; \
            v_shared[i] = scale * float(int(nibble) - 8); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        float partial = 0.0f; \
        for (uint b = 0; b < BLOCKS; ++b) { \
            uint dim = 32 * b + lane; \
            partial += q_cache[b] * k_shared[dim]; \
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
        for (uint b = 0; b < BLOCKS; ++b) { \
            uint dim = 32 * b + lane; \
            float value = v_shared[dim]; \
            if (b == 0) acc0 = acc0 * rescale + weight * value; \
            FLASH_ACC_UPDATES \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    FLASH_ACC_STORES_V5(BLOCKS) \
}

DEFINE_FLASH_ATTENTION_V5_BATCH(attention_prefill_gemma4_simd_q4_0_flash16_v5, 512, 16)
DEFINE_FLASH_ATTENTION_V5_BATCH(attention_prefill_gemma4_simd_q4_0_flash16_swa_v5, 256, 8)

// Flash16 v6 (phase-13.14 Lever 1): matrix-unit prefill attention.  One
// threadgroup per 8-token tile with one SIMD group per head (heads == 8,
// kv_heads == 1, the Gemma 4 E2B geometry).  The K/V q4_0 dequant for each
// KEY_BLOCK=16 keys is done once cooperatively into f16 threadgroup memory and
// shared across all heads (the phase-13.13 v5 property), but the per-key
// compute is no longer the serial simd_sum/exp chain: S = Q·K^T and O += P·V
// are simdgroup_matrix multiplies that batch the whole key block onto the
// matrix units.  Q is read from a pre-cast fp16 query buffer (see
// gemma4_cast_f32_to_f16_batch).  Softmax runs in two passes over the key
// range: pass 1 reduces the per-row max/denominator (scalar online-softmax
// stats only), pass 2 recomputes S, forms the already-normalized
// P = exp(S - M) / D, and accumulates O += P·V with no per-row rescaling of the
// fragment accumulators.  Tolerance-level (fp16 inputs + reordered reduction),
// the same class as the decode Flash16 v4/v5 paths.  Reads the packed
// key_control table (key_start << 16 | key_count) so causality and
// sliding-window heads are masked per token row.
#define DEFINE_FLASH_ATTENTION_V6_BATCH(NAME, HEAD_DIM) \
kernel void NAME( \
    device const half *query [[buffer(0)]], device const uchar *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], device const uint *key_control [[buffer(7)]], \
    constant uint &layers [[buffer(8)]], constant uint &batch [[buffer(9)]], \
    uint token_tile [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    const uint TOKENS = 8; \
    const uint KEY_BLOCK = 16; \
    const uint BLOCKS = HEAD_DIM / 32; \
    if (head_dim != HEAD_DIM) return; \
    if (simd_group >= heads) return; \
    threadgroup half tg_kv[KEY_BLOCK * HEAD_DIM]; \
    threadgroup float tg_s[8 * TOKENS * KEY_BLOCK]; \
    threadgroup half tg_p[8 * TOKENS * KEY_BLOCK]; \
    threadgroup uint tg_control[TOKENS]; \
    threadgroup float tg_m[8 * TOKENS]; \
    threadgroup float tg_d[8 * TOKENS]; \
    uint base_token = token_tile * TOKENS; \
    uint head = simd_group; \
    uint kv_head = head / (heads / kv_heads); \
    uint blocks_per_position = kv_heads * BLOCKS; \
    uint value_base = capacity * blocks_per_position; \
    uint qbase = head * HEAD_DIM; \
    uint q_stride = heads * HEAD_DIM; \
    uint sbase = head * (TOKENS * KEY_BLOCK); \
    uint stat = head * TOKENS; \
    if (simd_group == 0 && lane < TOKENS) { \
        uint token = base_token + lane; \
        tg_control[lane] = (token < batch) ? key_control[token * layers] : 0u; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint min_start = 0xffffffffu; \
    uint max_end = 0u; \
    for (uint r = 0; r < TOKENS; ++r) { \
        uint token = base_token + r; \
        if (token >= batch) continue; \
        uint control = tg_control[r]; \
        uint ks = control >> 16; \
        uint kc = control & 0xffffu; \
        if (kc == 0) continue; \
        min_start = min(min_start, ks); \
        max_end = max(max_end, ks + kc); \
    } \
    uint my_key_start = 0u; \
    uint my_key_count = 0u; \
    if (lane < TOKENS) { \
        uint control = tg_control[lane]; \
        my_key_start = control >> 16; \
        my_key_count = control & 0xffffu; \
    } \
    bool my_valid = (lane < TOKENS) && (base_token + lane < batch) && (my_key_count > 0u); \
    if (max_end <= min_start) { \
        for (uint idx = tid; idx < TOKENS * HEAD_DIM; idx += 256) { \
            uint r = idx / HEAD_DIM; \
            uint dim = idx % HEAD_DIM; \
            if (base_token + r < batch) output[(base_token + r) * q_stride + qbase + dim] = 0.0f; \
        } \
        return; \
    } \
    if (lane < TOKENS) { \
        tg_m[stat + lane] = -INFINITY; \
        tg_d[stat + lane] = 0.0f; \
    } \
    /* Pass 1: online softmax statistics (per-row max and denominator). */ \
    for (uint kb = min_start; kb < max_end; kb += KEY_BLOCK) { \
        for (uint idx = tid; idx < KEY_BLOCK * HEAD_DIM; idx += 256) { \
            uint key = idx / HEAD_DIM; \
            uint dim = idx % HEAD_DIM; \
            uint gkey = kb + key; \
            if (gkey < max_end) { \
                uint block = dim / 32; \
                uint within = dim % 32; \
                device const uchar *base = cache + (gkey * blocks_per_position + kv_head * BLOCKS + block) * 18; \
                float scale = float(*(device const half *)base); \
                uchar packed = base[2 + (within & 15)]; \
                uchar nibble = within < 16 ? packed & 15 : packed >> 4; \
                tg_kv[key * HEAD_DIM + dim] = half(scale * float(int(nibble) - 8)); \
            } else { \
                tg_kv[key * HEAD_DIM + dim] = half(0.0f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_float8x8 s0 = simdgroup_float8x8(0.0f); \
        simdgroup_float8x8 s1 = simdgroup_float8x8(0.0f); \
        _Pragma("clang loop unroll(full)") for (uint koff = 0; koff < HEAD_DIM; koff += 8) { \
            simdgroup_half8x8 qa, kb0, kb1; \
            simdgroup_load(qa, query + base_token * q_stride + qbase + koff, q_stride, ulong2(0, 0), false); \
            simdgroup_load(kb0, tg_kv + koff, HEAD_DIM, ulong2(0, 0), true); \
            simdgroup_load(kb1, tg_kv + 8 * HEAD_DIM + koff, HEAD_DIM, ulong2(0, 0), true); \
            simdgroup_multiply_accumulate(s0, qa, kb0, s0); \
            simdgroup_multiply_accumulate(s1, qa, kb1, s1); \
        } \
        simdgroup_store(s0, tg_s + sbase + 0, KEY_BLOCK, ulong2(0, 0), false); \
        simdgroup_store(s1, tg_s + sbase + 8, KEY_BLOCK, ulong2(0, 0), false); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (my_valid) { \
            float m_local = -INFINITY; \
            _Pragma("unroll") \
            for (uint c = 0; c < KEY_BLOCK; ++c) { \
                uint gkey = kb + c; \
                if (gkey >= my_key_start && gkey < my_key_start + my_key_count) { \
                    m_local = max(m_local, tg_s[sbase + lane * KEY_BLOCK + c]); \
                } \
            } \
            float l_local = 0.0f; \
            _Pragma("unroll") \
            for (uint c = 0; c < KEY_BLOCK; ++c) { \
                uint gkey = kb + c; \
                if (gkey >= my_key_start && gkey < my_key_start + my_key_count) { \
                    l_local += exp(tg_s[sbase + lane * KEY_BLOCK + c] - m_local); \
                } \
            } \
            float m_old = tg_m[stat + lane]; \
            float m_new = max(m_old, m_local); \
            if (m_local == -INFINITY) { \
                /* no valid keys in this block; stats unchanged */ \
            } else if (m_old == -INFINITY) { \
                tg_d[stat + lane] = l_local; \
            } else { \
                tg_d[stat + lane] = tg_d[stat + lane] * exp(m_old - m_new) + l_local * exp(m_local - m_new); \
            } \
            tg_m[stat + lane] = m_new; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    /* Pass 2: normalized P = exp(S - M) / D and O += P·V. */ \
    simdgroup_float8x8 o[HEAD_DIM / 8]; \
    _Pragma("clang loop unroll(full)") for (uint dt = 0; dt < HEAD_DIM / 8; ++dt) o[dt] = simdgroup_float8x8(0.0f); \
    for (uint kb = min_start; kb < max_end; kb += KEY_BLOCK) { \
        for (uint idx = tid; idx < KEY_BLOCK * HEAD_DIM; idx += 256) { \
            uint key = idx / HEAD_DIM; \
            uint dim = idx % HEAD_DIM; \
            uint gkey = kb + key; \
            if (gkey < max_end) { \
                uint block = dim / 32; \
                uint within = dim % 32; \
                device const uchar *base = cache + (gkey * blocks_per_position + kv_head * BLOCKS + block) * 18; \
                float scale = float(*(device const half *)base); \
                uchar packed = base[2 + (within & 15)]; \
                uchar nibble = within < 16 ? packed & 15 : packed >> 4; \
                tg_kv[key * HEAD_DIM + dim] = half(scale * float(int(nibble) - 8)); \
            } else { \
                tg_kv[key * HEAD_DIM + dim] = half(0.0f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_float8x8 s0 = simdgroup_float8x8(0.0f); \
        simdgroup_float8x8 s1 = simdgroup_float8x8(0.0f); \
        _Pragma("clang loop unroll(full)") for (uint koff = 0; koff < HEAD_DIM; koff += 8) { \
            simdgroup_half8x8 qa, kb0, kb1; \
            simdgroup_load(qa, query + base_token * q_stride + qbase + koff, q_stride, ulong2(0, 0), false); \
            simdgroup_load(kb0, tg_kv + koff, HEAD_DIM, ulong2(0, 0), true); \
            simdgroup_load(kb1, tg_kv + 8 * HEAD_DIM + koff, HEAD_DIM, ulong2(0, 0), true); \
            simdgroup_multiply_accumulate(s0, qa, kb0, s0); \
            simdgroup_multiply_accumulate(s1, qa, kb1, s1); \
        } \
        simdgroup_store(s0, tg_s + sbase + 0, KEY_BLOCK, ulong2(0, 0), false); \
        simdgroup_store(s1, tg_s + sbase + 8, KEY_BLOCK, ulong2(0, 0), false); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (lane < TOKENS) { \
            float m = tg_m[stat + lane]; \
            float d = tg_d[stat + lane]; \
            bool valid = my_valid && (d > 0.0f); \
            _Pragma("unroll") \
            for (uint c = 0; c < KEY_BLOCK; ++c) { \
                uint gkey = kb + c; \
                float p = 0.0f; \
                if (valid && gkey >= my_key_start && gkey < my_key_start + my_key_count) { \
                    p = exp(tg_s[sbase + lane * KEY_BLOCK + c] - m) / d; \
                } \
                tg_p[sbase + lane * KEY_BLOCK + c] = half(p); \
            } \
        } \
        for (uint idx = tid; idx < KEY_BLOCK * HEAD_DIM; idx += 256) { \
            uint key = idx / HEAD_DIM; \
            uint dim = idx % HEAD_DIM; \
            uint gkey = kb + key; \
            if (gkey < max_end) { \
                uint block = dim / 32; \
                uint within = dim % 32; \
                device const uchar *base = cache + (value_base + gkey * blocks_per_position + kv_head * BLOCKS + block) * 18; \
                float scale = float(*(device const half *)base); \
                uchar packed = base[2 + (within & 15)]; \
                uchar nibble = within < 16 ? packed & 15 : packed >> 4; \
                tg_kv[key * HEAD_DIM + dim] = half(scale * float(int(nibble) - 8)); \
            } else { \
                tg_kv[key * HEAD_DIM + dim] = half(0.0f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_half8x8 pa0, pa1; \
        simdgroup_load(pa0, tg_p + sbase + 0, KEY_BLOCK, ulong2(0, 0), false); \
        simdgroup_load(pa1, tg_p + sbase + 8, KEY_BLOCK, ulong2(0, 0), false); \
        _Pragma("clang loop unroll(full)") for (uint dt = 0; dt < HEAD_DIM / 8; ++dt) { \
            simdgroup_half8x8 vb0, vb1; \
            simdgroup_load(vb0, tg_kv + dt * 8, HEAD_DIM, ulong2(0, 0), false); \
            simdgroup_load(vb1, tg_kv + 8 * HEAD_DIM + dt * 8, HEAD_DIM, ulong2(0, 0), false); \
            simdgroup_multiply_accumulate(o[dt], pa0, vb0, o[dt]); \
            simdgroup_multiply_accumulate(o[dt], pa1, vb1, o[dt]); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    _Pragma("clang loop unroll(full)") for (uint dt = 0; dt < HEAD_DIM / 8; ++dt) { \
        simdgroup_store(o[dt], output + base_token * q_stride + qbase + dt * 8, q_stride, ulong2(0, 0), false); \
    } \
}

DEFINE_FLASH_ATTENTION_V6_BATCH(attention_prefill_gemma4_simd_q4_0_flash16_v6, 512)
DEFINE_FLASH_ATTENTION_V6_BATCH(attention_prefill_gemma4_simd_q4_0_flash16_swa_v6, 256)

// Flash16-v7: single-pass matrix-unit prefill attention with online softmax
// rescaling.  Same tiling, buffers and math as v6 (one threadgroup per 8-token
// tile, one SIMD group per head, heads == 8, kv_heads == 1), but each key
// block is dequantized only once for K and once for V and S = Q·K^T is
// computed once, not twice: the per-row block max merges into the running max
// with denominator rescaling, the O accumulator fragments are rescaled by
// exp(m_old - m_new) only when the max actually moves, and the final
// normalization divides the accumulator once by the denominator rows.
//
// Metal exposes no scalar element access for simdgroup matrices
// (thread_elements() writable indices do not connect to the matrix register
// file on this toolchain - verified empirically with load/store probes), so
// the per-row rescale is expressed as a public-API matrix multiply: the row
// scale factors form a diagonal fragment L built in threadgroup memory, and
// each O fragment becomes o[dt] = L·o[dt] (a full 8x8 fp32 simdgroup MAC).
// The rescale fires only when at least one row's max actually moved, which
// happens only in the early key blocks of a causal window; steady-state blocks
// add P·V with zero per-row scaling cost.  The final division by the
// denominator rows uses the same diagonal-multiply form.  Rows with a zero
// denominator stay zero.
#define DEFINE_FLASH_ATTENTION_V7_BATCH(NAME, HEAD_DIM) \
kernel void NAME( \
    device const half *query [[buffer(0)]], device const uchar *cache [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &heads [[buffer(3)]], \
    constant uint &kv_heads [[buffer(4)]], constant uint &head_dim [[buffer(5)]], \
    constant uint &capacity [[buffer(6)]], device const uint *key_control [[buffer(7)]], \
    constant uint &layers [[buffer(8)]], constant uint &batch [[buffer(9)]], \
    uint token_tile [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    const uint TOKENS = 8; \
    const uint KEY_BLOCK = 16; \
    const uint BLOCKS = HEAD_DIM / 32; \
    if (head_dim != HEAD_DIM) return; \
    if (simd_group >= heads) return; \
    threadgroup half tg_kv[KEY_BLOCK * HEAD_DIM]; \
    threadgroup float tg_s[8 * TOKENS * KEY_BLOCK]; \
    threadgroup half tg_p[8 * TOKENS * KEY_BLOCK]; \
    threadgroup uint tg_control[TOKENS]; \
    threadgroup float tg_m[8 * TOKENS]; \
    threadgroup float tg_d[8 * TOKENS]; \
    threadgroup float tg_scale[8 * TOKENS]; \
    threadgroup float tg_diag[8 * 8 * 8]; \
    uint base_token = token_tile * TOKENS; \
    uint head = simd_group; \
    uint kv_head = head / (heads / kv_heads); \
    uint blocks_per_position = kv_heads * BLOCKS; \
    uint value_base = capacity * blocks_per_position; \
    uint qbase = head * HEAD_DIM; \
    uint q_stride = heads * HEAD_DIM; \
    uint sbase = head * (TOKENS * KEY_BLOCK); \
    uint stat = head * TOKENS; \
    if (simd_group == 0 && lane < TOKENS) { \
        uint token = base_token + lane; \
        tg_control[lane] = (token < batch) ? key_control[token * layers] : 0u; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint min_start = 0xffffffffu; \
    uint max_end = 0u; \
    for (uint r = 0; r < TOKENS; ++r) { \
        uint token = base_token + r; \
        if (token >= batch) continue; \
        uint control = tg_control[r]; \
        uint ks = control >> 16; \
        uint kc = control & 0xffffu; \
        if (kc == 0) continue; \
        min_start = min(min_start, ks); \
        max_end = max(max_end, ks + kc); \
    } \
    uint my_key_start = 0u; \
    uint my_key_count = 0u; \
    if (lane < TOKENS) { \
        uint control = tg_control[lane]; \
        my_key_start = control >> 16; \
        my_key_count = control & 0xffffu; \
    } \
    bool my_valid = (lane < TOKENS) && (base_token + lane < batch) && (my_key_count > 0u); \
    if (max_end <= min_start) { \
        for (uint idx = tid; idx < TOKENS * HEAD_DIM; idx += 256) { \
            uint r = idx / HEAD_DIM; \
            uint dim = idx % HEAD_DIM; \
            if (base_token + r < batch) output[(base_token + r) * q_stride + qbase + dim] = 0.0f; \
        } \
        return; \
    } \
    simdgroup_float8x8 o[HEAD_DIM / 8]; \
    _Pragma("clang loop unroll(full)") for (uint dt = 0; dt < HEAD_DIM / 8; ++dt) o[dt] = simdgroup_float8x8(0.0f); \
    if (lane < TOKENS) { \
        tg_m[stat + lane] = -INFINITY; \
        tg_d[stat + lane] = 0.0f; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint kb = min_start; kb < max_end; kb += KEY_BLOCK) { \
        for (uint idx = tid; idx < KEY_BLOCK * HEAD_DIM; idx += 256) { \
            uint key = idx / HEAD_DIM; \
            uint dim = idx % HEAD_DIM; \
            uint gkey = kb + key; \
            if (gkey < max_end) { \
                uint block = dim / 32; \
                uint within = dim % 32; \
                device const uchar *base = cache + (gkey * blocks_per_position + kv_head * BLOCKS + block) * 18; \
                float scale = float(*(device const half *)base); \
                uchar packed = base[2 + (within & 15)]; \
                uchar nibble = within < 16 ? packed & 15 : packed >> 4; \
                tg_kv[key * HEAD_DIM + dim] = half(scale * float(int(nibble) - 8)); \
            } else { \
                tg_kv[key * HEAD_DIM + dim] = half(0.0f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_float8x8 s0 = simdgroup_float8x8(0.0f); \
        simdgroup_float8x8 s1 = simdgroup_float8x8(0.0f); \
        _Pragma("clang loop unroll(full)") for (uint koff = 0; koff < HEAD_DIM; koff += 8) { \
            simdgroup_half8x8 qa, kb0, kb1; \
            simdgroup_load(qa, query + base_token * q_stride + qbase + koff, q_stride, ulong2(0, 0), false); \
            simdgroup_load(kb0, tg_kv + koff, HEAD_DIM, ulong2(0, 0), true); \
            simdgroup_load(kb1, tg_kv + 8 * HEAD_DIM + koff, HEAD_DIM, ulong2(0, 0), true); \
            simdgroup_multiply_accumulate(s0, qa, kb0, s0); \
            simdgroup_multiply_accumulate(s1, qa, kb1, s1); \
        } \
        simdgroup_store(s0, tg_s + sbase + 0, KEY_BLOCK, ulong2(0, 0), false); \
        simdgroup_store(s1, tg_s + sbase + 8, KEY_BLOCK, ulong2(0, 0), false); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (lane < TOKENS) { \
            float m_local = -INFINITY; \
            _Pragma("unroll") \
            for (uint c = 0; c < KEY_BLOCK; ++c) { \
                uint gkey = kb + c; \
                if (gkey >= my_key_start && gkey < my_key_start + my_key_count) { \
                    float s = tg_s[sbase + lane * KEY_BLOCK + c]; \
                    m_local = max(m_local, s); \
                } \
            } \
            float m_old = tg_m[stat + lane]; \
            float m_new = max(m_old, m_local); \
            float block_scale = 1.0f; \
            float l_local = 0.0f; \
            if (m_local != -INFINITY) { \
                block_scale = exp(m_old - m_new); \
                _Pragma("unroll") \
                for (uint c = 0; c < KEY_BLOCK; ++c) { \
                    uint gkey = kb + c; \
                    if (gkey >= my_key_start && gkey < my_key_start + my_key_count) { \
                        l_local += exp(tg_s[sbase + lane * KEY_BLOCK + c] - m_new); \
                    } \
                } \
                tg_d[stat + lane] = tg_d[stat + lane] * block_scale + l_local; \
                tg_m[stat + lane] = m_new; \
            } \
            tg_scale[stat + lane] = block_scale; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        bool changed = false; \
        for (uint h = 0; h < heads; ++h) { \
            _Pragma("unroll") \
            for (uint r = 0; r < TOKENS; ++r) changed |= (tg_scale[h * TOKENS + r] != 1.0f); \
        } \
        if (changed) { \
            if (lane < TOKENS) { \
                float scale = tg_scale[stat + lane]; \
                _Pragma("unroll") \
                for (uint c = 0; c < TOKENS; ++c) { \
                    tg_diag[head * 64 + lane * 8 + c] = (c == lane) ? scale : 0.0f; \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            simdgroup_float8x8 diag = simdgroup_float8x8(0.0f); \
            simdgroup_float8x8 zero = simdgroup_float8x8(0.0f); \
            simdgroup_load(diag, tg_diag + head * 64, TOKENS, ulong2(0, 0), false); \
            _Pragma("clang loop unroll(full)") for (uint dt = 0; dt < HEAD_DIM / 8; ++dt) { \
                simdgroup_float8x8 scaled; \
                simdgroup_multiply_accumulate(scaled, diag, o[dt], zero); \
                o[dt] = scaled; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        if (lane < TOKENS) { \
            float m = tg_m[stat + lane]; \
            bool valid = my_valid && (tg_d[stat + lane] > 0.0f); \
            _Pragma("unroll") \
            for (uint c = 0; c < KEY_BLOCK; ++c) { \
                uint gkey = kb + c; \
                float p = 0.0f; \
                if (valid && gkey >= my_key_start && gkey < my_key_start + my_key_count) { \
                    p = exp(tg_s[sbase + lane * KEY_BLOCK + c] - m); \
                } \
                tg_p[sbase + lane * KEY_BLOCK + c] = half(p); \
            } \
        } \
        for (uint idx = tid; idx < KEY_BLOCK * HEAD_DIM; idx += 256) { \
            uint key = idx / HEAD_DIM; \
            uint dim = idx % HEAD_DIM; \
            uint gkey = kb + key; \
            if (gkey < max_end) { \
                uint block = dim / 32; \
                uint within = dim % 32; \
                device const uchar *base = cache + (value_base + gkey * blocks_per_position + kv_head * BLOCKS + block) * 18; \
                float scale = float(*(device const half *)base); \
                uchar packed = base[2 + (within & 15)]; \
                uchar nibble = within < 16 ? packed & 15 : packed >> 4; \
                tg_kv[key * HEAD_DIM + dim] = half(scale * float(int(nibble) - 8)); \
            } else { \
                tg_kv[key * HEAD_DIM + dim] = half(0.0f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_half8x8 pa0, pa1; \
        simdgroup_load(pa0, tg_p + sbase + 0, KEY_BLOCK, ulong2(0, 0), false); \
        simdgroup_load(pa1, tg_p + sbase + 8, KEY_BLOCK, ulong2(0, 0), false); \
        _Pragma("clang loop unroll(full)") for (uint dt = 0; dt < HEAD_DIM / 8; ++dt) { \
            simdgroup_half8x8 vb0, vb1; \
            simdgroup_load(vb0, tg_kv + dt * 8, HEAD_DIM, ulong2(0, 0), false); \
            simdgroup_load(vb1, tg_kv + 8 * HEAD_DIM + dt * 8, HEAD_DIM, ulong2(0, 0), false); \
            simdgroup_multiply_accumulate(o[dt], pa0, vb0, o[dt]); \
            simdgroup_multiply_accumulate(o[dt], pa1, vb1, o[dt]); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    if (lane < TOKENS) { \
        for (uint c = 0; c < TOKENS; ++c) { \
            float d = tg_d[stat + lane]; \
            tg_diag[head * 64 + lane * 8 + c] = (c == lane) ? ((d > 0.0f) ? 1.0f / d : 0.0f) : 0.0f; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    simdgroup_float8x8 diag = simdgroup_float8x8(0.0f); \
    simdgroup_float8x8 zero = simdgroup_float8x8(0.0f); \
    simdgroup_load(diag, tg_diag + head * 64, TOKENS, ulong2(0, 0), false); \
    _Pragma("clang loop unroll(full)") for (uint dt = 0; dt < HEAD_DIM / 8; ++dt) { \
        simdgroup_float8x8 scaled; \
        simdgroup_multiply_accumulate(scaled, diag, o[dt], zero); \
        o[dt] = scaled; \
        simdgroup_store(o[dt], output + base_token * q_stride + qbase + dt * 8, q_stride, ulong2(0, 0), false); \
    } \
}

DEFINE_FLASH_ATTENTION_V7_BATCH(attention_prefill_gemma4_simd_q4_0_flash16_v7, 512)
DEFINE_FLASH_ATTENTION_V7_BATCH(attention_prefill_gemma4_simd_q4_0_flash16_swa_v7, 256)

DEFINE_FLASH_ATTENTION_V4(attention_decode_gemma4_simd_q4_0_flash16_swa_v4, 256, 8, 24)




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

// 16-row-per-threadgroup qkv fusion (phase-13.16 follow-up): the same fuse
// as matmul_q4_0_qkv_32row_mv_rms with a 4-row simdgroup band (4 SIMD groups
// x 4 rows per threadgroup, dispatched with 128 threads and grid
// ceil(q/16) + 2*ceil(kv/16)).  The per-lane yl/RMS/block-dot order is
// unchanged, so every row is bitwise identical to the 32-row kernel's.
kernel void matmul_q4_0_qkv_16row_mv_rms(
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
    uint q_groups = (q_width + 15) / 16;
    uint kv_groups = (kv_width + 15) / 16;
    uint projection = group < q_groups ? 0 : (group < q_groups + kv_groups ? 1 : 2);
    uint local_group = projection == 0 ? group :
        (projection == 1 ? group - q_groups : group - q_groups - kv_groups);
    uint output_width = projection == 0 ? q_width : kv_width;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = local_group * 16 + simdgroup * 4;
    bool active = row < output_width;
    uint blocks = input_width / 32;
    uint ix = lane / 2;
    uint il = (lane % 2) * 8;
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq = 0.0f;
    device const uchar *weights = projection == 0 ? q_weights :
        (projection == 1 ? k_weights : v_weights);
    device const uchar *ax[4];
    for (uint r = 0; r < 4; ++r) {
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
            for (uint r = 0; r < 4; ++r) {
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
    for (uint r = 0; r < 4; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 4; ++r) {
            uint out_row = row + r;
            if (out_row >= output_width) continue;
            float value = sumf[r] * inverse_rms;
            if (projection == 0) q_output[out_row] = value;
            else if (projection == 1) k_output[out_row] = value;
            else v_output[out_row] = value;
        }
    }
}

// 16-row-per-threadgroup gate/up fusion (phase-13.16 follow-up): 4 SIMD
// groups x 4 rows per threadgroup, grid 2*ceil(output_width/16); bitwise
// identical to matmul_q4_0_gate_up_32row_mv_rms per row.
kernel void matmul_q4_0_gate_up_16row_mv_rms(
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
    uint groups = (output_width + 15) / 16;
    uint projection = group < groups ? 0 : 1;
    uint local_group = projection == 0 ? group : group - groups;
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = local_group * 16 + simdgroup * 4;
    bool active = row < output_width;
    uint blocks = input_width / 32;
    uint ix = lane / 2;
    uint il = (lane % 2) * 8;
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq = 0.0f;
    device const uchar *weights = projection == 0 ? gate_weights : up_weights;
    device const uchar *ax[4];
    for (uint r = 0; r < 4; ++r) {
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
            for (uint r = 0; r < 4; ++r) {
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
    for (uint r = 0; r < 4; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 4; ++r) {
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

// 16-row-per-threadgroup q4_0 matvec matching llama.cpp's current
// mul_mat_vec granularity (128 threads, 4 SIMD groups x 4 rows per
// threadgroup, grid `output_width / 16`).  The per-lane block stride (ib +=
// 16), y-cache (`yl`) and q4 block-dot arithmetic are identical to the
// 64-row family above, so each lane accumulates the exact same values in
// the exact same order and the produced rows are bitwise identical to
// `matvec_q4_0_64row_mv`'s.  Four times the threadgroups of the 64-row
// variant; used for the small-M decode matvecs (ffn-down, attention-output,
// PLE) where the 64-row dispatch is occupancy-limited (production default,
// opt out with `ATLAS_GEMMA4_DECODE_16ROW=0`).
kernel void matvec_q4_0_16row_mv(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = group * 16 + simdgroup * 4;
    bool active = row < output_width;
    uint blocks = input_width / 32;
    uint ix = lane / 2;
    uint il = (lane % 2) * 8;
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const uchar *ax[4];
    for (uint r = 0; r < 4; ++r) {
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
            for (uint r = 0; r < 4; ++r) {
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
    for (uint r = 0; r < 4; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 4; ++r) {
            uint out_row = row + r;
            if (out_row < output_width) output[out_row] = sumf[r];
        }
    }
}

// RMS-input counterpart of matvec_q4_0_16row_mv with the same per-lane
// sum-of-squares and yl-scale order as the 64-row `_rms` kernel.
kernel void matvec_q4_0_16row_mv_rms(
    device const float *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &input_width [[buffer(3)]],
    constant uint &output_width [[buffer(4)]],
    device const float *rms_weight [[buffer(5)]],
    constant float &epsilon [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint simdgroup = tid / 32;
    uint lane = tid % 32;
    uint row = group * 16 + simdgroup * 4;
    bool active = row < output_width;
    uint blocks = input_width / 32;
    uint ix = lane / 2;
    uint il = (lane % 2) * 8;
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq = 0.0f;
    device const uchar *ax[4];
    for (uint r = 0; r < 4; ++r) {
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
            for (uint r = 0; r < 4; ++r) {
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
    for (uint r = 0; r < 4; ++r) sumf[r] = simd_sum(sumf[r]);
    if (lane == 0) {
        for (uint r = 0; r < 4; ++r) {
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

// ============================================================================
// Vendored from llama.cpp — MIT License.
//
// Faithful port of llama.cpp's classic (simdgroup_load-based) `kernel_mul_mm`
// (ggml/src/ggml-metal/ggml-metal.metal), specialized for q4_0 weights x f32
// activations, a single batch, and contiguous row-major tensors. Only the
// host/device ABI is adapted (scalar constants instead of ggml's kargs
// struct); the compute core is llama.cpp's: a small 64x32 output tile per
// threadgroup, dequantization of a 64x32 weight tile + a 32x32 activation
// tile into threadgroup memory each K-step, then simdgroup_matrix
// multiply-accumulate over K. This is the "fill the matrix" design that
// realizes the matrix-unit throughput (as opposed to dequantizing the whole
// weight into a device buffer first).
//
// MIT License
// Copyright (c) 2023-2025 ggml-org / llama.cpp authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ============================================================================

typedef struct {
    half  d;        // delta (scale)
    uchar qs[16];   // nibbles / quants
} llama_block_q4_0;

static inline void llama_dequantize_q4_0(
        device const llama_block_q4_0 * xb,
        short il,
        thread half4x4 & reg) {
    device const ushort * qs = ((device const ushort *)xb + 1);
    const float d1 = il ? ((float)xb->d / 16.f) : (float)xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.f * (float)xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (float)(qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (float)(qs[i] & mask1) + md;
    }
    reg = (half4x4)reg_f;
}

// C(M,N) = A(M,K) @ B(K,N) where A = src0 is the q4_0 weight [M=out x K],
// B = src1 is the f32 activation laid out [N=tokens x K] (so B(k,n) reads
// src1[n*K + k]), and dst is written [N x M] (dst[n*M + m]). Grid:
// tgpig.x = ceil(N/32), tgpig.y = ceil(M/64), depth 1. 128 threads/threadgroup.
kernel void llama_mul_mm_q4_0_f32(
        device const uchar * src0,
        device const float * src1,
        device       float * dst,
        constant uint & ne00,   // K
        constant uint & ne0,    // M (output rows)
        constant uint & ne1,    // N (tokens)
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;   // 2
    constexpr int NL1 = NK/8;    // 4
    constexpr short nl = 2;      // q4_0

    const uint nb01 = (ne00/32)*18;   // weight row stride (bytes)
    const uint nb10 = 4;              // activation element stride (bytes)
    const uint nb11 = ne00*4;         // activation row stride (bytes)

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = ((int)ne0 - r0 < NR0) ? ((int)ne0 - r0) : NR0;
    const short nr1 = ((int)ne1 - r1 < NR1) ? ((int)ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const short offset1 = il0/nl;   // 0 for q4_0

    device const llama_block_q4_0 * x =
        (device const llama_block_q4_0 *)(src0 + nb01*(r0 + lr0)) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(
        (device const uchar *)src1 + nb11*(r1 + lr1) + nb10*iy);

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < (int)ne00; loop_k += NK) {
        // A: dequantize the q4 weight tile into threadgroup memory
        half4x4 temp_a;
        llama_dequantize_q4_0(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        // B: load the f32 activation tile into threadgroup memory (K % NK == 0)
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) =
                (half2x4)(*((device float2x4 *)y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load fragments from threadgroup memory and accumulate outer products
        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        #pragma unroll
        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= (int)ne0 && r1 + NR1 <= (int)ne1) {
        device float * C = (device float *)dst +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*ne0*(i/4), ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str =
            ((threadgroup float *)shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *)dst + r0 + (r1 + j)*ne0;
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

// C(M,N) = A(M,K) @ B(K,N) where A = src0 is the fp16 weight [M x K],
// B = src1 is the f32 activation [N x K] (so B(k,n) reads src1[n*K + k]), and
// dst is written [N x M] (dst[n*M + m]). Same tile/threadgroup geometry as
// llama_mul_mm_q4_0_f32 (64x32 output tile, 4 simdgroups, NK=32), but the A
// operand is fp16 with NO dequantization: a direct half4x4 load (llama.cpp's
// dequantize_f16, MIT-attributed as above). Grid: tgpig.x = ceil(N/32),
// tgpig.y = ceil(M/64). 128 threads/threadgroup.
kernel void llama_mul_mm_f16_f32(
        device const half  * src0,
        device const float * src1,
        device       float * dst,
        constant uint & ne00,   // K
        constant uint & ne0,    // M (output rows)
        constant uint & ne1,    // N (tokens)
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;   // 2
    constexpr int NL1 = NK/8;    // 4

    const uint nb01 = ne00*2;    // weight row stride (bytes, f16)
    const uint nb10 = 4;         // activation element stride (bytes)
    const uint nb11 = ne00*4;    // activation row stride (bytes)

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = ((int)ne0 - r0 < NR0) ? ((int)ne0 - r0) : NR0;
    const short nr1 = ((int)ne1 - r1 < NR1) ? ((int)ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    device const half4x4 * x =
        (device const half4x4 *)((device const uchar *)src0 + nb01*(r0 + lr0)) + il0;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(
        (device const uchar *)src1 + nb11*(r1 + lr1) + nb10*iy);

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < (int)ne00; loop_k += NK) {
        half4x4 temp_a = *x;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) =
                (half2x4)(*((device float2x4 *)y));
        }

        x += 2;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        #pragma unroll
        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= (int)ne0 && r1 + NR1 <= (int)ne1) {
        device float * C = (device float *)dst +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*ne0*(i/4), ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str =
            ((threadgroup float *)shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *)dst + r0 + (r1 + j)*ne0;
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

