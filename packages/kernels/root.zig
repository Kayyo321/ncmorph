//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Plain f32 matrix kernels used as the correctness reference for optimized kernels.

const std = @import("std");
const bf16 = @import("bf16");

/// Rounds one F32 value to BF16 using the repository's PyTorch-validated conversion.
pub fn round_bf16(value: f32) f32 {
    return bf16.bf16_t.from_f32(value).to_f32();
}

extern "c" fn powf(base: f32, exponent: f32) f32;
extern "c" fn cosf(value: f32) f32;
extern "c" fn sinf(value: f32) f32;

/// Inner reduction chunk selected for the blocked implementation (D61).
pub const InnerBlockSize = 32;

/// Computes row-major `output = input * weightsᵀ` using 32-element tiles on
/// the token, output, and inner dimensions. The slices must have exactly the
/// sizes implied by the dimensions. Reduction terms are accumulated in
/// increasing `k`, with f32 products and partial sums.
pub fn matmul(
    input: []const f32,
    weights: []const f32,
    output: []f32,
    token_count: usize,
    input_width: usize,
    output_width: usize,
) void {
    std.debug.assert(input.len == token_count * input_width);
    std.debug.assert(weights.len == output_width * input_width);
    std.debug.assert(output.len == token_count * output_width);

    var token_start: usize = 0;
    while (token_start < token_count) : (token_start += InnerBlockSize) {
        const token_end = @min(token_start + InnerBlockSize, token_count);
        var output_start: usize = 0;
        while (output_start < output_width) : (output_start += InnerBlockSize) {
            const output_end = @min(output_start + InnerBlockSize, output_width);

            for (token_start..token_end) |token_index| {
                for (output_start..output_end) |output_index| {
                    output[token_index * output_width + output_index] = 0;
                }
            }

            var input_start: usize = 0;
            while (input_start < input_width) : (input_start += InnerBlockSize) {
                const input_end = @min(input_start + InnerBlockSize, input_width);
                for (token_start..token_end) |token_index| {
                    for (output_start..output_end) |output_index| {
                        const output_offset = token_index * output_width + output_index;
                        for (input_start..input_end) |input_index| {
                            output[output_offset] += input[token_index * input_width + input_index] *
                                weights[output_index * input_width + input_index];
                        }
                    }
                }
            }
        }
    }
}

/// Computes row-major `output = input * weightsᵀ` for `input[T, in]` and
/// `weights[out, in]`. The slices must have exactly the sizes implied by the
/// dimensions. The accumulation order is increasing `k`, and each product and
/// partial sum is f32.
pub fn matmul_reference(
    input: []const f32,
    weights: []const f32,
    output: []f32,
    token_count: usize,
    input_width: usize,
    output_width: usize,
) void {
    std.debug.assert(input.len == token_count * input_width);
    std.debug.assert(weights.len == output_width * input_width);
    std.debug.assert(output.len == token_count * output_width);

    for (0..token_count) |token_index| {
        for (0..output_width) |output_index| {
            var sum: f32 = 0;
            for (0..input_width) |input_index| {
                sum += input[token_index * input_width + input_index] *
                    weights[output_index * input_width + input_index];
            }
            output[token_index * output_width + output_index] = sum;
        }
    }
}

/// Computes an F32-accumulated matrix product and rounds each tensor output to BF16.
pub fn matmul_bf16(input: []const f32, weights: []const f32, output: []f32, token_count: usize, input_width: usize, output_width: usize) void {
    matmul_bf16_impl(input, weights, null, output, token_count, input_width, output_width);
}

/// BF16 F.linear semantics: adds an optional BF16 bias to the F32 accumulator
/// before the single BF16 output conversion.
pub fn matmul_bf16_bias(input: []const f32, weights: []const f32, bias: []const f32, output: []f32, token_count: usize, input_width: usize, output_width: usize) void {
    matmul_bf16_impl(input, weights, bias, output, token_count, input_width, output_width);
}

fn matmul_bf16_impl(input: []const f32, weights: []const f32, bias: ?[]const f32, output: []f32, token_count: usize, input_width: usize, output_width: usize) void {
    std.debug.assert(input.len == token_count * input_width);
    std.debug.assert(weights.len == output_width * input_width);
    std.debug.assert(output.len == token_count * output_width);
    if (bias) |values| std.debug.assert(values.len == output_width);

    const reduction_block_size = @min(input_width, if (input_width > 1024) @as(usize, 1024) else @as(usize, 512));
    for (0..token_count) |token_index| {
        for (0..output_width) |output_index| {
            var sum: f32 = 0;
            var block_start: usize = 0;
            while (block_start < input_width) : (block_start += reduction_block_size) {
                const block_end = @min(block_start + reduction_block_size, input_width);
                var partial: f32 = 0;
                var input_index = block_start;
                while (input_index + 1 < block_end) : (input_index += 2) {
                    const input_offset = token_index * input_width + input_index;
                    const weight_offset = output_index * input_width + input_index;
                    const input_high = flush_f32_subnormal(round_bf16(input[input_offset + 1]));
                    const weight_high = flush_f32_subnormal(round_bf16(weights[weight_offset + 1]));
                    const input_low = flush_f32_subnormal(round_bf16(input[input_offset]));
                    const weight_low = flush_f32_subnormal(round_bf16(weights[weight_offset]));
                    partial = flush_f32_subnormal(@mulAdd(f32, input_high, weight_high, partial));
                    partial = flush_f32_subnormal(@mulAdd(f32, input_low, weight_low, partial));
                }
                if (input_index < block_end) {
                    const input_value = flush_f32_subnormal(round_bf16(input[token_index * input_width + input_index]));
                    const weight_value = flush_f32_subnormal(round_bf16(weights[output_index * input_width + input_index]));
                    partial = flush_f32_subnormal(@mulAdd(f32, input_value, weight_value, partial));
                }
                sum += partial;
            }
            if (bias) |values| sum += round_bf16(values[output_index]);
            output[token_index * output_width + output_index] = round_bf16(sum);
        }
    }
}

fn flush_f32_subnormal(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    if ((bits & 0x7f80_0000) == 0) return @bitCast(bits & 0x8000_0000);
    return value;
}

/// Applies RMSNorm to one vector using f32 accumulation and output values.
/// `input`, `weight`, and `output` must have equal nonzero lengths.
fn pinned_torch_sum_squares_f32(input: []const f32) f32 {
    const vector_width = 8;
    const vector_count = input.len / vector_width;
    const row_count = vector_count / 4;
    var log2_row_count: usize = 0;
    var log2_limit: usize = 1;
    while (log2_limit < row_count) : (log2_row_count += 1) log2_limit *= 2;
    const level_power = @max(4, log2_row_count / 4);
    const level_step = @as(usize, 1) << @intCast(level_power);
    const level_mask = level_step - 1;

    var accumulators: [4][4][vector_width]f32 = undefined;
    for (&accumulators) |*level| {
        for (level) |*row| @memset(row, 0);
    }

    var row_index: usize = 0;
    while (row_index + level_step <= row_count) {
        for (0..level_step) |step_index| {
            for (0..4) |row| {
                const vector_index = (row_index + step_index) * 4 + row;
                for (0..vector_width) |lane| {
                    const value = input[vector_index * vector_width + lane];
                    accumulators[0][row][lane] += value * value;
                }
            }
        }
        row_index += level_step;
        for (1..4) |level| {
            for (0..4) |row| {
                for (0..vector_width) |lane| {
                    accumulators[level][row][lane] += accumulators[level - 1][row][lane];
                    accumulators[level - 1][row][lane] = 0;
                }
            }
            if ((row_index & (level_mask << @intCast(level * level_power))) != 0) break;
        }
    }

    for (row_index..row_count) |row_group| {
        for (0..4) |row| {
            const vector_index = row_group * 4 + row;
            for (0..vector_width) |lane| {
                const value = input[vector_index * vector_width + lane];
                accumulators[0][row][lane] += value * value;
            }
        }
    }
    for (1..4) |level| {
        for (0..4) |row| {
            for (0..vector_width) |lane| {
                accumulators[0][row][lane] += accumulators[level][row][lane];
            }
        }
    }
    for (row_count * 4..vector_count) |vector_index| {
        for (0..vector_width) |lane| {
            const value = input[vector_index * vector_width + lane];
            accumulators[0][0][lane] += value * value;
        }
    }

    var vector_sum: [vector_width]f32 = accumulators[0][0];
    for (1..4) |row| {
        for (0..vector_width) |lane| vector_sum[lane] += accumulators[0][row][lane];
    }
    var sum_squares: f32 = 0;
    for (vector_count * vector_width..input.len) |index| {
        sum_squares += input[index] * input[index];
    }
    for (vector_sum) |partial| sum_squares += partial;
    return sum_squares;
}

pub fn rms_norm(input: []const f32, weight: []const f32, output: []f32, epsilon: f32) void {
    std.debug.assert(input.len != 0);
    std.debug.assert(weight.len == input.len);
    std.debug.assert(output.len == input.len);

    const sum_squares = pinned_torch_sum_squares_f32(input);
    const mean_squares = sum_squares / @as(f32, @floatFromInt(input.len));
    const inverse_rms = 1.0 / std.math.sqrt(mean_squares + epsilon);

    for (input, weight, output) |value, scale, *result| {
        const normalized = value * inverse_rms;
        result.* = normalized * scale;
    }
}

/// BF16 Llama RMSNorm: F32 variance and normalization, cast normalized values to BF16,
/// then multiply the BF16 scale and round the result to BF16.
pub fn rms_norm_bf16(input: []const f32, weight: []const f32, output: []f32, epsilon: f32) void {
    std.debug.assert(input.len != 0 and weight.len == input.len and output.len == input.len);
    const sum_squares = pinned_torch_sum_squares_f32(input);
    const mean_squares = sum_squares / @as(f32, @floatFromInt(input.len));
    const inverse_rms = 1.0 / std.math.sqrt(mean_squares + epsilon);
    for (input, weight, output) |value, scale, *result| {
        const normalized = round_bf16(value * inverse_rms);
        result.* = round_bf16(normalized * round_bf16(scale));
    }
}

/// Applies the SiLU function elementwise: `x * sigmoid(x)`.
pub fn silu(input: []const f32, output: []f32) void {
    std.debug.assert(output.len == input.len);
    for (input, output) |value, *result| {
        const sigmoid = 1.0 / (1.0 + std.math.exp(-value));
        result.* = value * sigmoid;
    }
}

/// BF16 SiLU with an F32 scalar intermediate and BF16 tensor output.
pub fn silu_bf16(input: []const f32, output: []f32) void {
    std.debug.assert(output.len == input.len);
    for (input, output) |value, *result| {
        result.* = round_bf16(value / (1.0 + std.math.exp(-value)));
    }
}

/// Applies softmax to one nonempty vector, subtracting its maximum first.
pub fn softmax(input: []const f32, output: []f32) void {
    std.debug.assert(input.len != 0);
    std.debug.assert(output.len == input.len);

    var max_value = input[0];
    for (input[1..]) |value| max_value = @max(max_value, value);

    var sum: f32 = 0;
    for (input, output) |value, *result| {
        result.* = std.math.exp(value - max_value);
        sum += result.*;
    }
    for (output) |*result| result.* /= sum;
}

/// Copies one row from a row-major `[vocabulary_size, embedding_width]` table.
pub fn embedding_row(
    table: []const f32,
    vocabulary_size: usize,
    embedding_width: usize,
    token_id: usize,
    output: []f32,
) void {
    std.debug.assert(table.len == vocabulary_size * embedding_width);
    std.debug.assert(token_id < vocabulary_size);
    std.debug.assert(output.len == embedding_width);

    const row_start = token_id * embedding_width;
    for (output, 0..) |*value, index| value.* = table[row_start + index];
}

/// Builds default RoPE inverse frequencies and rotate-half cosine/sine tables.
/// Outputs use row-major `[position_count, head_dim]` layout.
pub fn rope_default(
    head_dim: usize,
    rope_theta: f32,
    position_count: usize,
    inverse_frequencies: []f32,
    cosine: []f32,
    sine: []f32,
) void {
    std.debug.assert(head_dim != 0 and head_dim % 2 == 0);
    std.debug.assert(rope_theta > 0);
    std.debug.assert(inverse_frequencies.len == head_dim / 2);
    std.debug.assert(cosine.len == position_count * head_dim);
    std.debug.assert(sine.len == position_count * head_dim);

    const head_dim_f32: f32 = @floatFromInt(head_dim);
    for (inverse_frequencies, 0..) |*inverse_frequency, index| {
        const exponent = @as(f32, @floatFromInt(2 * index)) / head_dim_f32;
        inverse_frequency.* = 1.0 / powf(rope_theta, exponent);
    }
    rope_tables(inverse_frequencies, position_count, cosine, sine);
}

/// Expands inverse frequencies into duplicated rotate-half cosine and sine tables.
pub fn rope_tables(inverse_frequencies: []const f32, position_count: usize, cosine: []f32, sine: []f32) void {
    const head_dim = inverse_frequencies.len * 2;
    std.debug.assert(head_dim != 0);
    std.debug.assert(cosine.len == position_count * head_dim);
    std.debug.assert(sine.len == position_count * head_dim);

    const half_dim = head_dim / 2;
    for (0..position_count) |position| {
        const position_f32: f32 = @floatFromInt(position);
        const row_start = position * head_dim;
        for (inverse_frequencies, 0..) |inverse_frequency, index| {
            const angle = position_f32 * inverse_frequency;
            const cos_value = cosf(angle);
            const sin_value = sinf(angle);
            cosine[row_start + index] = cos_value;
            cosine[row_start + half_dim + index] = cos_value;
            sine[row_start + index] = sin_value;
            sine[row_start + half_dim + index] = sin_value;
        }
    }
}

/// Weights for one decoder attention layer. Biases and QK-normalization scales
/// are independently optional; query and key normalization must be paired.
pub const attention_weights_t = struct {
    query_weight: []const f32,
    query_bias: ?[]const f32 = null,
    key_weight: []const f32,
    key_bias: ?[]const f32 = null,
    value_weight: []const f32,
    value_bias: ?[]const f32 = null,
    query_norm_weight: ?[]const f32 = null,
    key_norm_weight: ?[]const f32 = null,
    output_weight: []const f32,
    output_bias: ?[]const f32 = null,
};

/// Shape, positional data and mask for one prefill attention call.
pub const attention_config_t = struct {
    token_count: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_epsilon: f32,
    cosine: []const f32,
    sine: []const f32,
    attention_mask: []const bool,
};

/// Produces a row-major `[query, key]` allow mask for causal attention.
/// With a window `W`, the allowed key positions satisfy `q - W < kv <= q`.
pub fn causal_mask(
    query_positions: []const usize,
    key_positions: []const usize,
    sliding_window: ?usize,
    output: []bool,
) void {
    std.debug.assert(output.len == query_positions.len * key_positions.len);
    if (sliding_window) |window| std.debug.assert(window != 0);

    for (query_positions, 0..) |query_position, query_index| {
        for (key_positions, 0..) |key_position, key_index| {
            const is_causal = key_position <= query_position;
            const is_in_window = if (sliding_window) |window|
                query_position < window or key_position > query_position - window
            else
                true;
            output[query_index * key_positions.len + key_index] = is_causal and is_in_window;
        }
    }
}

/// Runs a single f32 attention layer over a contiguous prefill sequence.
/// Projections use `[out, in]` weights; the mask uses `[query, key]` order and
/// `true` means that the key is allowed to contribute.
pub fn attention(
    allocator: std.mem.Allocator,
    input: []const f32,
    weights: attention_weights_t,
    config: attention_config_t,
    output: []f32,
) !void {
    try attention_impl(allocator, input, weights, config, output, false, null);
}

/// Runs f32 attention and copies the pre-output-projection context for diagnostics.
pub fn attention_with_context(
    allocator: std.mem.Allocator,
    input: []const f32,
    weights: attention_weights_t,
    config: attention_config_t,
    output: []f32,
    context_trace: []f32,
) !void {
    try attention_impl(allocator, input, weights, config, output, false, context_trace);
}

/// Runs attention with BF16 tensor outputs and F32 reductions, matching the pinned
/// eager Transformers path for BF16 inputs.
pub fn attention_bf16(
    allocator: std.mem.Allocator,
    input: []const f32,
    weights: attention_weights_t,
    config: attention_config_t,
    output: []f32,
) !void {
    try attention_impl(allocator, input, weights, config, output, true, null);
}

/// Runs BF16 attention and copies the pre-output-projection context for parity diagnostics.
pub fn attention_bf16_with_context(
    allocator: std.mem.Allocator,
    input: []const f32,
    weights: attention_weights_t,
    config: attention_config_t,
    output: []f32,
    context_trace: []f32,
) !void {
    try attention_impl(allocator, input, weights, config, output, true, context_trace);
}

fn attention_impl(
    allocator: std.mem.Allocator,
    input: []const f32,
    weights: attention_weights_t,
    config: attention_config_t,
    output: []f32,
    bf16_mode: bool,
    context_trace: ?[]f32,
) !void {
    const token_count = config.token_count;
    const hidden_size = config.hidden_size;
    const head_dim = config.head_dim;
    const query_heads = config.num_attention_heads;
    const key_value_heads = config.num_key_value_heads;
    std.debug.assert(token_count != 0 and hidden_size != 0 and head_dim != 0);
    std.debug.assert(query_heads != 0 and key_value_heads != 0 and query_heads % key_value_heads == 0);
    std.debug.assert(input.len == token_count * hidden_size);
    std.debug.assert(output.len == token_count * hidden_size);
    std.debug.assert(config.cosine.len == token_count * head_dim);
    std.debug.assert(config.sine.len == token_count * head_dim);
    std.debug.assert(config.attention_mask.len == token_count * token_count);
    std.debug.assert((weights.query_norm_weight == null) == (weights.key_norm_weight == null));

    const query_width = query_heads * head_dim;
    const key_value_width = key_value_heads * head_dim;
    if (context_trace) |trace| std.debug.assert(trace.len == token_count * query_width);
    std.debug.assert(weights.query_weight.len == query_width * hidden_size);
    std.debug.assert(weights.key_weight.len == key_value_width * hidden_size);
    std.debug.assert(weights.value_weight.len == key_value_width * hidden_size);
    std.debug.assert(weights.output_weight.len == hidden_size * query_width);
    if (weights.query_bias) |bias| std.debug.assert(bias.len == query_width);
    if (weights.key_bias) |bias| std.debug.assert(bias.len == key_value_width);
    if (weights.value_bias) |bias| std.debug.assert(bias.len == key_value_width);
    if (weights.output_bias) |bias| std.debug.assert(bias.len == hidden_size);
    if (weights.query_norm_weight) |scale| std.debug.assert(scale.len == head_dim);
    if (weights.key_norm_weight) |scale| std.debug.assert(scale.len == head_dim);

    const queries = try allocator.alloc(f32, token_count * query_width);
    defer allocator.free(queries);
    const keys = try allocator.alloc(f32, token_count * key_value_width);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, token_count * key_value_width);
    defer allocator.free(values);
    const context = try allocator.alloc(f32, token_count * query_width);
    defer allocator.free(context);

    project(input, weights.query_weight, weights.query_bias, queries, token_count, hidden_size, query_width, bf16_mode);
    project(input, weights.key_weight, weights.key_bias, keys, token_count, hidden_size, key_value_width, bf16_mode);
    project(input, weights.value_weight, weights.value_bias, values, token_count, hidden_size, key_value_width, bf16_mode);

    const group_count = query_heads / key_value_heads;
    for (0..token_count) |token_index| {
        for (0..query_heads) |head_index| {
            const query_start = token_index * query_width + head_index * head_dim;
            if (weights.query_norm_weight) |scale| {
                const query_slice = queries[query_start .. query_start + head_dim];
                if (bf16_mode) rms_norm_bf16(query_slice, scale, query_slice, config.rms_norm_epsilon) else rms_norm(query_slice, scale, query_slice, config.rms_norm_epsilon);
            }
            rotate_head(
                queries[query_start .. query_start + head_dim],
                config.cosine[token_index * head_dim ..][0..head_dim],
                config.sine[token_index * head_dim ..][0..head_dim],
                bf16_mode,
            );
        }
    }
    for (0..token_count) |token_index| {
        for (0..key_value_heads) |head_index| {
            const key_start = token_index * key_value_width + head_index * head_dim;
            if (weights.key_norm_weight) |norm_scale| {
                const key_slice = keys[key_start .. key_start + head_dim];
                if (bf16_mode) rms_norm_bf16(key_slice, norm_scale, key_slice, config.rms_norm_epsilon) else rms_norm(key_slice, norm_scale, key_slice, config.rms_norm_epsilon);
            }
            rotate_head(
                keys[key_start .. key_start + head_dim],
                config.cosine[token_index * head_dim ..][0..head_dim],
                config.sine[token_index * head_dim ..][0..head_dim],
                bf16_mode,
            );
        }
    }

    const scores = try allocator.alloc(f32, token_count);
    defer allocator.free(scores);
    const probabilities = try allocator.alloc(f32, token_count);
    defer allocator.free(probabilities);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (0..token_count) |query_index| {
        for (0..query_heads) |head_index| {
            const query_start = query_index * query_width + head_index * head_dim;
            const key_value_head = head_index / group_count;
            var any_key_allowed = false;
            for (0..token_count) |key_index| {
                if (!config.attention_mask[query_index * token_count + key_index]) {
                    scores[key_index] = -std.math.inf(f32);
                    continue;
                }
                any_key_allowed = true;
                const key_start = key_index * key_value_width + key_value_head * head_dim;
                var score: f32 = 0;
                for (0..head_dim) |dimension| {
                    score += queries[query_start + dimension] * keys[key_start + dimension];
                }
                const score_value = maybe_round_bf16(score, bf16_mode);
                scores[key_index] = maybe_round_bf16(score_value * scale, bf16_mode);
            }
            if (any_key_allowed) {
                softmax(scores, probabilities);
                if (bf16_mode) {
                    for (probabilities) |*probability| probability.* = round_bf16(probability.*);
                }
            } else {
                @memset(probabilities, 0);
            }

            const context_start = query_index * query_width + head_index * head_dim;
            for (0..head_dim) |dimension| {
                var sum: f32 = 0;
                for (0..token_count) |key_index| {
                    const value_start = key_index * key_value_width + key_value_head * head_dim;
                    sum += probabilities[key_index] * values[value_start + dimension];
                }
                context[context_start + dimension] = maybe_round_bf16(sum, bf16_mode);
            }
        }
    }

    if (context_trace) |trace| {
        for (context, 0..) |value, index| trace[index] = value;
    }
    project(context, weights.output_weight, weights.output_bias, output, token_count, query_width, hidden_size, bf16_mode);
}

fn project(
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    output: []f32,
    token_count: usize,
    input_width: usize,
    output_width: usize,
    bf16_mode: bool,
) void {
    if (bf16_mode) {
        if (bias) |values_bias| {
            matmul_bf16_bias(input, weight, values_bias, output, token_count, input_width, output_width);
        } else {
            matmul_bf16(input, weight, output, token_count, input_width, output_width);
        }
        return;
    }
    matmul(input, weight, output, token_count, input_width, output_width);
    if (bias) |values_bias| {
        for (0..token_count) |token_index| {
            for (0..output_width) |output_index| {
                output[token_index * output_width + output_index] += values_bias[output_index];
            }
        }
    }
}

fn maybe_round_bf16(value: f32, enabled: bool) f32 {
    return if (enabled) round_bf16(value) else value;
}

fn rotate_head(values: []f32, cosine: []const f32, sine: []const f32, bf16_mode: bool) void {
    std.debug.assert(values.len != 0 and values.len % 2 == 0);
    std.debug.assert(cosine.len == values.len and sine.len == values.len);
    const half_dim = values.len / 2;
    for (0..half_dim) |index| {
        const first = values[index];
        const second = values[index + half_dim];
        const cos_first = maybe_round_bf16(cosine[index], bf16_mode);
        const sin_first = maybe_round_bf16(sine[index], bf16_mode);
        const cos_second = maybe_round_bf16(cosine[index + half_dim], bf16_mode);
        const sin_second = maybe_round_bf16(sine[index + half_dim], bf16_mode);
        const left_product = maybe_round_bf16(first * cos_first, bf16_mode);
        const right_product = maybe_round_bf16(-second * sin_first, bf16_mode);
        const right_product_2 = maybe_round_bf16(first * sin_second, bf16_mode);
        const left_product_2 = maybe_round_bf16(second * cos_second, bf16_mode);
        values[index] = maybe_round_bf16(left_product + right_product, bf16_mode);
        values[index + half_dim] = maybe_round_bf16(left_product_2 + right_product_2, bf16_mode);
    }
}

/// Builds Llama 3 adjusted inverse frequencies in the reference f32 operation order.
pub fn rope_llama3(
    head_dim: usize,
    rope_theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position_embeddings: usize,
    inverse_frequencies: []f32,
) void {
    std.debug.assert(head_dim != 0 and head_dim % 2 == 0);
    std.debug.assert(rope_theta > 0 and factor > 0);
    std.debug.assert(low_freq_factor > 0 and high_freq_factor > low_freq_factor);
    std.debug.assert(inverse_frequencies.len == head_dim / 2);

    rope_default_frequencies(head_dim, rope_theta, inverse_frequencies);

    const old_context_len: f32 = @floatFromInt(original_max_position_embeddings);
    const low_freq_wavelen: f32 = @floatCast(
        @as(f64, @floatFromInt(original_max_position_embeddings)) / @as(f64, low_freq_factor),
    );
    const high_freq_wavelen: f32 = @floatCast(
        @as(f64, @floatFromInt(original_max_position_embeddings)) / @as(f64, high_freq_factor),
    );
    const two_pi: f32 = 2.0 * std.math.pi;

    for (inverse_frequencies) |*inverse_frequency| {
        const wavelen = two_pi / inverse_frequency.*;
        const llama_frequency = if (wavelen > low_freq_wavelen)
            inverse_frequency.* / factor
        else
            inverse_frequency.*;
        const smooth_factor = (old_context_len / wavelen - low_freq_factor) /
            (high_freq_factor - low_freq_factor);
        const smoothed_frequency = (1.0 - smooth_factor) * llama_frequency / factor +
            smooth_factor * llama_frequency;
        const is_medium_frequency = !(wavelen < high_freq_wavelen) and !(wavelen > low_freq_wavelen);
        inverse_frequency.* = if (is_medium_frequency) smoothed_frequency else llama_frequency;
    }
}

fn rope_default_frequencies(head_dim: usize, rope_theta: f32, inverse_frequencies: []f32) void {
    const head_dim_f32: f32 = @floatFromInt(head_dim);
    for (inverse_frequencies, 0..) |*inverse_frequency, index| {
        const exponent = @as(f32, @floatFromInt(2 * index)) / head_dim_f32;
        inverse_frequency.* = 1.0 / powf(rope_theta, exponent);
    }
}

test "matmul_reference matches a hand-computed matrix" {
    const input = [_]f32{ 5, 6, 7, 8 };
    const weights = [_]f32{ 1, 2, 3, 4 };
    var output: [4]f32 = undefined;
    var reference: [4]f32 = undefined;

    matmul(&input, &weights, &output, 2, 2, 2);
    matmul_reference(&input, &weights, &reference, 2, 2, 2);

    try std.testing.expectEqualSlices(f32, &.{ 17, 39, 23, 53 }, &output);
    try std.testing.expectEqualSlices(f32, &reference, &output);
}

test "matmul_reference satisfies the D53 bound for seeded inputs" {
    var prng = std.Random.Xoshiro256.init(0xB1_2026);
    const random = prng.random();

    try check_random_case(1, InnerBlockSize + 3, 4, random);
    try check_random_case(3, 7, 5, random);
    try check_random_case(2, 1, 3, random);
    try check_random_case(33, InnerBlockSize + 3, 34, random);
}

test "BF16 linear uses paired fused products and oneDNN K blocks" {
    const token_count = 4;
    const input_width = 1536;
    const output_width = 576;
    const allocator = std.testing.allocator;
    const input = try allocator.alloc(f32, token_count * input_width);
    defer allocator.free(input);
    const weights = try allocator.alloc(f32, output_width * input_width);
    defer allocator.free(weights);
    const output = try allocator.alloc(f32, token_count * output_width);
    defer allocator.free(output);

    for (0..token_count) |token_index| {
        for (0..input_width) |input_index| {
            const numerator: i32 = @intCast((token_index * 29 + input_index * 37) % 251);
            input[token_index * input_width + input_index] = @as(f32, @floatFromInt(numerator - 125)) / 128.0;
        }
    }
    for (0..output_width) |output_index| {
        for (0..input_width) |input_index| {
            const numerator: i32 = @intCast((output_index * 17 + input_index * 11) % 257);
            weights[output_index * input_width + input_index] = @as(f32, @floatFromInt(numerator - 128)) / 128.0;
        }
    }

    matmul_bf16(input, weights, output, token_count, input_width, output_width);

    var digest: u64 = 14_695_981_039_346_656_037;
    for (output) |value| {
        const bf16_bits = @as(u16, @truncate(@as(u32, @bitCast(value)) >> 16));
        digest = (digest ^ bf16_bits) *% 1_099_511_628_211;
    }
    try std.testing.expectEqual(@as(u64, 0x547f_69f6_67c8_103f), digest);

    try std.testing.expectEqual(@as(f32, -0.466796875), output[0]);
    try std.testing.expectEqual(@as(f32, 1.109375), output[1]);
    try std.testing.expectEqual(@as(f32, -0.796875), output[1793]);
}

test "BF16 linear flushes subnormal products like VDPBF16PS" {
    const value: f32 = 5.421010862427522e-20;
    const input = [_]f32{ value, value };
    const weights = [_]f32{ value, value };
    var output: [1]f32 = undefined;

    matmul_bf16(&input, &weights, &output, 1, 2, 1);

    try std.testing.expectEqual(@as(f32, 0), output[0]);
}

test "BF16 linear combines the pinned 512-wide K-block partials" {
    const input_width = 576;
    var input: [input_width]f32 = undefined;
    @memset(&input, 5.960464477539063e-8);
    input[0] = 1;
    input[input_width - 1] = 0.00390625;
    var weights: [input_width]f32 = undefined;
    @memset(&weights, 1);
    var output: [1]f32 = undefined;

    matmul_bf16(&input, &weights, &output, 1, input_width, 1);

    try std.testing.expectEqual(@as(f32, 1.0078125), output[0]);
}

test "softmax of two equal logits is exactly uniform" {
    const input = [_]f32{ 2, 2 };
    var output: [2]f32 = undefined;

    softmax(&input, &output);

    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5 }, &output);
}

test "SiLU of zero is exactly zero" {
    var output: [1]f32 = undefined;

    silu(&.{0}, &output);

    try std.testing.expectEqual(@as(f32, 0), output[0]);
}

test "RMSNorm of a constant vector is exact" {
    const input = [_]f32{ 3, 3, 3, 3 };
    const weight = [_]f32{ 1, 1, 1, 1 };
    var output: [4]f32 = undefined;

    rms_norm(&input, &weight, &output, 0);

    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, &output);
}

test "RMSNorm sum of squares follows the pinned CPU cascade" {
    var input: [896]f32 = undefined;
    for (&input, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt((index * 17) % 101 + 1)) * 0.125;
    }
    try std.testing.expectEqual(
        @as(u32, 0x473b_e6e0),
        @as(u32, @bitCast(pinned_torch_sum_squares_f32(&input))),
    );

    var tail_input: [11]f32 = undefined;
    for (&tail_input, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(index + 1)) * 0.125;
    }
    try std.testing.expectEqual(
        @as(u32, 0x40fd_0000),
        @as(u32, @bitCast(pinned_torch_sum_squares_f32(&tail_input))),
    );
}

test "embedding_row copies the selected table row" {
    const table = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output: [2]f32 = undefined;

    embedding_row(&table, 3, 2, 1, &output);

    try std.testing.expectEqualSlices(f32, &.{ 3, 4 }, &output);
}

test "elementwise kernels satisfy D53 bounds for seeded inputs" {
    var prng = std.Random.Xoshiro256.init(0xB2_2026);
    const random = prng.random();
    try check_rms_norm(random);
    try check_silu(random);
    try check_softmax(random);
}

fn check_random_case(
    comptime token_count: usize,
    comptime input_width: usize,
    comptime output_width: usize,
    random: std.Random,
) !void {
    var input: [token_count * input_width]f32 = undefined;
    var weights: [output_width * input_width]f32 = undefined;
    var output: [token_count * output_width]f32 = undefined;
    var reference_output: [token_count * output_width]f32 = undefined;

    for (&input) |*value| value.* = random.float(f32) * 2 - 1;
    for (&weights) |*value| value.* = random.float(f32) * 2 - 1;

    matmul(&input, &weights, &output, token_count, input_width, output_width);
    matmul_reference(&input, &weights, &reference_output, token_count, input_width, output_width);
    try std.testing.expectEqualSlices(f32, &reference_output, &output);

    for (0..token_count) |token_index| {
        for (0..output_width) |output_index| {
            var reference: f64 = 0;
            var absolute_products: f64 = 0;
            for (0..input_width) |input_index| {
                const product = @as(f64, input[token_index * input_width + input_index]) *
                    @as(f64, weights[output_index * input_width + input_index]);
                reference += product;
                absolute_products += @abs(product);
            }

            const got: f64 = output[token_index * output_width + output_index];
            const difference = @abs(got - reference);
            const tolerance = @as(f64, @floatFromInt(input_width + 1)) *
                (1.0 / 16_777_216.0) * absolute_products;
            try std.testing.expect(difference <= tolerance);
        }
    }
}

fn check_rms_norm(random: std.Random) !void {
    const input = random_values(17, random, 2);
    const weight = random_values(17, random, 1);
    const epsilon: f32 = 1e-5;
    var output: [17]f32 = undefined;

    rms_norm(&input, &weight, &output, epsilon);

    var sum_squares: f64 = 0;
    for (input) |value| sum_squares += @as(f64, value) * @as(f64, value);
    const mean_squares = sum_squares / 17.0;
    const inverse_rms = 1.0 / @sqrt(mean_squares + @as(f64, epsilon));
    const tolerance = (17.0 + 8.0) / 16_777_216.0;
    for (input, weight, output) |value, scale, got| {
        const reference = @as(f64, value) * inverse_rms * @as(f64, scale);
        try expect_relative(got, reference, tolerance);
    }
}

fn check_silu(random: std.Random) !void {
    const input = random_values(64, random, 8);
    var output: [64]f32 = undefined;

    silu(&input, &output);

    const tolerance = 4.0 / 16_777_216.0;
    for (input, output) |value, got| {
        const value_f64: f64 = value;
        const reference = value_f64 / (1.0 + @exp(-value_f64));
        try expect_relative(got, reference, tolerance);
    }
}

fn check_softmax(random: std.Random) !void {
    const input = random_values(19, random, 8);
    var output: [19]f32 = undefined;

    softmax(&input, &output);

    var max_value: f64 = input[0];
    for (input[1..]) |value| max_value = @max(max_value, @as(f64, value));
    var sum: f64 = 0;
    var reference: [19]f64 = undefined;
    for (input, &reference) |value, *result| {
        result.* = @exp(@as(f64, value) - max_value);
        sum += result.*;
    }
    const tolerance = (19.0 + 8.0) / 16_777_216.0;
    for (reference, output) |value, got| try expect_relative(got, value / sum, tolerance);
}

fn random_values(comptime count: usize, random: std.Random, scale: f32) [count]f32 {
    var values: [count]f32 = undefined;
    for (&values) |*value| value.* = (random.float(f32) * 2 - 1) * scale;
    return values;
}

fn expect_relative(got: f32, reference: f64, tolerance: f64) !void {
    const got_f64: f64 = got;
    if (reference == 0) {
        try std.testing.expect(got_f64 == 0);
    } else {
        try std.testing.expect(@abs(got_f64 - reference) <= tolerance * @abs(reference));
    }
}

test "default RoPE matches transformers golden tables" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const golden = try read_rope_golden(arena.allocator());

    try std.testing.expectEqualStrings("5.20.2", golden.transformers_version);
    try std.testing.expectEqual(@as(usize, 0), golden.position_start);
    try std.testing.expectEqual(@as(usize, 4095), golden.position_end_inclusive);

    const cases = [_]rope_case_t{
        .{ .head_dim = 64, .rope_theta = 10_000, .reference = golden.tables.default_head_dim_64_theta_10000 },
        .{ .head_dim = 64, .rope_theta = 500_000, .reference = golden.tables.default_head_dim_64_theta_500000 },
        .{ .head_dim = 128, .rope_theta = 10_000, .reference = golden.tables.default_head_dim_128_theta_10000 },
        .{ .head_dim = 128, .rope_theta = 500_000, .reference = golden.tables.default_head_dim_128_theta_500000 },
    };

    var max_error: f64 = 0;
    for (cases) |case| max_error = @max(max_error, try check_rope_default_case(case));
    std.debug.print("B3 maximum absolute cosine/sine error: {d:.12}\n", .{max_error});
}

test "llama3 RoPE inverse frequencies match transformers golden values" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const golden = try read_rope_golden(arena.allocator());
    const reference = golden.tables.llama3_head_dim_64_theta_500000;

    try std.testing.expectEqual(@as(usize, 64), reference.head_dim);
    try std.testing.expectEqualStrings("llama3", reference.rope_parameters.rope_type);
    try std.testing.expectEqual(@as(f64, 500_000), reference.rope_parameters.rope_theta);
    try std.testing.expectEqual(@as(f64, 32), reference.rope_parameters.factor.?);
    try std.testing.expectEqual(@as(f64, 1), reference.rope_parameters.low_freq_factor.?);
    try std.testing.expectEqual(@as(f64, 4), reference.rope_parameters.high_freq_factor.?);
    try std.testing.expectEqual(@as(usize, 8192), reference.rope_parameters.original_max_position_embeddings.?);

    var inverse_frequencies: [32]f32 = undefined;
    rope_llama3(64, 500_000, 32, 1, 4, 8192, &inverse_frequencies);

    var max_relative_error: f64 = 0;
    for (inverse_frequencies, reference.inv_freq) |got, expected| {
        const error_value = @abs(@as(f64, got) - expected) / @abs(expected);
        max_relative_error = @max(max_relative_error, error_value);
        try std.testing.expect(error_value <= 1e-6);
    }
    std.debug.print("B4 maximum relative inverse-frequency error: {d:.12}\n", .{max_relative_error});
}

test "Qwen3 attention with grouped heads, biases and QK-norm matches golden output" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const raw_json = std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        "zig-out/golden/qwen3_attention_bias/reference.json",
        arena.allocator(),
        .limited(10 * 1024 * 1024),
    ) catch |err| return skip_without_golden(err, "zig-out/golden/qwen3_attention_bias/reference.json");
    const golden = try std.json.parseFromSliceLeaky(attention_golden_t, arena.allocator(), raw_json, .{
        .ignore_unknown_fields = true,
    });
    try std.testing.expectEqualStrings("5.20.2", golden.transformers_version);
    const fixture = golden.attention_fixture;

    try std.testing.expectEqual(@as(usize, 2), fixture.num_attention_heads);
    try std.testing.expectEqual(@as(usize, 1), fixture.num_key_value_heads);
    try std.testing.expect(fixture.query_bias.len != 0 and fixture.key_bias.len != 0);
    try std.testing.expect(fixture.value_bias.len != 0 and fixture.output_bias.len != 0);
    try std.testing.expect(fixture.query_norm_weight.len != 0 and fixture.key_norm_weight.len != 0);

    const allocator = arena.allocator();
    const input = try flatten_f32_matrix(allocator, fixture.input);
    const query_weight = try flatten_f32_matrix(allocator, fixture.query_weight);
    const key_weight = try flatten_f32_matrix(allocator, fixture.key_weight);
    const value_weight = try flatten_f32_matrix(allocator, fixture.value_weight);
    const output_weight = try flatten_f32_matrix(allocator, fixture.output_weight);
    const cosine = try flatten_f32_matrix(allocator, fixture.cosine);
    const sine = try flatten_f32_matrix(allocator, fixture.sine);
    const expected = try flatten_f32_matrix(allocator, fixture.expected_output);
    const attention_mask = try flatten_bool_matrix(allocator, fixture.attention_mask);

    const token_count = fixture.input.len;
    const hidden_size = fixture.input[0].len;
    const actual = try allocator.alloc(f32, token_count * hidden_size);
    try attention(
        allocator,
        input,
        .{
            .query_weight = query_weight,
            .query_bias = fixture.query_bias,
            .key_weight = key_weight,
            .key_bias = fixture.key_bias,
            .value_weight = value_weight,
            .value_bias = fixture.value_bias,
            .query_norm_weight = fixture.query_norm_weight,
            .key_norm_weight = fixture.key_norm_weight,
            .output_weight = output_weight,
            .output_bias = fixture.output_bias,
        },
        .{
            .token_count = token_count,
            .hidden_size = hidden_size,
            .num_attention_heads = fixture.num_attention_heads,
            .num_key_value_heads = fixture.num_key_value_heads,
            .head_dim = fixture.head_dim,
            .rms_norm_epsilon = fixture.rms_norm_epsilon,
            .cosine = cosine,
            .sine = sine,
            .attention_mask = attention_mask,
        },
        actual,
    );

    var max_error: f64 = 0;
    for (0..token_count) |token_index| {
        var position_max: f64 = 0;
        for (0..hidden_size) |dimension| {
            position_max = @max(position_max, @abs(@as(f64, expected[token_index * hidden_size + dimension])));
        }
        const tolerance = 1e-4 * @max(1.0, position_max);
        for (0..hidden_size) |dimension| {
            const offset = token_index * hidden_size + dimension;
            const error_value = @abs(@as(f64, actual[offset]) - @as(f64, expected[offset]));
            max_error = @max(max_error, error_value);
            try std.testing.expect(error_value <= tolerance);
        }
    }
    std.debug.print("B5 maximum absolute attention-output error: {d:.12}\n", .{max_error});
}

test "attention returns zero context for a fully masked query before output bias" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 2, -3 };
    const projection = [_]f32{ 1, 0, 0, 1 };
    const output_bias = [_]f32{ 0.25, -0.5 };
    const cosine = [_]f32{ 1, 1 };
    const sine = [_]f32{ 0, 0 };
    const attention_mask = [_]bool{false};
    var output: [2]f32 = undefined;

    try attention(
        allocator,
        &input,
        .{
            .query_weight = &projection,
            .key_weight = &projection,
            .value_weight = &projection,
            .output_weight = &projection,
            .output_bias = &output_bias,
        },
        .{
            .token_count = 1,
            .hidden_size = 2,
            .num_attention_heads = 1,
            .num_key_value_heads = 1,
            .head_dim = 2,
            .rms_norm_epsilon = 1e-5,
            .cosine = &cosine,
            .sine = &sine,
            .attention_mask = &attention_mask,
        },
        &output,
    );
    try std.testing.expectEqualSlices(f32, &output_bias, &output);
}

test "causal and sliding-window masks match transformers golden masks" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const cases = [_]mask_case_t{
        .{ .path = "zig-out/golden/llama_default/reference.json" },
        .{ .path = "zig-out/golden/llama_llama3/reference.json" },
        .{ .path = "zig-out/golden/qwen2/reference.json" },
        .{ .path = "zig-out/golden/qwen3/reference.json" },
        .{ .path = "zig-out/golden/qwen3_attention_bias/reference.json" },
        .{ .path = "zig-out/golden/mistral/reference.json", .sliding_window = 3 },
    };

    for (cases) |case| {
        const golden = try read_mask_golden(arena.allocator(), case.path);
        const expected = golden.attention_mask;
        const query_count = expected.len;
        try std.testing.expect(query_count != 0);
        const key_count = expected[0].len;
        try std.testing.expect(key_count != 0);
        if (case.sliding_window) |window| try std.testing.expect(query_count > window);

        const query_positions = try arena.allocator().alloc(usize, query_count);
        const key_positions = try arena.allocator().alloc(usize, key_count);
        const actual = try arena.allocator().alloc(bool, query_count * key_count);
        for (query_positions, 0..) |*position, index| position.* = index;
        for (key_positions, 0..) |*position, index| position.* = index;
        causal_mask(query_positions, key_positions, case.sliding_window, actual);

        for (expected, 0..) |row, query_index| {
            try std.testing.expectEqual(key_count, row.len);
            try std.testing.expectEqualSlices(bool, row, actual[query_index * key_count ..][0..key_count]);
        }
    }
}

fn read_rope_golden(allocator: std.mem.Allocator) !rope_tables_t {
    const raw_json = std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        "zig-out/golden/rope_tables.json",
        allocator,
        .limited(100 * 1024 * 1024),
    ) catch |err| return skip_without_golden(err, "zig-out/golden/rope_tables.json");
    return std.json.parseFromSliceLeaky(rope_tables_t, allocator, raw_json, .{});
}

fn skip_without_golden(err: anyerror, path: []const u8) anyerror {
    if (err == error.FileNotFound) {
        std.debug.print("SKIP golden acceptance: missing {s}; generate it with tools/golden_helper.py\n", .{path});
        return error.SkipZigTest;
    }
    return err;
}

fn check_rope_default_case(case: rope_case_t) !f64 {
    const reference = case.reference;
    try std.testing.expectEqual(case.head_dim, reference.head_dim);
    try std.testing.expectEqualStrings("default", reference.rope_parameters.rope_type);
    try std.testing.expectEqual(@as(f64, case.rope_theta), reference.rope_parameters.rope_theta);
    try std.testing.expectEqual(@as(usize, 4096), reference.cos.len);
    try std.testing.expectEqual(@as(usize, 4096), reference.sin.len);

    const output_len = 4096 * case.head_dim;
    const allocator = std.testing.allocator;
    const inverse_frequencies = try allocator.alloc(f32, case.head_dim / 2);
    defer allocator.free(inverse_frequencies);
    const cosine = try allocator.alloc(f32, output_len);
    defer allocator.free(cosine);
    const sine = try allocator.alloc(f32, output_len);
    defer allocator.free(sine);

    rope_default(case.head_dim, case.rope_theta, 4096, inverse_frequencies, cosine, sine);

    var max_error: f64 = 0;
    for (0..4096) |position| {
        try std.testing.expectEqual(case.head_dim, reference.cos[position].len);
        try std.testing.expectEqual(case.head_dim, reference.sin[position].len);
        for (0..case.head_dim) |index| {
            const cosine_error = @abs(@as(f64, cosine[position * case.head_dim + index]) - reference.cos[position][index]);
            const sine_error = @abs(@as(f64, sine[position * case.head_dim + index]) - reference.sin[position][index]);
            try std.testing.expect(cosine_error <= 1e-6);
            try std.testing.expect(sine_error <= 1e-6);
            max_error = @max(max_error, @max(cosine_error, sine_error));
        }
    }
    return max_error;
}

const rope_parameters_t = struct {
    rope_type: []const u8,
    rope_theta: f64,
    factor: ?f64 = null,
    low_freq_factor: ?f64 = null,
    high_freq_factor: ?f64 = null,
    original_max_position_embeddings: ?usize = null,
};

const rope_table_t = struct {
    head_dim: usize,
    rope_parameters: rope_parameters_t,
    inv_freq: []const f64,
    cos: []const []const f64,
    sin: []const []const f64,
};

const rope_tables_t = struct {
    transformers_version: []const u8,
    position_start: usize,
    position_end_inclusive: usize,
    tables: struct {
        default_head_dim_64_theta_10000: rope_table_t,
        default_head_dim_64_theta_500000: rope_table_t,
        default_head_dim_128_theta_10000: rope_table_t,
        default_head_dim_128_theta_500000: rope_table_t,
        llama3_head_dim_64_theta_500000: rope_table_t,
    },
};

const rope_case_t = struct {
    head_dim: usize,
    rope_theta: f32,
    reference: rope_table_t,
};

const attention_golden_t = struct {
    transformers_version: []const u8,
    attention_fixture: attention_fixture_t,
};

const mask_golden_t = struct {
    attention_mask: []const []const bool,
};

const mask_case_t = struct {
    path: []const u8,
    sliding_window: ?usize = null,
};

const attention_fixture_t = struct {
    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_epsilon: f32,
    input: []const []const f32,
    query_weight: []const []const f32,
    query_bias: []const f32,
    key_weight: []const []const f32,
    key_bias: []const f32,
    value_weight: []const []const f32,
    value_bias: []const f32,
    query_norm_weight: []const f32,
    key_norm_weight: []const f32,
    output_weight: []const []const f32,
    output_bias: []const f32,
    cosine: []const []const f32,
    sine: []const []const f32,
    attention_mask: []const []const bool,
    expected_output: []const []const f32,
};

fn flatten_f32_matrix(allocator: std.mem.Allocator, rows: []const []const f32) ![]f32 {
    var element_count: usize = 0;
    for (rows) |row| element_count += row.len;
    const values = try allocator.alloc(f32, element_count);
    var offset: usize = 0;
    for (rows) |row| {
        @memcpy(values[offset .. offset + row.len], row);
        offset += row.len;
    }
    return values;
}

fn flatten_bool_matrix(allocator: std.mem.Allocator, rows: []const []const bool) ![]bool {
    var element_count: usize = 0;
    for (rows) |row| element_count += row.len;
    const values = try allocator.alloc(bool, element_count);
    var offset: usize = 0;
    for (rows) |row| {
        @memcpy(values[offset .. offset + row.len], row);
        offset += row.len;
    }
    return values;
}

fn read_mask_golden(allocator: std.mem.Allocator, path: []const u8) !mask_golden_t {
    const raw_json = std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        path,
        allocator,
        .limited(10 * 1024 * 1024),
    ) catch |err| return skip_without_golden(err, path);
    return std.json.parseFromSliceLeaky(mask_golden_t, allocator, raw_json, .{
        .ignore_unknown_fields = true,
    });
}
