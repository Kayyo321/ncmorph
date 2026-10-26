//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! F32 forward pass for the supported decoder-only source models.

const std = @import("std");
const config = @import("hf_config").config;
const kernels = @import("kernels");

pub const layer_weights_t = struct {
    input_norm: []const f32,
    attention: kernels.attention_weights_t,
    post_attention_norm: []const f32,
    gate_weight: []const f32,
    gate_bias: ?[]const f32 = null,
    up_weight: []const f32,
    up_bias: ?[]const f32 = null,
    down_weight: []const f32,
    down_bias: ?[]const f32 = null,
};

pub const model_weights_t = struct {
    embed_tokens: []const f32,
    final_norm: []const f32,
    lm_head: ?[]const f32,
    layers: []const layer_weights_t,
    bf16_compute: bool = false,
};

pub const decoder_trace_t = struct {
    token_count: usize,
    hidden_size: usize,
    intermediate_size: usize,
    query_width: usize,
    layer_count: usize,
    block_input: []f32,
    attention_context: []f32,
    input_norm: []f32,
    attention_output: []f32,
    attention_residual: []f32,
    post_attention_norm: []f32,
    gate_projection: []f32,
    up_projection: []f32,
    gate_up_product: []f32,
    mlp_output: []f32,
    block_output: []f32,
};

/// Computes logits for one prefill sequence of token ids, returning row-major [T, vocab_size].
pub fn forward(
    allocator: std.mem.Allocator,
    spec: config.spec_t,
    weights: model_weights_t,
    token_ids: []const usize,
    logits: []f32,
) !void {
    try forward_with_trace(allocator, spec, weights, token_ids, logits, null);
}

/// Computes logits and optionally stores each block's output in layer-major
/// `[layer, token, hidden]` order for acceptance diagnostics.
pub fn forward_with_trace(
    allocator: std.mem.Allocator,
    spec: config.spec_t,
    weights: model_weights_t,
    token_ids: []const usize,
    logits: []f32,
    trace: ?decoder_trace_t,
) !void {
    if (token_ids.len == 0) return error.EmptyInput;

    const hidden_size: usize = @intCast(spec.hidden_size);
    const vocab_size: usize = @intCast(spec.vocab_size);
    const intermediate_size: usize = @intCast(spec.intermediate_size);
    const token_count = token_ids.len;
    const head_dim: usize = @intCast(spec.head_dim);
    const attention_heads: usize = @intCast(spec.num_attention_heads);
    const kv_heads: usize = @intCast(spec.num_key_value_heads);
    const layer_count: usize = @intCast(spec.num_hidden_layers);
    const query_width = attention_heads * head_dim;
    std.debug.assert(weights.embed_tokens.len == vocab_size * hidden_size);
    std.debug.assert(weights.final_norm.len == hidden_size);
    std.debug.assert(weights.layers.len == layer_count);
    std.debug.assert(logits.len == token_count * vocab_size);
    if (trace) |values| {
        std.debug.assert(values.token_count == token_count);
        std.debug.assert(values.hidden_size == hidden_size);
        std.debug.assert(values.intermediate_size == intermediate_size);
        std.debug.assert(values.query_width == query_width);
        std.debug.assert(values.layer_count == layer_count);
        std.debug.assert(values.block_input.len == token_count * hidden_size);
        std.debug.assert(values.attention_context.len == layer_count * token_count * attention_heads * head_dim);
        std.debug.assert(values.input_norm.len == layer_count * token_count * hidden_size);
        std.debug.assert(values.attention_output.len == layer_count * token_count * hidden_size);
        std.debug.assert(values.attention_residual.len == layer_count * token_count * hidden_size);
        std.debug.assert(values.post_attention_norm.len == layer_count * token_count * hidden_size);
        std.debug.assert(values.gate_projection.len == layer_count * token_count * intermediate_size);
        std.debug.assert(values.up_projection.len == layer_count * token_count * intermediate_size);
        std.debug.assert(values.gate_up_product.len == layer_count * token_count * intermediate_size);
        std.debug.assert(values.mlp_output.len == layer_count * token_count * hidden_size);
        std.debug.assert(values.block_output.len == layer_count * token_count * hidden_size);
    }

    const hidden = try allocator.alloc(f32, token_count * hidden_size);
    defer allocator.free(hidden);
    for (token_ids, 0..) |token_id, token_index| {
        if (token_id >= vocab_size) return error.TokenIdOutOfRange;
        kernels.embedding_row(
            weights.embed_tokens,
            vocab_size,
            hidden_size,
            token_id,
            hidden[token_index * hidden_size ..][0..hidden_size],
        );
    }
    if (weights.bf16_compute) round_bf16(hidden);
    if (trace) |values| copy_trace(values.block_input, 0, hidden);

    const position_count = token_count;
    const rotary_width = head_dim;
    const cosine = try allocator.alloc(f32, position_count * rotary_width);
    defer allocator.free(cosine);
    const sine = try allocator.alloc(f32, position_count * rotary_width);
    defer allocator.free(sine);
    const inverse_frequencies = try allocator.alloc(f32, head_dim / 2);
    defer allocator.free(inverse_frequencies);
    switch (spec.rope_scaling) {
        .Default => kernels.rope_default(
            head_dim,
            @floatCast(spec.rope_theta),
            position_count,
            inverse_frequencies,
            cosine,
            sine,
        ),
        .Llama3 => |parameters| {
            kernels.rope_llama3(
                head_dim,
                @floatCast(spec.rope_theta),
                @floatCast(parameters.factor),
                @floatCast(parameters.low_freq_factor),
                @floatCast(parameters.high_freq_factor),
                @intCast(parameters.original_max_position_embeddings),
                inverse_frequencies,
            );
            kernels.rope_tables(inverse_frequencies, position_count, cosine, sine);
        },
    }

    const positions = try allocator.alloc(usize, token_count);
    defer allocator.free(positions);
    for (positions, 0..) |*position, index| position.* = index;
    const attention_mask = try allocator.alloc(bool, token_count * token_count);
    defer allocator.free(attention_mask);
    kernels.causal_mask(positions, positions, spec.sliding_window, attention_mask);

    const normalized = try allocator.alloc(f32, token_count * hidden_size);
    defer allocator.free(normalized);
    const attention_output = try allocator.alloc(f32, token_count * hidden_size);
    defer allocator.free(attention_output);
    const gate = try allocator.alloc(f32, token_count * intermediate_size);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, token_count * intermediate_size);
    defer allocator.free(up);
    const mlp_output = try allocator.alloc(f32, token_count * hidden_size);
    defer allocator.free(mlp_output);

    for (weights.layers, 0..) |layer, layer_index| {
        const trace_hidden_offset = layer_index * token_count * hidden_size;
        const trace_intermediate_offset = layer_index * token_count * intermediate_size;
        normalize_rows(hidden, layer.input_norm, normalized, token_count, hidden_size, @floatCast(spec.rms_norm_eps), weights.bf16_compute);
        if (trace) |values| copy_trace(values.input_norm, trace_hidden_offset, normalized);
        const attention_config: kernels.attention_config_t = .{
            .token_count = token_count,
            .hidden_size = hidden_size,
            .num_attention_heads = attention_heads,
            .num_key_value_heads = kv_heads,
            .head_dim = head_dim,
            .rms_norm_epsilon = @floatCast(spec.rms_norm_eps),
            .cosine = cosine,
            .sine = sine,
            .attention_mask = attention_mask,
        };
        if (weights.bf16_compute) {
            if (trace) |values| {
                const context_offset = layer_index * token_count * query_width;
                try kernels.attention_bf16_with_context(allocator, normalized, layer.attention, attention_config, attention_output, values.attention_context[context_offset..][0 .. token_count * query_width]);
            } else {
                try kernels.attention_bf16(allocator, normalized, layer.attention, attention_config, attention_output);
            }
        } else {
            if (trace) |values| {
                const context_offset = layer_index * token_count * query_width;
                try kernels.attention_with_context(allocator, normalized, layer.attention, attention_config, attention_output, values.attention_context[context_offset..][0 .. token_count * query_width]);
            } else {
                try kernels.attention(allocator, normalized, layer.attention, attention_config, attention_output);
            }
        }
        if (trace) |values| copy_trace(values.attention_output, trace_hidden_offset, attention_output);
        add_rows(hidden, attention_output, token_count, hidden_size, weights.bf16_compute);
        if (trace) |values| copy_trace(values.attention_residual, trace_hidden_offset, hidden);

        normalize_rows(hidden, layer.post_attention_norm, normalized, token_count, hidden_size, @floatCast(spec.rms_norm_eps), weights.bf16_compute);
        if (trace) |values| copy_trace(values.post_attention_norm, trace_hidden_offset, normalized);
        project(normalized, layer.gate_weight, layer.gate_bias, gate, token_count, hidden_size, intermediate_size, weights.bf16_compute);
        if (trace) |values| copy_trace(values.gate_projection, trace_intermediate_offset, gate);
        project(normalized, layer.up_weight, layer.up_bias, up, token_count, hidden_size, intermediate_size, weights.bf16_compute);
        if (trace) |values| copy_trace(values.up_projection, trace_intermediate_offset, up);
        if (weights.bf16_compute) kernels.silu_bf16(gate, gate) else kernels.silu(gate, gate);
        for (gate, up) |*gate_value, up_value| gate_value.* = maybe_bf16(gate_value.* * up_value, weights.bf16_compute);
        if (trace) |values| copy_trace(values.gate_up_product, trace_intermediate_offset, gate);
        project(gate, layer.down_weight, layer.down_bias, mlp_output, token_count, intermediate_size, hidden_size, weights.bf16_compute);
        if (trace) |values| copy_trace(values.mlp_output, trace_hidden_offset, mlp_output);
        add_rows(hidden, mlp_output, token_count, hidden_size, weights.bf16_compute);
        if (trace) |values| copy_trace(values.block_output, trace_hidden_offset, hidden);
    }

    normalize_rows(hidden, weights.final_norm, normalized, token_count, hidden_size, @floatCast(spec.rms_norm_eps), weights.bf16_compute);
    const head_weight = if (spec.tie_word_embeddings)
        weights.embed_tokens
    else
        (weights.lm_head orelse return error.MissingLmHead);
    if (weights.bf16_compute) {
        kernels.matmul_bf16(normalized, head_weight, logits, token_count, hidden_size, vocab_size);
    } else {
        kernels.matmul(normalized, head_weight, logits, token_count, hidden_size, vocab_size);
    }
}

fn copy_trace(destination: []f32, offset: usize, source: []const f32) void {
    for (source, 0..) |value, index| destination[offset + index] = value;
}

fn normalize_rows(
    input: []const f32,
    weight: []const f32,
    output: []f32,
    row_count: usize,
    row_width: usize,
    epsilon: f32,
    bf16_compute: bool,
) void {
    for (0..row_count) |row| {
        const begin = row * row_width;
        if (bf16_compute) {
            kernels.rms_norm_bf16(input[begin..][0..row_width], weight, output[begin..][0..row_width], epsilon);
        } else {
            kernels.rms_norm(input[begin..][0..row_width], weight, output[begin..][0..row_width], epsilon);
        }
    }
}

fn project(
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    output: []f32,
    row_count: usize,
    input_width: usize,
    output_width: usize,
    bf16_compute: bool,
) void {
    if (bf16_compute) {
        if (bias) |values| {
            kernels.matmul_bf16_bias(input, weight, values, output, row_count, input_width, output_width);
        } else {
            kernels.matmul_bf16(input, weight, output, row_count, input_width, output_width);
        }
    } else {
        kernels.matmul(input, weight, output, row_count, input_width, output_width);
        if (bias) |values| {
            for (0..row_count) |row| {
                for (0..output_width) |column| output[row * output_width + column] += values[column];
            }
        }
    }
}

fn add_rows(destination: []f32, addition: []const f32, row_count: usize, row_width: usize, bf16_compute: bool) void {
    std.debug.assert(destination.len == row_count * row_width and addition.len == destination.len);
    for (destination, addition) |*value, addend| value.* = maybe_bf16(value.* + addend, bf16_compute);
}

fn round_bf16(values: []f32) void {
    for (values) |*value| value.* = kernels.round_bf16(value.*);
}

fn maybe_bf16(value: f32, enabled: bool) f32 {
    return if (enabled) kernels.round_bf16(value) else value;
}

test "tiny decoder forward returns token-major vocabulary logits" {
    const spec: config.spec_t = .{
        .family = .Llama,
        .vocab_size = 4,
        .hidden_size = 2,
        .intermediate_size = 3,
        .num_hidden_layers = 1,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .head_dim = 2,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10_000,
        .rope_scaling = .Default,
        .qkv_bias = false,
        .o_bias = false,
        .mlp_bias = false,
        .qk_norm = false,
        .tie_word_embeddings = true,
        .sliding_window = null,
    };
    const attention = kernels.attention_weights_t{
        .query_weight = &.{ 0, 0, 0, 0 },
        .key_weight = &.{ 0, 0, 0, 0 },
        .value_weight = &.{ 1, 0, 0, 1 },
        .output_weight = &.{ 1, 0, 0, 1 },
    };
    const layer = layer_weights_t{
        .input_norm = &.{ 1, 1 },
        .attention = attention,
        .post_attention_norm = &.{ 1, 1 },
        .gate_weight = &.{ 0, 0, 0, 0, 0, 0 },
        .up_weight = &.{ 0, 0, 0, 0, 0, 0 },
        .down_weight = &.{ 0, 0, 0, 0, 0, 0 },
    };
    const model_weights = model_weights_t{
        .embed_tokens = &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 },
        .final_norm = &.{ 1, 1 },
        .lm_head = null,
        .layers = &.{layer},
    };
    const token_ids = [_]usize{ 1, 3 };
    var logits: [8]f32 = undefined;
    var block_input: [4]f32 = undefined;
    var attention_context = [4]f32{ 123, 123, 123, 123 };
    var hidden_trace: [4]f32 = undefined;
    var input_norm_trace: [4]f32 = undefined;
    var intermediate_trace: [6]f32 = undefined;
    const trace: decoder_trace_t = .{
        .token_count = token_ids.len,
        .hidden_size = 2,
        .intermediate_size = 3,
        .query_width = 2,
        .layer_count = 1,
        .block_input = &block_input,
        .attention_context = &attention_context,
        .input_norm = &input_norm_trace,
        .attention_output = &hidden_trace,
        .attention_residual = &hidden_trace,
        .post_attention_norm = &hidden_trace,
        .gate_projection = &intermediate_trace,
        .up_projection = &intermediate_trace,
        .gate_up_product = &intermediate_trace,
        .mlp_output = &hidden_trace,
        .block_output = &hidden_trace,
    };
    try forward_with_trace(std.testing.allocator, spec, model_weights, &token_ids, &logits, trace);
    try std.testing.expectEqual(@as(usize, token_ids.len * @as(usize, @intCast(spec.vocab_size))), logits.len);
    for (logits) |value| try std.testing.expect(std.math.isFinite(value));
    try std.testing.expectApproxEqAbs(input_norm_trace[0], attention_context[0], 1e-6);
    try std.testing.expectApproxEqAbs(input_norm_trace[1], attention_context[1], 1e-6);
    try std.testing.expectApproxEqAbs((input_norm_trace[0] + input_norm_trace[2]) * 0.5, attention_context[2], 1e-6);
    try std.testing.expectApproxEqAbs((input_norm_trace[1] + input_norm_trace[3]) * 0.5, attention_context[3], 1e-6);
}
