//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! NCM transformer body for source-mode reduction and hidden-state checks.

const std = @import("std");
const decoder = @import("decoder");
const config = @import("hf_config").config;
const kernels = @import("kernels");
const ncm_norm = @import("ncm_norm");

/// Runs the causal source-mode body and stores embedding, intermediate block,
/// and final-norm states in `[layer_count + 1, token_count, hidden_size]` order.
pub fn forward_source_mode(
    allocator: std.mem.Allocator,
    spec: config.spec_t,
    weights: decoder.model_weights_t,
    token_ids: []const usize,
    hidden_states: []f32,
) !void {
    if (token_ids.len == 0) return error.EmptyInput;
    if (weights.bf16_compute) return error.Bf16SourceModeNotImplemented;

    const hidden_size: usize = @intCast(spec.hidden_size);
    const vocab_size: usize = @intCast(spec.vocab_size);
    const intermediate_size: usize = @intCast(spec.intermediate_size);
    const head_dim: usize = @intCast(spec.head_dim);
    const query_heads: usize = @intCast(spec.num_attention_heads);
    const key_value_heads: usize = @intCast(spec.num_key_value_heads);
    const layer_count: usize = @intCast(spec.num_hidden_layers);
    const token_count = token_ids.len;
    const row_size = token_count * hidden_size;
    std.debug.assert(hidden_states.len == (layer_count + 1) * row_size);
    std.debug.assert(weights.layers.len == layer_count);
    std.debug.assert(weights.embed_tokens.len == vocab_size * hidden_size);
    std.debug.assert(weights.final_norm.len == hidden_size);

    const hidden = try allocator.alloc(f32, row_size);
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
    @memcpy(hidden_states[0..row_size], hidden);

    const cosine = try allocator.alloc(f32, token_count * head_dim);
    defer allocator.free(cosine);
    const sine = try allocator.alloc(f32, cosine.len);
    defer allocator.free(sine);
    const inverse_frequencies = try allocator.alloc(f32, head_dim / 2);
    defer allocator.free(inverse_frequencies);
    switch (spec.rope_scaling) {
        .Default => kernels.rope_default(head_dim, @floatCast(spec.rope_theta), token_count, inverse_frequencies, cosine, sine),
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
            kernels.rope_tables(inverse_frequencies, token_count, cosine, sine);
        },
    }

    const positions = try allocator.alloc(usize, token_count);
    defer allocator.free(positions);
    for (positions, 0..) |*position, index| position.* = index;
    const attention_mask = try allocator.alloc(bool, token_count * token_count);
    defer allocator.free(attention_mask);
    kernels.causal_mask(positions, positions, spec.sliding_window, attention_mask);

    const normalized = try allocator.alloc(f32, row_size);
    defer allocator.free(normalized);
    const attention_output = try allocator.alloc(f32, row_size);
    defer allocator.free(attention_output);
    const gate = try allocator.alloc(f32, token_count * intermediate_size);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, gate.len);
    defer allocator.free(up);
    const mlp_output = try allocator.alloc(f32, row_size);
    defer allocator.free(mlp_output);
    const attention_config: kernels.attention_config_t = .{
        .token_count = token_count,
        .hidden_size = hidden_size,
        .num_attention_heads = query_heads,
        .num_key_value_heads = key_value_heads,
        .head_dim = head_dim,
        .rms_norm_epsilon = @floatCast(spec.rms_norm_eps),
        .cosine = cosine,
        .sine = sine,
        .attention_mask = attention_mask,
    };

    for (weights.layers, 0..) |layer, layer_index| {
        ncm_norm.rms_norm_rows(hidden, layer.input_norm, normalized, token_count, hidden_size, @floatCast(spec.rms_norm_eps));
        try kernels.attention(allocator, normalized, layer.attention, attention_config, attention_output);
        add_rows(hidden, attention_output, token_count, hidden_size);

        ncm_norm.rms_norm_rows(hidden, layer.post_attention_norm, normalized, token_count, hidden_size, @floatCast(spec.rms_norm_eps));
        project(normalized, layer.gate_weight, layer.gate_bias, gate, token_count, hidden_size, intermediate_size);
        project(normalized, layer.up_weight, layer.up_bias, up, token_count, hidden_size, intermediate_size);
        kernels.silu(gate, gate);
        for (gate, up) |*gate_value, up_value| gate_value.* *= up_value;
        project(gate, layer.down_weight, layer.down_bias, mlp_output, token_count, intermediate_size, hidden_size);
        add_rows(hidden, mlp_output, token_count, hidden_size);
        if (layer_index + 1 < layer_count) {
            @memcpy(hidden_states[(layer_index + 1) * row_size ..][0..row_size], hidden);
        }
    }

    ncm_norm.rms_norm_rows(
        hidden,
        weights.final_norm,
        hidden_states[layer_count * row_size ..][0..row_size],
        token_count,
        hidden_size,
        @floatCast(spec.rms_norm_eps),
    );
}

fn project(
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    output: []f32,
    row_count: usize,
    input_width: usize,
    output_width: usize,
) void {
    kernels.matmul(input, weight, output, row_count, input_width, output_width);
    if (bias) |values| {
        for (0..row_count) |row| {
            const offset = row * output_width;
            for (0..output_width) |column| output[offset + column] += values[column];
        }
    }
}

fn add_rows(destination: []f32, source: []const f32, row_count: usize, row_width: usize) void {
    for (0..row_count) |row| {
        const offset = row * row_width;
        for (0..row_width) |column| destination[offset + column] += source[offset + column];
    }
}
