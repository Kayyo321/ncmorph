//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! NCM forward body with level inputs, learned level-0 queries, and an EOS prefix.

const std = @import("std");
const kernels = @import("kernels");
const ncm_mask = @import("ncm_mask");
const ncm_norm = @import("ncm_norm");
const ncm_position = @import("ncm_position");
const Allocator = std.mem.Allocator;

pub const rope_type_t = enum { Default, Llama3 };

pub const config_t = struct {
    hidden_size: usize,
    vq_width: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    rms_norm_epsilon: f32,
    rope_type: rope_type_t,
    rope_theta: f32,
    rope_factor: f32 = 1,
    rope_low_freq_factor: f32 = 1,
    rope_high_freq_factor: f32 = 4,
    rope_original_max_position_embeddings: usize = 8192,
    level_lengths: []const usize,
    prefix_length: usize,
    query_level0: bool,
    use_level_encoding: bool = true,
    source_causal: bool = false,
    sliding_window: ?usize = null,
};

pub const layer_weights_t = struct {
    fused_qkv_weight: []const f32,
    fused_qkv_bias: ?[]const f32 = null,
    output_weight: []const f32,
    output_bias: ?[]const f32 = null,
    query_norm_weight: ?[]const f32 = null,
    key_norm_weight: ?[]const f32 = null,
    ln_1_weight: []const f32,
    ln_1_bias: ?[]const f32 = null,
    gate_weight: []const f32,
    gate_bias: ?[]const f32 = null,
    up_weight: []const f32,
    up_bias: ?[]const f32 = null,
    down_weight: []const f32,
    down_bias: ?[]const f32 = null,
    ln_2_weight: []const f32,
    ln_2_bias: ?[]const f32 = null,
};

pub const post_upsample_conv_t = struct {
    weight: []const f32,
    bias: []const f32,
};

pub const weights_t = struct {
    prefix_token_embedding: []const f32,
    prefix_vocab_size: usize,
    input_projection: []const f32,
    input_projection_bias: []const f32,
    level_embedding: []const f32,
    l0_queries: []const f32,
    final_norm_weight: []const f32,
    final_norm_bias: ?[]const f32 = null,
    layers: []const layer_weights_t,
    post_upsample_convs: []const post_upsample_conv_t = &.{},
};

/// Runs the NCM sequence body for one prefix and one continuous input per target
/// position. Level 0 is replaced by `l0_queries` when `query_level0` is enabled.
/// Returns final-normalized hidden states in `[prefix | levels]` order.
pub fn forward(
    allocator: Allocator,
    config: config_t,
    weights: weights_t,
    prefix_token_ids: []const usize,
    prefix_pad_mask: []const bool,
    target_inputs: []const f32,
    output: []f32,
) !void {
    try forward_with_trace(allocator, config, weights, prefix_token_ids, prefix_pad_mask, target_inputs, output, null);
}

pub fn forward_with_trace(
    allocator: Allocator,
    config: config_t,
    weights: weights_t,
    prefix_token_ids: []const usize,
    prefix_pad_mask: []const bool,
    target_inputs: []const f32,
    output: []f32,
    hidden_states: ?[]f32,
) !void {
    if (prefix_token_ids.len != config.prefix_length) return error.InvalidPrefixLength;
    if (prefix_pad_mask.len != 0 and prefix_pad_mask.len != config.prefix_length) return error.InvalidPrefixMask;
    if (config.level_lengths.len == 0) return error.InvalidNcmMode;
    if (config.hidden_size == 0 or config.vq_width == 0 or config.head_dim == 0 or config.intermediate_size == 0) return error.InvalidNcmDimensions;
    if (config.num_attention_heads == 0 or config.num_key_value_heads == 0 or config.head_dim % 2 != 0) return error.InvalidNcmDimensions;
    if (config.num_attention_heads % config.num_key_value_heads != 0) return error.InvalidNcmDimensions;

    var target_length: usize = 0;
    for (config.level_lengths) |level_length| target_length = try std.math.add(usize, target_length, level_length);
    const sequence_length = try std.math.add(usize, config.prefix_length, target_length);
    const sequence_width = try std.math.mul(usize, sequence_length, config.hidden_size);
    const target_width = try std.math.mul(usize, target_length, config.vq_width);
    const prefix_table_size = try std.math.mul(usize, weights.prefix_vocab_size, config.vq_width);
    const query_width = try std.math.mul(usize, config.num_attention_heads, config.head_dim);
    const key_value_width = try std.math.mul(usize, config.num_key_value_heads, config.head_dim);
    const fused_width = try std.math.add(usize, query_width, try std.math.mul(usize, 2, key_value_width));
    const embedding_size = try std.math.mul(usize, config.level_lengths.len, config.hidden_size);

    if (target_inputs.len != target_width or output.len != sequence_width) return error.InvalidNcmTensorSize;
    if (weights.prefix_token_embedding.len != prefix_table_size) return error.InvalidNcmTensorSize;
    if (weights.input_projection.len != try std.math.mul(usize, config.hidden_size, config.vq_width) or weights.input_projection_bias.len != config.hidden_size) return error.InvalidNcmTensorSize;
    if (weights.level_embedding.len != embedding_size) return error.InvalidNcmTensorSize;
    if (weights.l0_queries.len != (if (config.query_level0) try std.math.mul(usize, config.level_lengths[0], config.vq_width) else 0)) return error.InvalidNcmTensorSize;
    if (weights.final_norm_weight.len != config.hidden_size or weights.layers.len == 0) return error.InvalidNcmTensorSize;
    if (weights.post_upsample_convs.len != 0 and weights.post_upsample_convs.len != config.level_lengths.len) return error.InvalidNcmTensorSize;

    const input_size = try std.math.mul(usize, sequence_length, config.vq_width);
    const projected = try allocator.alloc(f32, input_size);
    defer allocator.free(projected);
    for (prefix_token_ids, 0..) |token_id, prefix_index| {
        if (token_id >= weights.prefix_vocab_size) return error.PrefixTokenOutOfRange;
        const row_start = token_id * config.vq_width;
        const target_start = prefix_index * config.vq_width;
        @memcpy(projected[target_start..][0..config.vq_width], weights.prefix_token_embedding[row_start..][0..config.vq_width]);
    }
    @memcpy(projected[config.prefix_length * config.vq_width ..][0..target_width], target_inputs);
    if (config.query_level0) {
        @memcpy(
            projected[config.prefix_length * config.vq_width ..][0..weights.l0_queries.len],
            weights.l0_queries,
        );
    }
    if (weights.post_upsample_convs.len != 0) {
        var target_offset: usize = 0;
        for (config.level_lengths, weights.post_upsample_convs) |level_length, convolution| {
            const chunk_offset = (config.prefix_length * config.vq_width) + target_offset * config.vq_width;
            try apply_post_upsample_conv(
                projected[chunk_offset..][0 .. level_length * config.vq_width],
                convolution,
                level_length,
                config.vq_width,
                allocator,
            );
            target_offset += level_length;
        }
    }

    const hidden = try allocator.alloc(f32, sequence_width);
    defer allocator.free(hidden);
    project(projected, weights.input_projection, weights.input_projection_bias, hidden, sequence_length, config.vq_width, config.hidden_size);
    var level_offset = config.prefix_length;
    for (config.level_lengths, 0..) |level_length, level_index| {
        if (!config.use_level_encoding) break;
        const embedding_offset = level_index * config.hidden_size;
        for (level_offset..level_offset + level_length) |position| {
            const row_offset = position * config.hidden_size;
            for (0..config.hidden_size) |column| hidden[row_offset + column] += weights.level_embedding[embedding_offset + column];
        }
        level_offset += level_length;
    }
    if (hidden_states) |states| {
        if (states.len != try std.math.mul(usize, sequence_width, weights.layers.len + 1)) return error.InvalidNcmTensorSize;
        @memcpy(states[0..sequence_width], hidden);
    }

    const positions = try allocator.alloc(usize, sequence_length);
    defer allocator.free(positions);
    try ncm_position.fill_level_positions(config.prefix_length, config.level_lengths, positions);
    const rope_table_width = try std.math.mul(usize, sequence_length, config.head_dim);
    const cosine = try allocator.alloc(f32, rope_table_width);
    defer allocator.free(cosine);
    const sine = try allocator.alloc(f32, rope_table_width);
    defer allocator.free(sine);
    switch (config.rope_type) {
        .Default => try ncm_position.default_tables_at_positions(allocator, config.head_dim, config.rope_theta, positions, cosine, sine),
        .Llama3 => try ncm_position.llama3_tables_at_positions(
            allocator,
            config.head_dim,
            config.rope_theta,
            config.rope_factor,
            config.rope_low_freq_factor,
            config.rope_high_freq_factor,
            config.rope_original_max_position_embeddings,
            positions,
            cosine,
            sine,
        ),
    }

    const attention_mask = try allocator.alloc(bool, try std.math.mul(usize, sequence_length, sequence_length));
    defer allocator.free(attention_mask);
    if (config.source_causal) {
        for (0..sequence_length) |query| for (0..sequence_length) |key| {
            attention_mask[query * sequence_length + key] = key <= query and
                (config.sliding_window == null or query - key < config.sliding_window.?);
        };
    } else {
        try ncm_mask.block_diagonal_mask(config.prefix_length, config.level_lengths, prefix_pad_mask, attention_mask);
    }
    const normalized = try allocator.alloc(f32, sequence_width);
    defer allocator.free(normalized);
    const attention_output = try allocator.alloc(f32, sequence_width);
    defer allocator.free(attention_output);
    const gate = try allocator.alloc(f32, try std.math.mul(usize, sequence_length, config.intermediate_size));
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, gate.len);
    defer allocator.free(up);
    const mlp_output = try allocator.alloc(f32, sequence_width);
    defer allocator.free(mlp_output);
    const attention_config: kernels.attention_config_t = .{
        .token_count = sequence_length,
        .hidden_size = config.hidden_size,
        .num_attention_heads = config.num_attention_heads,
        .num_key_value_heads = config.num_key_value_heads,
        .head_dim = config.head_dim,
        .rms_norm_epsilon = config.rms_norm_epsilon,
        .cosine = cosine,
        .sine = sine,
        .attention_mask = attention_mask,
    };

    for (weights.layers, 0..) |layer, layer_index| {
        ncm_norm.rms_norm_rows_with_bias(hidden, layer.ln_1_weight, layer.ln_1_bias, normalized, sequence_length, config.hidden_size, config.rms_norm_epsilon);
        const attention_weights = split_attention_weights(config, layer, fused_width, query_width, key_value_width);
        try kernels.attention(allocator, normalized, attention_weights, attention_config, attention_output);
        add_rows(hidden, attention_output);

        ncm_norm.rms_norm_rows_with_bias(hidden, layer.ln_2_weight, layer.ln_2_bias, normalized, sequence_length, config.hidden_size, config.rms_norm_epsilon);
        project(normalized, layer.gate_weight, layer.gate_bias, gate, sequence_length, config.hidden_size, config.intermediate_size);
        project(normalized, layer.up_weight, layer.up_bias, up, sequence_length, config.hidden_size, config.intermediate_size);
        kernels.silu(gate, gate);
        for (gate, up) |*gate_value, up_value| gate_value.* *= up_value;
        project(gate, layer.down_weight, layer.down_bias, mlp_output, sequence_length, config.intermediate_size, config.hidden_size);
        add_rows(hidden, mlp_output);
        if (hidden_states) |states| {
            if (layer_index + 1 < weights.layers.len) @memcpy(states[(layer_index + 1) * sequence_width ..][0..sequence_width], hidden);
        }
    }

    ncm_norm.rms_norm_rows_with_bias(hidden, weights.final_norm_weight, weights.final_norm_bias, output, sequence_length, config.hidden_size, config.rms_norm_epsilon);
    if (hidden_states) |states| @memcpy(states[weights.layers.len * sequence_width ..][0..sequence_width], output);
}

fn split_attention_weights(
    config: config_t,
    layer: layer_weights_t,
    fused_width: usize,
    query_width: usize,
    key_value_width: usize,
) kernels.attention_weights_t {
    const hidden_size = config.hidden_size;
    std.debug.assert(layer.fused_qkv_weight.len == fused_width * hidden_size);
    if (layer.fused_qkv_bias) |bias| std.debug.assert(bias.len == fused_width);
    const query_size = query_width * hidden_size;
    const key_size = key_value_width * hidden_size;
    return .{
        .query_weight = layer.fused_qkv_weight[0..query_size],
        .query_bias = if (layer.fused_qkv_bias) |bias| bias[0..query_width] else null,
        .key_weight = layer.fused_qkv_weight[query_size..][0..key_size],
        .key_bias = if (layer.fused_qkv_bias) |bias| bias[query_width..][0..key_value_width] else null,
        .value_weight = layer.fused_qkv_weight[query_size + key_size ..][0..key_size],
        .value_bias = if (layer.fused_qkv_bias) |bias| bias[query_width + key_value_width ..][0..key_value_width] else null,
        .query_norm_weight = layer.query_norm_weight,
        .key_norm_weight = layer.key_norm_weight,
        .output_weight = layer.output_weight,
        .output_bias = layer.output_bias,
    };
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

pub fn apply_post_upsample_conv(
    input: []f32,
    convolution: post_upsample_conv_t,
    row_count: usize,
    width: usize,
    allocator: Allocator,
) !void {
    const input_size = try std.math.mul(usize, row_count, width);
    const kernel_width = try std.math.mul(usize, try std.math.mul(usize, width, width), 3);
    if (input.len != input_size or convolution.weight.len != kernel_width or convolution.bias.len != width) {
        return error.InvalidNcmTensorSize;
    }
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);
    for (0..row_count) |position| {
        for (0..width) |out_channel| {
            var value = convolution.bias[out_channel];
            for (0..width) |in_channel| {
                for (0..3) |kernel_index| {
                    const input_position = @as(isize, @intCast(position)) + @as(isize, @intCast(kernel_index)) - 1;
                    if (input_position < 0 or input_position >= row_count) continue;
                    const input_offset = @as(usize, @intCast(input_position)) * width + in_channel;
                    const weight_offset = out_channel * width * 3 + in_channel * 3 + kernel_index;
                    value += convolution.weight[weight_offset] * input[input_offset];
                }
            }
            output[position * width + out_channel] = input[position * width + out_channel] + value;
        }
    }
    @memcpy(input, output);
}

fn add_rows(destination: []f32, source: []const f32) void {
    for (destination, source) |*value, residual| value.* += residual;
}

test "NCM forward uses EOS prefix, level embeddings and learned level-0 queries" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const levels = [_]usize{ 1, 2 };
    const config: config_t = .{
        .hidden_size = 4,
        .vq_width = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 2,
        .intermediate_size = 3,
        .rms_norm_epsilon = 1e-5,
        .rope_type = .Default,
        .rope_theta = 10_000,
        .level_lengths = &levels,
        .prefix_length = 1,
        .query_level0 = true,
    };
    var prefix_table = [_]f32{ 0.1, -0.2, 0.3, 0.2, -0.4, 0.5 };
    var input_projection = [_]f32{ 0.2, -0.1, -0.3, 0.4, 0.1, 0.2, -0.2, 0.3 };
    const input_projection_bias = [_]f32{ 0.01, -0.02, 0.03, -0.04 };
    const level_embedding = [_]f32{ 0.01, 0.02, -0.02, 0.03, -0.03, 0.01, 0.02, -0.01 };
    const l0_queries = [_]f32{ 0.4, -0.3 };
    const final_norm_weight = [_]f32{ 1, 1, 1, 1 };
    var fused_qkv_weight: [32]f32 = undefined;
    for (&fused_qkv_weight, 0..) |*value, index| value.* = (@as(f32, @floatFromInt(index % 9)) - 4) * 0.025;
    const output_weight = [_]f32{
        0.1,  0.2,  -0.1, 0.05,
        -0.2, 0.1,  0.15, -0.05,
        0.05, -0.1, 0.2,  0.1,
        -0.1, 0.05, 0.1,  0.2,
    };
    const query_norm_weight = [_]f32{ 1, 1 };
    const key_norm_weight = [_]f32{ 1, 1 };
    const ln_1_weight = [_]f32{ 1, 1, 1, 1 };
    const ln_2_weight = [_]f32{ 1, 1, 1, 1 };
    const gate_weight = [_]f32{ 0.1, -0.1, 0.2, 0.05, -0.05, 0.1, -0.2, 0.15, 0.05, 0.2, 0.1, -0.1 };
    const up_weight = [_]f32{ 0.1, 0.05, -0.1, 0.15, -0.2, 0.1, 0.1, 0.2, -0.05, 0.1, -0.1, 0.05 };
    const down_weight = [_]f32{ 0.1, -0.1, 0.05, 0.2, 0.1, -0.05, -0.2, 0.15, 0.1, -0.1, 0.05, 0.1 };
    const one_layer: layer_weights_t = .{
        .fused_qkv_weight = &fused_qkv_weight,
        .output_weight = &output_weight,
        .query_norm_weight = &query_norm_weight,
        .key_norm_weight = &key_norm_weight,
        .ln_1_weight = &ln_1_weight,
        .gate_weight = &gate_weight,
        .up_weight = &up_weight,
        .down_weight = &down_weight,
        .ln_2_weight = &ln_2_weight,
    };
    const layer = [_]layer_weights_t{ one_layer, one_layer };
    const weights: weights_t = .{
        .prefix_token_embedding = &prefix_table,
        .prefix_vocab_size = 3,
        .input_projection = &input_projection,
        .input_projection_bias = &input_projection_bias,
        .level_embedding = &level_embedding,
        .l0_queries = &l0_queries,
        .final_norm_weight = &final_norm_weight,
        .layers = &layer,
    };
    const prefix_ids = [_]usize{2};
    const prefix_pad_mask = [_]bool{false};
    const target_inputs = [_]f32{ 99, -99, 0.1, 0.2, -0.3, 0.4 };
    const target_inputs_changed_l0 = [_]f32{ -33, 44, 0.1, 0.2, -0.3, 0.4 };
    var output: [16]f32 = undefined;
    var output_changed_l0: [16]f32 = undefined;

    try forward(allocator, config, weights, &prefix_ids, &prefix_pad_mask, &target_inputs, &output);
    try forward(allocator, config, weights, &prefix_ids, &prefix_pad_mask, &target_inputs_changed_l0, &output_changed_l0);
    try std.testing.expectEqualSlices(f32, &output, &output_changed_l0);

    var changed_l0_queries = l0_queries;
    changed_l0_queries[0] += 0.5;
    var weights_changed_l0 = weights;
    weights_changed_l0.l0_queries = &changed_l0_queries;
    var output_changed_query: [16]f32 = undefined;
    try forward(allocator, config, weights_changed_l0, &prefix_ids, &prefix_pad_mask, &target_inputs, &output_changed_query);
    try std.testing.expect(!std.mem.eql(f32, &output, &output_changed_query));

    var changed_level_embedding = level_embedding;
    changed_level_embedding[4] += 0.5;
    var weights_changed_level = weights;
    weights_changed_level.level_embedding = &changed_level_embedding;
    var output_changed_level: [16]f32 = undefined;
    try forward(allocator, config, weights_changed_level, &prefix_ids, &prefix_pad_mask, &target_inputs, &output_changed_level);
    try std.testing.expect(!std.mem.eql(f32, &output, &output_changed_level));

    prefix_table[4] += 1;
    var output_changed_prefix: [16]f32 = undefined;
    try forward(allocator, config, weights, &prefix_ids, &prefix_pad_mask, &target_inputs, &output_changed_prefix);
    try std.testing.expect(!std.mem.eql(f32, &output, &output_changed_prefix));

    const padded_prefix = [_]bool{true};
    try forward(allocator, config, weights, &prefix_ids, &padded_prefix, &target_inputs, &output);
    for (output) |value| try std.testing.expect(std.math.isFinite(value));
}
