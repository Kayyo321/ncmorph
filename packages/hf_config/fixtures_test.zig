//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Tests against the stored configs and safetensors headers of real checkpoints. The fixtures
//! hold metadata only: config.json, the exact header bytes of each shard, the index and the file
//! sizes. No weight data is stored or read. testdata/<name>/provenance.json records the revision.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("safetensors");
const config = @import("config.zig");
const manifest = @import("manifest.zig");

const shard_fixture_t = struct {
    file: []const u8,
    /// The 8-byte length prefix followed by the JSON header, exactly as stored in the file.
    header: []const u8,
};

const fixture_t = struct {
    name: []const u8,
    family: config.Family,
    config_json: []const u8,
    index_json: ?[]const u8,
    sizes_json: []const u8,
    shards: []const shard_fixture_t,
    /// Notes the manifest is expected to produce, such as a stored lm_head.weight with tied embeddings.
    expected_notes: usize,
};

const Fixtures = [_]fixture_t{
    .{
        .name = "smollm2-135m",
        .family = .Llama,
        .config_json = @embedFile("testdata/smollm2-135m/config.json"),
        .index_json = null,
        .sizes_json = @embedFile("testdata/smollm2-135m/sizes.json"),
        .shards = &.{
            .{ .file = "model.safetensors", .header = @embedFile("testdata/smollm2-135m/model.safetensors.header") },
        },
        .expected_notes = 0,
    },
    .{
        .name = "qwen2.5-0.5b",
        .family = .Qwen2,
        .config_json = @embedFile("testdata/qwen2.5-0.5b/config.json"),
        .index_json = null,
        .sizes_json = @embedFile("testdata/qwen2.5-0.5b/sizes.json"),
        .shards = &.{
            .{ .file = "model.safetensors", .header = @embedFile("testdata/qwen2.5-0.5b/model.safetensors.header") },
        },
        .expected_notes = 0,
    },
    .{
        .name = "qwen3-0.6b",
        .family = .Qwen3,
        .config_json = @embedFile("testdata/qwen3-0.6b/config.json"),
        .index_json = null,
        .sizes_json = @embedFile("testdata/qwen3-0.6b/sizes.json"),
        .shards = &.{
            .{ .file = "model.safetensors", .header = @embedFile("testdata/qwen3-0.6b/model.safetensors.header") },
        },
        .expected_notes = 1,
    },
    .{
        .name = "llama-3.2-1b",
        .family = .Llama,
        .config_json = @embedFile("testdata/llama-3.2-1b/config.json"),
        .index_json = null,
        .sizes_json = @embedFile("testdata/llama-3.2-1b/sizes.json"),
        .shards = &.{
            .{ .file = "model.safetensors", .header = @embedFile("testdata/llama-3.2-1b/model.safetensors.header") },
        },
        .expected_notes = 0,
    },
    .{
        .name = "mistral-7b-v0.3",
        .family = .Mistral,
        .config_json = @embedFile("testdata/mistral-7b-v0.3/config.json"),
        .index_json = @embedFile("testdata/mistral-7b-v0.3/model.safetensors.index.json"),
        .sizes_json = @embedFile("testdata/mistral-7b-v0.3/sizes.json"),
        .shards = &.{
            .{ .file = "model-00001-of-00003.safetensors", .header = @embedFile("testdata/mistral-7b-v0.3/model-00001-of-00003.safetensors.header") },
            .{ .file = "model-00002-of-00003.safetensors", .header = @embedFile("testdata/mistral-7b-v0.3/model-00002-of-00003.safetensors.header") },
            .{ .file = "model-00003-of-00003.safetensors", .header = @embedFile("testdata/mistral-7b-v0.3/model-00003-of-00003.safetensors.header") },
        },
        .expected_notes = 0,
    },
    .{
        .name = "mistral-7b-v0.1",
        .family = .Mistral,
        .config_json = @embedFile("testdata/mistral-7b-v0.1/config.json"),
        .index_json = @embedFile("testdata/mistral-7b-v0.1/model.safetensors.index.json"),
        .sizes_json = @embedFile("testdata/mistral-7b-v0.1/sizes.json"),
        .shards = &.{
            .{ .file = "model-00001-of-00002.safetensors", .header = @embedFile("testdata/mistral-7b-v0.1/model-00001-of-00002.safetensors.header") },
            .{ .file = "model-00002-of-00002.safetensors", .header = @embedFile("testdata/mistral-7b-v0.1/model-00002-of-00002.safetensors.header") },
        },
        .expected_notes = 0,
    },
};

fn size_of(arena: Allocator, sizes_json: []const u8, file: []const u8) !u64 {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, sizes_json, .{});
    const object = switch (root) {
        .object => |object| object,
        else => return error.InvalidSizes,
    };
    const value = object.get(file) orelse return error.MissingSize;
    return switch (value) {
        .integer => |number| @as(u64, @intCast(number)),
        else => error.InvalidSizes,
    };
}

fn report_problems(name: []const u8, problems: []const config.problem_t) void {
    for (problems) |problem| {
        std.debug.print("{s}: {s}: {s}\n", .{ name, problem.feature, problem.message });
    }
}

test "real checkpoints parse and match their manifests" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    for (Fixtures) |fixture| {
        const report = try config.parse_config(allocator, fixture.config_json);
        report_problems(fixture.name, report.problems);
        try std.testing.expectEqual(@as(usize, 0), report.problems.len);
        const spec = report.spec.?;
        try std.testing.expectEqual(fixture.family, spec.family);

        var present: std.ArrayList(safetensors.tensor_info_t) = .empty;
        var shards: std.ArrayList(manifest.shard_t) = .empty;
        for (fixture.shards) |shard| {
            const size = try size_of(allocator, fixture.sizes_json, shard.file);
            const header = try safetensors.parse_header(allocator, shard.header, size);
            try shards.append(allocator, .{ .file = shard.file, .tensors = header.tensors });
            try present.appendSlice(allocator, header.tensors);
        }

        const result = try manifest.check(allocator, spec, present.items);
        report_problems(fixture.name, result.problems);
        try std.testing.expectEqual(@as(usize, 0), result.problems.len);
        try std.testing.expectEqual(fixture.expected_notes, result.notes.len);

        if (fixture.index_json) |index_bytes| {
            const index = try safetensors.parse_index(allocator, index_bytes);
            const index_problems = try manifest.check_index(allocator, index, shards.items);
            report_problems(fixture.name, index_problems);
            try std.testing.expectEqual(@as(usize, 0), index_problems.len);
        }
    }
}

test "fixture tensor counts match the expected layout" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    // Qwen2.5-0.5B: 24 layers of 12 tensors, plus the embedding and the final norm.
    const qwen2 = try config.parse_config(allocator, Fixtures[1].config_json);
    var qwen2_count: usize = 0;
    for (try manifest.templates(allocator, qwen2.spec.?)) |template| {
        if (!template.optional and !template.per_layer) qwen2_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), qwen2_count);
    // SmolLM2 has no bias, so each layer holds 9 tensors.
    const smollm = try config.parse_config(allocator, Fixtures[0].config_json);
    var smollm_per_layer: usize = 0;
    for (try manifest.templates(allocator, smollm.spec.?)) |template| {
        if (template.per_layer and !template.optional) smollm_per_layer += 1;
    }
    try std.testing.expectEqual(@as(usize, 9), smollm_per_layer);
}
