//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! ncmorph command-line tool. `inspect` checks a local checkpoint directory against its
//! config.json. It reads the safetensors headers only, never the weight data.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("safetensors");
const hf_config = @import("hf_config");

const UsageText =
    \\usage: ncmorph inspect <checkpoint-dir>
    \\
    \\Checks config.json and the safetensors headers in a local directory. No weight data is read.
    \\Exit status: 0 when the checkpoint matches its configuration, 1 when it does not,
    \\2 on usage or I/O errors.
    \\
;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    var args = init.minimal.args.iterate();
    _ = args.next();
    const command = args.next() orelse {
        std.debug.print("{s}", .{UsageText});
        std.process.exit(2);
    };
    if (!std.mem.eql(u8, command, "inspect")) {
        std.debug.print("unknown command '{s}'\n{s}", .{ command, UsageText });
        std.process.exit(2);
    }
    const dir_path = args.next() orelse {
        std.debug.print("inspect needs a checkpoint directory\n{s}", .{UsageText});
        std.process.exit(2);
    };

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const out = &stdout_writer.interface;
    const status = inspect(io, arena, out, dir_path) catch |err| {
        std.debug.print("ncmorph: {s}: {s}\n", .{ dir_path, @errorName(err) });
        std.process.exit(2);
    };
    try out.flush();
    std.process.exit(status);
}

/// Inspects one checkpoint directory and prints the result. Returns the exit status.
fn inspect(io: std.Io, arena: Allocator, out: *std.Io.Writer, dir_path: []const u8) !u8 {
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);

    const config_bytes = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 20));
    const report = try hf_config.config.parse_config(arena, config_bytes);

    var problems: std.ArrayList(hf_config.config.problem_t) = .empty;
    var notes: std.ArrayList(hf_config.config.problem_t) = .empty;
    try problems.appendSlice(arena, report.problems);

    const index_bytes = dir.readFileAlloc(io, "model.safetensors.index.json", arena, .limited(64 << 20)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    var index: ?safetensors.index_t = null;
    var shard_names: std.ArrayList([]const u8) = .empty;
    if (index_bytes) |bytes| {
        if (safetensors.parse_index(arena, bytes)) |parsed| {
            index = parsed;
            for (parsed.weight_map) |entry| {
                if (!contains_name(shard_names.items, entry.shard)) try shard_names.append(arena, entry.shard);
            }
        } else |err| {
            try problems.append(arena, .{ .feature = "model.safetensors.index.json", .message = @errorName(err) });
        }
    } else {
        try shard_names.append(arena, "model.safetensors");
    }

    var shards: std.ArrayList(hf_config.manifest.shard_t) = .empty;
    var present: std.ArrayList(safetensors.tensor_info_t) = .empty;
    for (shard_names.items) |name| {
        const header = read_header(io, arena, dir, name) catch |err| {
            try problems.append(arena, .{
                .feature = name,
                .message = try std.fmt.allocPrint(arena, "the header could not be read: {s}", .{@errorName(err)}),
            });
            continue;
        };
        try shards.append(arena, .{ .file = name, .tensors = header.tensors });
        try present.appendSlice(arena, header.tensors);
    }

    if (report.spec) |spec| {
        const result = try hf_config.manifest.check(arena, spec, present.items);
        try problems.appendSlice(arena, result.problems);
        try notes.appendSlice(arena, result.notes);
        if (index) |checked_index| {
            try problems.appendSlice(arena, try hf_config.manifest.check_index(arena, checked_index, shards.items));
        }
    }

    try out.print("ncmorph inspect {s}\n", .{dir_path});
    if (report.spec) |spec| try print_spec(out, spec);
    try out.print("shards: {d}, tensors: {d}\n", .{ shard_names.items.len, present.items.len });
    for (notes.items) |note| try out.print("note {s}: {s}\n", .{ note.feature, note.message });
    for (problems.items) |problem| try out.print("error {s}: {s}\n", .{ problem.feature, problem.message });
    if (problems.items.len == 0) {
        try out.print("result: the checkpoint matches its configuration\n", .{});
        return 0;
    }
    try out.print("result: {d} problem(s); the checkpoint does not match its configuration\n", .{problems.items.len});
    return 1;
}

fn print_spec(out: *std.Io.Writer, spec: hf_config.config.spec_t) !void {
    try out.print("family: {s}\n", .{@tagName(spec.family)});
    try out.print("layers {d}, hidden {d}, intermediate {d}, heads {d}, kv heads {d}, head_dim {d}, vocab {d}\n", .{
        spec.num_hidden_layers,
        spec.hidden_size,
        spec.intermediate_size,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        spec.vocab_size,
    });
    const rope_name: []const u8 = switch (spec.rope_scaling) {
        .Default => "default",
        .Llama3 => "llama3",
    };
    try out.print("rope_theta {d}, rope_scaling {s}\n", .{ spec.rope_theta, rope_name });
    try out.print("qkv_bias {}, o_bias {}, mlp_bias {}, qk_norm {}, tie_word_embeddings {}, sliding_window {?d}\n", .{
        spec.qkv_bias,
        spec.o_bias,
        spec.mlp_bias,
        spec.qk_norm,
        spec.tie_word_embeddings,
        spec.sliding_window,
    });
}

/// Reads the header of one safetensors file. Only the header bytes are read.
fn read_header(io: std.Io, arena: Allocator, dir: std.Io.Dir, name: []const u8) !safetensors.header_t {
    var file = try dir.openFile(io, name, .{});
    defer file.close(io);
    const size = try file.length(io);
    var prefix: [safetensors.PrefixLen]u8 = undefined;
    try read_exact(io, file, &prefix, 0);
    const header_len = try safetensors.header_length(&prefix);
    const bytes = try arena.alloc(u8, safetensors.PrefixLen + header_len);
    try read_exact(io, file, bytes, 0);
    return safetensors.parse_header(arena, bytes, size);
}

fn read_exact(io: std.Io, file: std.Io.File, buffer: []u8, offset: u64) !void {
    const count = try file.readPositionalAll(io, buffer, offset);
    if (count != buffer.len) return error.UnexpectedEndOfFile;
}

fn contains_name(names: []const []const u8, name: []const u8) bool {
    for (names) |existing| {
        if (std.mem.eql(u8, existing, name)) return true;
    }
    return false;
}
