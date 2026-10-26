//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Reader for the safetensors header and the sharded index. The checks follow the reference
//! implementation, huggingface/safetensors (safetensors/src/tensor.rs): the header is at most
//! 100 MB, tensors sorted by data offset are contiguous from zero, each tensor's byte size
//! matches its shape and dtype, and the file is exactly the 8-byte prefix, the header and the data.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Largest accepted JSON header in bytes, as in the reference implementation.
pub const MaxHeaderSize: u64 = 100_000_000;
/// The file starts with the header length as a little-endian u64.
pub const PrefixLen: usize = 8;

pub const parse_error_t = error{
    HeaderTooSmall,
    HeaderTooLarge,
    InvalidHeaderLength,
    InvalidUtf8,
    InvalidHeader,
    UnknownDtype,
    InvalidOffsets,
    ShapeOverflow,
    MisalignedBits,
    SizeMismatch,
    FileSizeMismatch,
    InvalidIndex,
    UnsafeShardName,
};

/// Element types, named by their safetensors strings in `DtypeNames`.
pub const Dtype = enum {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F4,
    F6E2m3,
    F6E3m2,
    F8E5m2,
    F8E4m3,
    F8E8m0,
    F8E4m3fnuz,
    F8E5m2fnuz,
    F16,
    Bf16,
    F32,
    F64,
    C64,

    /// Size of one element in bits, as in the reference implementation.
    pub fn bit_size(self: Dtype) u64 {
        return switch (self) {
            .F4 => 4,
            .F6E2m3, .F6E3m2 => 6,
            .Bool, .U8, .I8, .F8E5m2, .F8E4m3, .F8E8m0, .F8E4m3fnuz, .F8E5m2fnuz => 8,
            .U16, .I16, .F16, .Bf16 => 16,
            .U32, .I32, .F32 => 32,
            .U64, .I64, .F64, .C64 => 64,
        };
    }
};

const dtype_name_t = struct {
    name: []const u8,
    dtype: Dtype,
};

const DtypeNames = [_]dtype_name_t{
    .{ .name = "BOOL", .dtype = .Bool },
    .{ .name = "U8", .dtype = .U8 },
    .{ .name = "I8", .dtype = .I8 },
    .{ .name = "U16", .dtype = .U16 },
    .{ .name = "I16", .dtype = .I16 },
    .{ .name = "U32", .dtype = .U32 },
    .{ .name = "I32", .dtype = .I32 },
    .{ .name = "U64", .dtype = .U64 },
    .{ .name = "I64", .dtype = .I64 },
    .{ .name = "F4", .dtype = .F4 },
    .{ .name = "F6_E2M3", .dtype = .F6E2m3 },
    .{ .name = "F6_E3M2", .dtype = .F6E3m2 },
    .{ .name = "F8_E5M2", .dtype = .F8E5m2 },
    .{ .name = "F8_E4M3", .dtype = .F8E4m3 },
    .{ .name = "F8_E8M0", .dtype = .F8E8m0 },
    .{ .name = "F8_E4M3FNUZ", .dtype = .F8E4m3fnuz },
    .{ .name = "F8_E5M2FNUZ", .dtype = .F8E5m2fnuz },
    .{ .name = "F16", .dtype = .F16 },
    .{ .name = "BF16", .dtype = .Bf16 },
    .{ .name = "F32", .dtype = .F32 },
    .{ .name = "F64", .dtype = .F64 },
    .{ .name = "C64", .dtype = .C64 },
};

/// Returns the dtype for a safetensors name such as "BF16", or null if it is unknown.
pub fn dtype_from_name(name: []const u8) ?Dtype {
    for (DtypeNames) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry.dtype;
    }
    return null;
}

/// Returns the safetensors name of `dtype`, such as "BF16".
pub fn dtype_name(dtype: Dtype) []const u8 {
    for (DtypeNames) |entry| {
        if (entry.dtype == dtype) return entry.name;
    }
    unreachable;
}

pub const tensor_info_t = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const u64,
    /// Byte range of the tensor data, relative to the start of the data section.
    data_begin: u64,
    data_end: u64,
};

pub const header_t = struct {
    /// Length of the JSON header, excluding the 8-byte prefix.
    header_len: u64,
    /// Length of the data section, which ends where the last tensor ends.
    data_len: u64,
    /// Tensors in data-offset order, as in the reference implementation.
    tensors: []const tensor_info_t,
};

pub const index_entry_t = struct {
    tensor: []const u8,
    shard: []const u8,
};

pub const index_t = struct {
    /// Tensor name to shard file name, from "weight_map".
    weight_map: []const index_entry_t,
    /// "metadata.total_size" when present.
    total_size: ?u64,
};

/// Reads the header length from the first 8 bytes of a file.
pub fn header_length(prefix: []const u8) parse_error_t!u64 {
    if (prefix.len < PrefixLen) return error.HeaderTooSmall;
    const length = std.mem.readInt(u64, prefix[0..PrefixLen], .little);
    if (length > MaxHeaderSize) return error.HeaderTooLarge;
    return length;
}

/// Parses and validates a safetensors header. `bytes` starts with the 8-byte prefix and holds
/// at least the whole JSON header. `file_size` is the size of the whole file. Allocations live
/// as long as `arena`.
pub fn parse_header(arena: Allocator, bytes: []const u8, file_size: u64) !header_t {
    const header_len = try header_length(bytes);
    const json_end = PrefixLen + header_len;
    if (bytes.len < json_end) return error.InvalidHeaderLength;
    const json_bytes = bytes[PrefixLen..json_end];
    if (!std.unicode.utf8ValidateSlice(json_bytes)) return error.InvalidUtf8;

    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, json_bytes, .{});
    const fields = switch (root) {
        .object => |object| object,
        else => return error.InvalidHeader,
    };

    var tensors: std.ArrayList(tensor_info_t) = .empty;
    var it = fields.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) {
            try check_metadata(entry.value_ptr.*);
            continue;
        }
        try tensors.append(arena, try parse_tensor(arena, name, entry.value_ptr.*));
    }
    std.mem.sort(tensor_info_t, tensors.items, {}, less_than_offsets);

    var expected_begin: u64 = 0;
    for (tensors.items) |tensor| {
        if (tensor.data_begin != expected_begin or tensor.data_end < tensor.data_begin) {
            return error.InvalidOffsets;
        }
        expected_begin = tensor.data_end;
        const elements = try element_count(tensor.shape);
        const bits = std.math.mul(u64, elements, tensor.dtype.bit_size()) catch return error.ShapeOverflow;
        if (bits % 8 != 0) return error.MisalignedBits;
        if (tensor.data_end - tensor.data_begin != bits / 8) return error.SizeMismatch;
    }
    if (PrefixLen + header_len + expected_begin != file_size) return error.FileSizeMismatch;

    return .{
        .header_len = header_len,
        .data_len = expected_begin,
        .tensors = tensors.items,
    };
}

/// Parses a sharded index file (`model.safetensors.index.json`). Shard names must be plain file
/// names, so that they cannot point outside the checkpoint directory.
pub fn parse_index(arena: Allocator, json_bytes: []const u8) !index_t {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, json_bytes, .{});
    const fields = switch (root) {
        .object => |object| object,
        else => return error.InvalidIndex,
    };
    const weight_map = switch (fields.get("weight_map") orelse return error.InvalidIndex) {
        .object => |object| object,
        else => return error.InvalidIndex,
    };

    var entries: std.ArrayList(index_entry_t) = .empty;
    var it = weight_map.iterator();
    while (it.next()) |entry| {
        const shard = switch (entry.value_ptr.*) {
            .string => |text| text,
            else => return error.InvalidIndex,
        };
        if (!is_plain_file_name(shard)) return error.UnsafeShardName;
        try entries.append(arena, .{ .tensor = entry.key_ptr.*, .shard = shard });
    }

    var total_size: ?u64 = null;
    if (fields.get("metadata")) |metadata| switch (metadata) {
        .object => |meta| if (meta.get("total_size")) |size| {
            total_size = try non_negative(size);
        },
        else => return error.InvalidIndex,
    };
    return .{ .weight_map = entries.items, .total_size = total_size };
}

fn parse_tensor(arena: Allocator, name: []const u8, value: std.json.Value) !tensor_info_t {
    const fields = switch (value) {
        .object => |object| object,
        else => return error.InvalidHeader,
    };
    const dtype_text = switch (fields.get("dtype") orelse return error.InvalidHeader) {
        .string => |text| text,
        else => return error.InvalidHeader,
    };
    const dtype = dtype_from_name(dtype_text) orelse return error.UnknownDtype;
    const shape_items = switch (fields.get("shape") orelse return error.InvalidHeader) {
        .array => |array| array.items,
        else => return error.InvalidHeader,
    };
    const shape = try arena.alloc(u64, shape_items.len);
    for (shape_items, shape) |item, *dim| {
        dim.* = try non_negative(item);
    }
    const offset_items = switch (fields.get("data_offsets") orelse return error.InvalidHeader) {
        .array => |array| array.items,
        else => return error.InvalidHeader,
    };
    if (offset_items.len != 2) return error.InvalidHeader;
    return .{
        .name = name,
        .dtype = dtype,
        .shape = shape,
        .data_begin = try non_negative(offset_items[0]),
        .data_end = try non_negative(offset_items[1]),
    };
}

fn check_metadata(value: std.json.Value) !void {
    const fields = switch (value) {
        .object => |object| object,
        else => return error.InvalidHeader,
    };
    var it = fields.iterator();
    while (it.next()) |entry| {
        switch (entry.value_ptr.*) {
            .string => {},
            else => return error.InvalidHeader,
        }
    }
}

fn non_negative(value: std.json.Value) !u64 {
    return switch (value) {
        .integer => |number| std.math.cast(u64, number) orelse error.InvalidHeader,
        else => error.InvalidHeader,
    };
}

fn element_count(shape: []const u64) parse_error_t!u64 {
    var count: u64 = 1;
    for (shape) |dim| {
        count = std.math.mul(u64, count, dim) catch return error.ShapeOverflow;
    }
    return count;
}

fn less_than_offsets(_: void, a: tensor_info_t, b: tensor_info_t) bool {
    if (a.data_begin != b.data_begin) return a.data_begin < b.data_begin;
    return a.data_end < b.data_end;
}

fn is_plain_file_name(name: []const u8) bool {
    if (name.len == 0 or std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return false;
    for (name) |byte| {
        if (byte == '/' or byte == '\\' or byte == 0) return false;
    }
    return true;
}

/// Builds a whole file in memory: the length prefix, the JSON header and `data_len` zero bytes.
fn make_file(arena: Allocator, json: []const u8, data_len: usize) ![]u8 {
    const bytes = try arena.alloc(u8, PrefixLen + json.len + data_len);
    std.mem.writeInt(u64, bytes[0..PrefixLen], json.len, .little);
    @memcpy(bytes[PrefixLen .. PrefixLen + json.len], json);
    @memset(bytes[PrefixLen + json.len ..], 0);
    return bytes;
}

test "accepts two contiguous tensors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
        \\ "b":{"dtype":"BF16","shape":[2,2],"data_offsets":[8,16]}}
    ;
    const bytes = try make_file(arena.allocator(), json, 16);
    const header = try parse_header(arena.allocator(), bytes, bytes.len);
    try std.testing.expectEqual(@as(usize, 2), header.tensors.len);
    try std.testing.expectEqual(@as(u64, 16), header.data_len);
    try std.testing.expectEqual(Dtype.Bf16, header.tensors[1].dtype);
}

test "sorts tensors by offset before checking contiguity" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},
        \\ "a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
    ;
    const bytes = try make_file(arena.allocator(), json, 8);
    const header = try parse_header(arena.allocator(), bytes, bytes.len);
    try std.testing.expectEqualStrings("a", header.tensors[0].name);
}

test "rejects a gap between tensors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
        \\ "b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}
    ;
    const bytes = try make_file(arena.allocator(), json, 12);
    try std.testing.expectError(error.InvalidOffsets, parse_header(arena.allocator(), bytes, bytes.len));
}

test "rejects a byte size that does not match the shape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}
    ;
    const bytes = try make_file(arena.allocator(), json, 8);
    try std.testing.expectError(error.SizeMismatch, parse_header(arena.allocator(), bytes, bytes.len));
}

test "rejects an unknown dtype" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"a":{"dtype":"F7","shape":[1],"data_offsets":[0,1]}}
    ;
    const bytes = try make_file(arena.allocator(), json, 1);
    try std.testing.expectError(error.UnknownDtype, parse_header(arena.allocator(), bytes, bytes.len));
}

test "rejects bit sizes that do not fill whole bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"a":{"dtype":"F4","shape":[3],"data_offsets":[0,2]}}
    ;
    const bytes = try make_file(arena.allocator(), json, 2);
    try std.testing.expectError(error.MisalignedBits, parse_header(arena.allocator(), bytes, bytes.len));
}

test "rejects a file that is not exactly header plus data" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
    ;
    const bytes = try make_file(arena.allocator(), json, 8);
    try std.testing.expectError(error.FileSizeMismatch, parse_header(arena.allocator(), bytes, bytes.len));
}

test "rejects a header longer than the maximum" {
    var prefix: [PrefixLen]u8 = undefined;
    std.mem.writeInt(u64, &prefix, MaxHeaderSize + 1, .little);
    try std.testing.expectError(error.HeaderTooLarge, header_length(&prefix));
}

test "accepts string metadata and rejects other metadata" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const good =
        \\{"__metadata__":{"format":"pt"},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}
    ;
    const good_bytes = try make_file(arena.allocator(), good, 1);
    _ = try parse_header(arena.allocator(), good_bytes, good_bytes.len);

    const bad =
        \\{"__metadata__":{"count":1},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}
    ;
    const bad_bytes = try make_file(arena.allocator(), bad, 1);
    try std.testing.expectError(error.InvalidHeader, parse_header(arena.allocator(), bad_bytes, bad_bytes.len));
}

test "dtype names round trip" {
    try std.testing.expectEqual(Dtype.Bf16, dtype_from_name("BF16").?);
    try std.testing.expectEqualStrings("F6_E2M3", dtype_name(.F6E2m3));
    try std.testing.expectEqual(@as(?Dtype, null), dtype_from_name("bf16"));
}

test "parses a sharded index and rejects unsafe shard names" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const good =
        \\{"metadata":{"total_size":16},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}
    ;
    const index = try parse_index(arena.allocator(), good);
    try std.testing.expectEqual(@as(usize, 2), index.weight_map.len);
    try std.testing.expectEqual(@as(?u64, 16), index.total_size);

    const unsafe =
        \\{"weight_map":{"a":"../escape.safetensors"}}
    ;
    try std.testing.expectError(error.UnsafeShardName, parse_index(arena.allocator(), unsafe));
}
