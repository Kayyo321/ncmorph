//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Tokenizer JSON loading, BPE encoding and decoding for the pinned model families.

const std = @import("std");
const pretokenizers = @import("pretokenizers");
const nfc = @import("nfc");

const Allocator = std.mem.Allocator;
const JsonValue = std.json.Value;
const Utf8Bytes = std.ArrayList(u8);
const TokenIds = std.ArrayList(u32);

pub const tokenizer_error_t = error{
    InvalidTokenizer,
    UnsupportedTokenizer,
    UnsupportedAddedTokenOptions,
    InvalidUtf8,
    TokenIdOutOfRange,
};

const Family = enum {
    qwen,
    llama3,
    smollm,
    llama2,
    mistral,
};

const merge_t = struct {
    rank: u32,
    result_id: u32,
};

const pair_t = struct {
    left: u32,
    right: u32,
};

const added_token_t = struct {
    content: []const u8,
    id: u32,
    normalized: bool,
    special: bool,
};

/// A parsed tokenizer.json. Its internal data uses an arena and remains valid until deinit.
pub const tokenizer_t = struct {
    arena: std.heap.ArenaAllocator,
    family: Family,
    vocab_by_id: []const []const u8,
    decoded_by_id: []const []const u8,
    special_by_id: []const bool,
    vocab_by_token: std.StringHashMapUnmanaged(u32),
    merges: std.AutoHashMapUnmanaged(pair_t, merge_t),
    added_tokens: []const added_token_t,
    unk_id: ?u32,
    byte_fallback: bool,
    fuse_unk: bool,
    ignore_merges: bool,
    bos_id: ?u32,

    /// Parses one of the pinned tokenizer.json pipelines without loading model weights.
    pub fn from_json(allocator: Allocator, bytes: []const u8) !tokenizer_t {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const arena_allocator = arena.allocator();
        const root = std.json.parseFromSliceLeaky(JsonValue, arena_allocator, bytes, .{}) catch return error.InvalidTokenizer;
        if (root != .object) return error.InvalidTokenizer;
        const model = object_field(root.object, "model") orelse return error.InvalidTokenizer;
        if (model != .object or !string_is(object_field(model.object, "type"), "BPE")) return error.UnsupportedTokenizer;

        const family = try detect_family(root.object, model.object);
        const dropout = object_field(model.object, "dropout") orelse return error.InvalidTokenizer;
        if (dropout != .null) return error.UnsupportedTokenizer;
        try require_empty_or_null(model.object, "continuing_subword_prefix");
        try require_empty_or_null(model.object, "end_of_word_suffix");

        const vocab_value = object_field(model.object, "vocab") orelse return error.InvalidTokenizer;
        if (vocab_value != .object or vocab_value.object.count() == 0) return error.InvalidTokenizer;
        var max_id: u32 = 0;
        var vocab_it = vocab_value.object.iterator();
        while (vocab_it.next()) |entry| {
            const token_id = json_u32(entry.value_ptr.*) orelse return error.InvalidTokenizer;
            max_id = @max(max_id, token_id);
        }

        const added_tokens = try parse_added_tokens(arena_allocator, root.object);
        for (added_tokens) |added| max_id = @max(max_id, added.id);
        const vocab_by_id = try arena_allocator.alloc([]const u8, @as(usize, max_id) + 1);
        const decoded_by_id = try arena_allocator.alloc([]const u8, @as(usize, max_id) + 1);
        const special_by_id = try arena_allocator.alloc(bool, @as(usize, max_id) + 1);
        @memset(vocab_by_id, "");
        @memset(decoded_by_id, "");
        @memset(special_by_id, false);
        var vocab_by_token: std.StringHashMapUnmanaged(u32) = .empty;
        vocab_it = vocab_value.object.iterator();
        while (vocab_it.next()) |entry| {
            const token_id = json_u32(entry.value_ptr.*).?;
            if (vocab_by_id[token_id].len != 0) return error.InvalidTokenizer;
            vocab_by_id[token_id] = entry.key_ptr.*;
            decoded_by_id[token_id] = entry.key_ptr.*;
            try vocab_by_token.put(arena_allocator, entry.key_ptr.*, token_id);
        }
        for (added_tokens) |added| {
            if (vocab_by_id[added.id].len != 0 and !std.mem.eql(u8, vocab_by_id[added.id], added.content)) return error.InvalidTokenizer;
            vocab_by_id[added.id] = added.content;
            // tokenizers 0.23.3 decodes added IDs through AddedVocabulary::simple_id_to_token,
            // which may return the normalizer's cached spelling. Its skip-special filter then
            // compares that spelling with the original special-token content (tokenizer/mod.rs
            // TokenizerImpl::decode; tokenizer/added_vocabulary.rs simple_id_to_token).
            const decoded_content = if (added.normalized) try normalize_added_token(arena_allocator, family, added.content) else added.content;
            decoded_by_id[added.id] = decoded_content;
            special_by_id[added.id] = added.special and std.mem.eql(u8, decoded_content, added.content);
            try vocab_by_token.put(arena_allocator, added.content, added.id);
        }
        for (vocab_by_id) |token| if (token.len == 0) return error.InvalidTokenizer;

        const merges_value = object_field(model.object, "merges") orelse return error.InvalidTokenizer;
        if (merges_value != .array) return error.InvalidTokenizer;
        var merges: std.AutoHashMapUnmanaged(pair_t, merge_t) = .empty;
        for (merges_value.array.items, 0..) |merge, rank| {
            const left, const right = try parse_merge(merge);
            const left_id = vocab_by_token.get(left) orelse return error.InvalidTokenizer;
            const right_id = vocab_by_token.get(right) orelse return error.InvalidTokenizer;
            const merged = try concatenate(arena_allocator, left, right);
            const result_id = vocab_by_token.get(merged) orelse return error.InvalidTokenizer;
            const merge_rank = std.math.cast(u32, rank) orelse return error.InvalidTokenizer;
            try merges.put(arena_allocator, .{ .left = left_id, .right = right_id }, .{ .rank = merge_rank, .result_id = result_id });
        }

        const unk_id = if (object_field(model.object, "unk_token")) |value| switch (value) {
            .null => null,
            .string => |token| vocab_by_token.get(token) orelse return error.InvalidTokenizer,
            else => return error.InvalidTokenizer,
        } else null;
        const byte_fallback = json_bool_default(model.object, "byte_fallback", false) catch return error.InvalidTokenizer;
        const fuse_unk = json_bool_default(model.object, "fuse_unk", false) catch return error.InvalidTokenizer;
        const ignore_merges = json_bool_default(model.object, "ignore_merges", false) catch return error.InvalidTokenizer;
        const bos_id = try parse_bos_id(arena_allocator, root.object, vocab_by_token);

        return .{
            .arena = arena,
            .family = family,
            .vocab_by_id = vocab_by_id,
            .decoded_by_id = decoded_by_id,
            .special_by_id = special_by_id,
            .vocab_by_token = vocab_by_token,
            .merges = merges,
            .added_tokens = added_tokens,
            .unk_id = unk_id,
            .byte_fallback = byte_fallback,
            .fuse_unk = fuse_unk,
            .ignore_merges = ignore_merges,
            .bos_id = bos_id,
        };
    }

    pub fn deinit(self: *tokenizer_t) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Encodes UTF-8 text and applies the tokenizer's configured single-sequence post-processor.
    pub fn encode(self: *const tokenizer_t, allocator: Allocator, text: []const u8) ![]u32 {
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
        var ids: TokenIds = .empty;
        errdefer ids.deinit(allocator);

        if (self.family == .llama2) {
            try self.encode_llama2(allocator, text, &ids);
        } else {
            var cursor: usize = 0;
            while (cursor < text.len) {
                if (find_added_token(self.added_tokens, text[cursor..])) |match| {
                    try self.encode_plain(allocator, text[cursor..][0..match.start], cursor == 0, &ids);
                    try ids.append(allocator, match.id);
                    cursor += match.start + match.len;
                } else {
                    try self.encode_plain(allocator, text[cursor..], cursor == 0, &ids);
                    cursor = text.len;
                }
            }
            if (text.len == 0) try self.encode_plain(allocator, text, true, &ids);
        }
        if (self.bos_id) |bos_id| try ids.insert(allocator, 0, bos_id);
        return ids.toOwnedSlice(allocator);
    }

    /// Decodes token ids through the configured decoder chain.
    pub fn decode(self: *const tokenizer_t, allocator: Allocator, ids: []const u32, skip_special_tokens: bool) ![]u8 {
        if (skip_special_tokens) {
            var filtered: TokenIds = .empty;
            defer filtered.deinit(allocator);
            for (ids) |id| {
                if (id >= self.special_by_id.len) return error.TokenIdOutOfRange;
                if (!self.special_by_id[id]) try filtered.append(allocator, id);
            }
            return self.decode_all(allocator, filtered.items);
        }
        return self.decode_all(allocator, ids);
    }

    fn decode_all(self: *const tokenizer_t, allocator: Allocator, ids: []const u32) ![]u8 {
        return switch (self.family) {
            .qwen, .llama3, .smollm => self.decode_byte_level(allocator, ids),
            .llama2, .mistral => self.decode_sentencepiece(allocator, ids),
        };
    }

    fn encode_llama2(self: *const tokenizer_t, allocator: Allocator, text: []const u8, output: *TokenIds) !void {
        const normalized = try normalize_llama2(allocator, text);
        defer allocator.free(normalized);
        var cursor: usize = 0;
        while (cursor < normalized.len) {
            if (try find_normalized_added_token(allocator, self.added_tokens, .llama2, normalized[cursor..])) |match| {
                try self.encode_piece(allocator, normalized[cursor..][0..match.start], output);
                try output.append(allocator, match.id);
                cursor += match.start + match.len;
            } else {
                try self.encode_piece(allocator, normalized[cursor..], output);
                cursor = normalized.len;
            }
        }
    }

    fn encode_plain(self: *const tokenizer_t, allocator: Allocator, input: []const u8, is_initial_text: bool, output: *TokenIds) !void {
        if (input.len == 0) return;
        var normalized: []u8 = undefined;
        var owns_normalized = false;
        switch (self.family) {
            .qwen => {
                normalized = try nfc.normalize(allocator, input);
                owns_normalized = true;
            },
            .mistral => {
                normalized = try normalize_metaspace(allocator, input, is_initial_text);
                owns_normalized = true;
            },
            .llama2, .llama3, .smollm => normalized = @constCast(input),
        }
        defer if (owns_normalized) allocator.free(normalized);

        switch (self.family) {
            .qwen => try self.encode_byte_pieces(allocator, normalized, .qwen_split, output),
            .llama3 => try self.encode_byte_pieces(allocator, normalized, .llama3_split, output),
            .smollm => try self.encode_byte_pieces(allocator, normalized, .digits_individual, output),
            .llama2, .mistral => try self.encode_piece(allocator, normalized, output),
        }
    }

    fn encode_byte_pieces(self: *const tokenizer_t, allocator: Allocator, text: []const u8, matcher: pretokenizers.Matcher, output: *TokenIds) !void {
        const pieces = try pretokenizers.split(allocator, text, matcher);
        defer allocator.free(pieces);
        for (pieces) |piece| {
            if (self.family == .smollm) {
                const words = try pretokenizers.split(allocator, piece, .gpt2_byte_level);
                defer allocator.free(words);
                for (words) |word| try self.encode_mapped_piece(allocator, word, output);
            } else {
                try self.encode_mapped_piece(allocator, piece, output);
            }
        }
    }

    fn encode_mapped_piece(self: *const tokenizer_t, allocator: Allocator, piece: []const u8, output: *TokenIds) !void {
        const mapped = try map_bytes_to_unicode(allocator, piece);
        defer allocator.free(mapped);
        try self.encode_piece(allocator, mapped, output);
    }

    fn encode_piece(self: *const tokenizer_t, allocator: Allocator, piece: []const u8, output: *TokenIds) !void {
        if (piece.len == 0) return;
        if (self.ignore_merges) {
            if (self.vocab_by_token.get(piece)) |id| {
                try output.append(allocator, id);
                return;
            }
        }

        var word: TokenIds = .empty;
        defer word.deinit(allocator);
        var cursor: usize = 0;
        var previous_unknown = false;
        while (cursor < piece.len) {
            const width = std.unicode.utf8ByteSequenceLength(piece[cursor]) catch return error.InvalidUtf8;
            const end = cursor + width;
            if (end > piece.len) return error.InvalidUtf8;
            const scalar = piece[cursor..end];
            if (self.vocab_by_token.get(scalar)) |id| {
                try word.append(allocator, id);
                previous_unknown = false;
            } else if (self.byte_fallback and (try self.try_byte_fallback(allocator, scalar, &word))) {
                previous_unknown = false;
            } else if (self.unk_id) |unk_id| {
                if (!(self.fuse_unk and previous_unknown)) try word.append(allocator, unk_id);
                previous_unknown = true;
            }
            cursor = end;
        }

        while (true) {
            var best_rank: ?u32 = null;
            var best_index: usize = 0;
            var best_id: u32 = 0;
            if (word.items.len < 2) break;
            for (0..word.items.len - 1) |index| {
                if (self.merges.get(.{ .left = word.items[index], .right = word.items[index + 1] })) |merge| {
                    if (best_rank == null or merge.rank < best_rank.?) {
                        best_rank = merge.rank;
                        best_index = index;
                        best_id = merge.result_id;
                    }
                }
            }
            if (best_rank == null) break;
            word.items[best_index] = best_id;
            std.mem.copyForwards(u32, word.items[best_index + 1 .. word.items.len - 1], word.items[best_index + 2 .. word.items.len]);
            word.items.len -= 1;
        }
        try output.appendSlice(allocator, word.items);
    }

    fn try_byte_fallback(self: *const tokenizer_t, allocator: Allocator, scalar: []const u8, word: *TokenIds) Allocator.Error!bool {
        var byte_ids: TokenIds = .empty;
        defer byte_ids.deinit(allocator);
        for (scalar) |byte| {
            const token = byte_token_name(byte);
            const id = self.vocab_by_token.get(token[0..]) orelse return false;
            try byte_ids.append(allocator, id);
        }
        try word.appendSlice(allocator, byte_ids.items);
        return true;
    }

    fn decode_byte_level(self: *const tokenizer_t, allocator: Allocator, ids: []const u32) ![]u8 {
        var bytes: Utf8Bytes = .empty;
        defer bytes.deinit(allocator);
        for (ids) |id| {
            const token = self.token_by_id(id) orelse return error.TokenIdOutOfRange;
            var decoded: Utf8Bytes = .empty;
            defer decoded.deinit(allocator);
            var mapped_all = true;
            var cursor: usize = 0;
            while (cursor < token.len) {
                const width = std.unicode.utf8ByteSequenceLength(token[cursor]) catch {
                    mapped_all = false;
                    break;
                };
                const end = cursor + width;
                if (end > token.len) {
                    mapped_all = false;
                    break;
                }
                const cp = std.unicode.utf8Decode(token[cursor..end]) catch {
                    mapped_all = false;
                    break;
                };
                if (unicode_to_byte(cp)) |byte| {
                    try decoded.append(allocator, byte);
                } else {
                    mapped_all = false;
                    break;
                }
                cursor = end;
            }
            try bytes.appendSlice(allocator, if (mapped_all) decoded.items else token);
        }
        return lossy_utf8(allocator, bytes.items);
    }

    fn decode_sentencepiece(self: *const tokenizer_t, allocator: Allocator, ids: []const u32) ![]u8 {
        var joined: Utf8Bytes = .empty;
        defer joined.deinit(allocator);
        var index: usize = 0;
        while (index < ids.len) {
            const id = ids[index];
            const token = self.token_by_id(id) orelse return error.TokenIdOutOfRange;
            if (parse_byte_token(token)) |byte| {
                var run: Utf8Bytes = .empty;
                defer run.deinit(allocator);
                try run.append(allocator, byte);
                index += 1;
                while (index < ids.len) : (index += 1) {
                    const next = self.token_by_id(ids[index]) orelse return error.TokenIdOutOfRange;
                    if (parse_byte_token(next)) |next_byte| {
                        try run.append(allocator, next_byte);
                    } else break;
                }
                const byte_slice = run.items;
                if (std.unicode.utf8ValidateSlice(byte_slice)) {
                    try joined.appendSlice(allocator, byte_slice);
                } else {
                    for (run.items) |_| try joined.appendSlice(allocator, "�");
                }
                continue;
            } else {
                var cursor: usize = 0;
                while (cursor < token.len) {
                    if (std.mem.startsWith(u8, token[cursor..], "▁")) {
                        try joined.append(allocator, ' ');
                        cursor += "▁".len;
                    } else {
                        try joined.append(allocator, token[cursor]);
                        cursor += 1;
                    }
                }
            }
            index += 1;
        }
        const valid = joined.items;
        var start: usize = 0;
        const stop = valid.len;
        if (start < stop and valid[start] == ' ') start += 1;
        return allocator.dupe(u8, valid[start..stop]);
    }

    fn token_by_id(self: *const tokenizer_t, id: u32) ?[]const u8 {
        if (id >= self.decoded_by_id.len) return null;
        return self.decoded_by_id[id];
    }
};

fn object_field(object: std.json.ObjectMap, key: []const u8) ?JsonValue {
    return object.get(key);
}

fn string_is(value: ?JsonValue, expected: []const u8) bool {
    return if (value) |item| item == .string and std.mem.eql(u8, item.string, expected) else false;
}

fn json_u32(value: JsonValue) ?u32 {
    return switch (value) {
        .integer => |integer| if (integer >= 0) std.math.cast(u32, integer) else null,
        else => null,
    };
}

fn json_bool_default(object: std.json.ObjectMap, key: []const u8, default: bool) !bool {
    const value = object.get(key) orelse return default;
    return switch (value) {
        .bool => |boolean| boolean,
        else => error.InvalidTokenizer,
    };
}

fn require_empty_or_null(object: std.json.ObjectMap, key: []const u8) !void {
    const value = object.get(key) orelse return;
    switch (value) {
        .null => {},
        .string => |string| if (string.len != 0) return error.UnsupportedTokenizer,
        else => return error.InvalidTokenizer,
    }
}

fn detect_family(root: std.json.ObjectMap, model: std.json.ObjectMap) !Family {
    const normalizer = root.get("normalizer") orelse return error.InvalidTokenizer;
    if (object_type_is(normalizer, "NFC")) return .qwen;
    if (object_type_is(normalizer, "Sequence")) {
        const normalizers = normalizer.object.get("normalizers") orelse return error.InvalidTokenizer;
        if (normalizers != .array or normalizers.array.items.len != 2) return error.UnsupportedTokenizer;
        if (normalizers.array.items[0] != .object or normalizers.array.items[1] != .object) return error.InvalidTokenizer;
        if (object_type_is(normalizers.array.items[0], "Prepend") and object_type_is(normalizers.array.items[1], "Replace")) return .llama2;
        return error.UnsupportedTokenizer;
    }
    const pre = root.get("pre_tokenizer") orelse return error.InvalidTokenizer;
    if (object_type_is(pre, "Metaspace")) return .mistral;
    if (object_type_is(pre, "Sequence")) {
        const items = pre.object.get("pretokenizers") orelse return error.InvalidTokenizer;
        if (items != .array or items.array.items.len != 2) return error.UnsupportedTokenizer;
        if (items.array.items[0] != .object or items.array.items[1] != .object) return error.InvalidTokenizer;
        if (object_type_is(items.array.items[0], "Digits")) return .smollm;
        if (object_type_is(items.array.items[0], "Split")) {
            const pattern = items.array.items[0].object.get("pattern") orelse return error.InvalidTokenizer;
            if (pattern == .object and pattern.object.get("Regex") != null) {
                const regex_value = pattern.object.get("Regex").?;
                if (regex_value != .string) return error.InvalidTokenizer;
                const regex = regex_value.string;
                if (std.mem.indexOf(u8, regex, "\\p{N}{1,3}") != null) return .llama3;
                return .qwen;
            }
        }
    }
    _ = model;
    return error.UnsupportedTokenizer;
}

fn object_type_is(value: JsonValue, expected: []const u8) bool {
    return value == .object and string_is(value.object.get("type"), expected);
}

fn parse_added_tokens(allocator: Allocator, root: std.json.ObjectMap) ![]const added_token_t {
    const value = root.get("added_tokens") orelse return &.{};
    if (value != .array) return error.InvalidTokenizer;
    var tokens: std.ArrayList(added_token_t) = .empty;
    for (value.array.items) |item| {
        if (item != .object) return error.InvalidTokenizer;
        const content = item.object.get("content") orelse return error.InvalidTokenizer;
        const id = item.object.get("id") orelse return error.InvalidTokenizer;
        if (content != .string) return error.InvalidTokenizer;
        const token_id = json_u32(id) orelse return error.InvalidTokenizer;
        inline for (.{ "single_word", "lstrip", "rstrip" }) |key| {
            if (try json_bool_default(item.object, key, false)) return error.UnsupportedAddedTokenOptions;
        }
        const normalized = try json_bool_default(item.object, "normalized", true);
        const special = try json_bool_default(item.object, "special", false);
        try tokens.append(allocator, .{ .content = content.string, .id = token_id, .normalized = normalized, .special = special });
    }
    std.mem.sort(added_token_t, tokens.items, {}, struct {
        fn lessThan(_: void, left: added_token_t, right: added_token_t) bool {
            return left.content.len > right.content.len;
        }
    }.lessThan);
    return tokens.toOwnedSlice(allocator);
}

const added_match_t = struct { start: usize, len: usize, id: u32 };

fn find_added_token(tokens: []const added_token_t, input: []const u8) ?added_match_t {
    var best: ?added_match_t = null;
    for (tokens) |token| {
        if (std.mem.indexOf(u8, input, token.content)) |start| {
            if (best == null or start < best.?.start or (start == best.?.start and token.content.len > best.?.len)) {
                best = .{ .start = start, .len = token.content.len, .id = token.id };
            }
        }
    }
    return best;
}

fn find_normalized_added_token(allocator: Allocator, tokens: []const added_token_t, family: Family, input: []const u8) !?added_match_t {
    var best: ?added_match_t = null;
    for (tokens) |token| {
        if (!token.normalized) continue;
        const pattern = try normalize_added_token(allocator, family, token.content);
        defer allocator.free(pattern);
        if (std.mem.indexOf(u8, input, pattern)) |start| {
            if (best == null or start < best.?.start or (start == best.?.start and pattern.len > best.?.len)) {
                best = .{ .start = start, .len = pattern.len, .id = token.id };
            }
        }
    }
    return best;
}

fn parse_merge(value: JsonValue) !struct { []const u8, []const u8 } {
    switch (value) {
        .string => |line| {
            var parts = std.mem.splitScalar(u8, line, ' ');
            const left = parts.next() orelse return error.InvalidTokenizer;
            const right = parts.next() orelse return error.InvalidTokenizer;
            if (left.len == 0 or right.len == 0 or parts.next() != null) return error.InvalidTokenizer;
            return .{ left, right };
        },
        .array => |array| {
            if (array.items.len != 2 or array.items[0] != .string or array.items[1] != .string) return error.InvalidTokenizer;
            return .{ array.items[0].string, array.items[1].string };
        },
        else => return error.InvalidTokenizer,
    }
}

fn concatenate(allocator: Allocator, left: []const u8, right: []const u8) Allocator.Error![]const u8 {
    const joined = try allocator.alloc(u8, left.len + right.len);
    @memcpy(joined[0..left.len], left);
    @memcpy(joined[left.len..], right);
    return joined;
}

fn parse_bos_id(allocator: Allocator, root: std.json.ObjectMap, vocab: std.StringHashMapUnmanaged(u32)) !?u32 {
    _ = allocator;
    const processor = root.get("post_processor") orelse return null;
    return find_template_bos(processor, vocab);
}

fn find_template_bos(value: JsonValue, vocab: std.StringHashMapUnmanaged(u32)) !?u32 {
    if (value == .object) {
        if (string_is(value.object.get("type"), "TemplateProcessing")) {
            const specials = value.object.get("special_tokens") orelse return error.InvalidTokenizer;
            if (specials != .object) return error.InvalidTokenizer;
            const single = value.object.get("single") orelse return error.InvalidTokenizer;
            if (single != .array or single.array.items.len == 0) return error.InvalidTokenizer;
            const first = single.array.items[0];
            if (first != .object) return error.InvalidTokenizer;
            const special = first.object.get("SpecialToken") orelse return null;
            if (special != .object) return error.InvalidTokenizer;
            const id_string = special.object.get("id") orelse return error.InvalidTokenizer;
            if (id_string != .string) return error.InvalidTokenizer;
            const detail = specials.object.get(id_string.string) orelse return error.InvalidTokenizer;
            if (detail != .object) return error.InvalidTokenizer;
            const ids = detail.object.get("ids") orelse return error.InvalidTokenizer;
            if (ids != .array or ids.array.items.len != 1) return error.InvalidTokenizer;
            return json_u32(ids.array.items[0]) orelse error.InvalidTokenizer;
        }
        if (string_is(value.object.get("type"), "Sequence")) {
            const processors = value.object.get("processors") orelse return error.InvalidTokenizer;
            if (processors != .array) return error.InvalidTokenizer;
            for (processors.array.items) |processor| {
                if (try find_template_bos(processor, vocab)) |id| return id;
            }
        }
    }
    return null;
}

fn normalize_llama2(allocator: Allocator, input: []const u8) Allocator.Error![]u8 {
    var output: Utf8Bytes = .empty;
    errdefer output.deinit(allocator);
    if (input.len != 0) try output.appendSlice(allocator, "▁");
    for (input) |byte| {
        if (byte == ' ') {
            try output.appendSlice(allocator, "▁");
        } else {
            try output.append(allocator, byte);
        }
    }
    return output.toOwnedSlice(allocator);
}

fn normalize_added_token(allocator: Allocator, family: Family, input: []const u8) ![]u8 {
    return switch (family) {
        .qwen => nfc.normalize(allocator, input),
        .llama2 => normalize_llama2(allocator, input),
        .llama3, .smollm, .mistral => allocator.dupe(u8, input),
    };
}

fn normalize_metaspace(allocator: Allocator, input: []const u8, is_initial_text: bool) Allocator.Error![]u8 {
    var output: Utf8Bytes = .empty;
    errdefer output.deinit(allocator);
    for (input) |byte| {
        if (byte == ' ') {
            try output.appendSlice(allocator, "▁");
        } else {
            try output.append(allocator, byte);
        }
    }
    if (is_initial_text and input.len != 0 and !std.mem.startsWith(u8, output.items, "▁")) {
        var prefixed: Utf8Bytes = .empty;
        errdefer prefixed.deinit(allocator);
        try prefixed.appendSlice(allocator, "▁");
        try prefixed.appendSlice(allocator, output.items);
        output.deinit(allocator);
        return prefixed.toOwnedSlice(allocator);
    }
    return output.toOwnedSlice(allocator);
}

fn map_bytes_to_unicode(allocator: Allocator, input: []const u8) Allocator.Error![]u8 {
    var output: Utf8Bytes = .empty;
    errdefer output.deinit(allocator);
    for (input) |byte| {
        const codepoint = byte_to_unicode(byte);
        var encoded: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(codepoint, &encoded) catch unreachable;
        try output.appendSlice(allocator, encoded[0..len]);
    }
    return output.toOwnedSlice(allocator);
}

fn byte_to_unicode(byte: u8) u21 {
    if ((byte >= 0x21 and byte <= 0x7E) or (byte >= 0xA1 and byte <= 0xAC) or (byte >= 0xAE)) return byte;
    var offset: u21 = 0;
    for (0..@as(usize, byte)) |prior| {
        const value: u8 = @intCast(prior);
        if (!((value >= 0x21 and value <= 0x7E) or (value >= 0xA1 and value <= 0xAC) or (value >= 0xAE))) offset += 1;
    }
    return 0x0100 + offset;
}

fn unicode_to_byte(codepoint: u21) ?u8 {
    if ((codepoint >= 0x21 and codepoint <= 0x7E) or (codepoint >= 0xA1 and codepoint <= 0xAC) or (codepoint >= 0xAE and codepoint <= 0xFF)) return @intCast(codepoint);
    if (codepoint >= 0x0100 and codepoint <= 0x0143) {
        var remaining: u21 = codepoint - 0x0100;
        var byte: u16 = 0;
        while (byte <= 0xFF) : (byte += 1) {
            const value: u8 = @intCast(byte);
            if ((value >= 0x21 and value <= 0x7E) or (value >= 0xA1 and value <= 0xAC) or value >= 0xAE) continue;
            if (remaining == 0) return value;
            remaining -= 1;
        }
    }
    return null;
}

fn parse_byte_token(token: []const u8) ?u8 {
    if (token.len != 6 or !std.mem.startsWith(u8, token, "<0x") or token[5] != '>') return null;
    return std.fmt.parseInt(u8, token[3..5], 16) catch null;
}

fn byte_token_name(byte: u8) [6]u8 {
    const digits = "0123456789ABCDEF";
    return .{ '<', '0', 'x', digits[byte >> 4], digits[byte & 0x0F], '>' };
}

fn lossy_sentencepiece_bytes(allocator: Allocator, input: []const u8) Allocator.Error![]u8 {
    var output: Utf8Bytes = .empty;
    errdefer output.deinit(allocator);
    var index: usize = 0;
    while (index < input.len) {
        const width = std.unicode.utf8ByteSequenceLength(input[index]) catch {
            try output.appendSlice(allocator, "�");
            index += 1;
            continue;
        };
        const decoded = if (index + width <= input.len) std.unicode.utf8Decode(input[index .. index + width]) catch null else null;
        if (decoded == null) {
            try output.appendSlice(allocator, "�");
            index += 1;
            continue;
        }
        try output.appendSlice(allocator, input[index .. index + width]);
        index += width;
    }
    return output.toOwnedSlice(allocator);
}

fn lossy_utf8(allocator: Allocator, input: []const u8) Allocator.Error![]u8 {
    var output: Utf8Bytes = .empty;
    errdefer output.deinit(allocator);
    var index: usize = 0;
    while (index < input.len) {
        const width = std.unicode.utf8ByteSequenceLength(input[index]) catch {
            try output.appendSlice(allocator, "�");
            index += 1;
            continue;
        };
        if (width == 1) {
            try output.append(allocator, input[index]);
            index += 1;
            continue;
        }
        if (index + 1 >= input.len or !valid_utf8_second_byte(input[index], input[index + 1])) {
            try output.appendSlice(allocator, "�");
            index += 1;
            continue;
        }
        var consumed: usize = 2;
        while (consumed < width and index + consumed < input.len and is_utf8_continuation(input[index + consumed])) : (consumed += 1) {}
        if (consumed != width) {
            try output.appendSlice(allocator, "�");
            index += consumed;
            continue;
        }
        try output.appendSlice(allocator, input[index .. index + width]);
        index += width;
    }
    return output.toOwnedSlice(allocator);
}

fn is_utf8_continuation(byte: u8) bool {
    return byte >= 0x80 and byte <= 0xBF;
}

fn valid_utf8_second_byte(lead: u8, second: u8) bool {
    return switch (lead) {
        0xC2...0xDF => is_utf8_continuation(second),
        0xE0 => second >= 0xA0 and second <= 0xBF,
        0xE1...0xEC, 0xEE...0xEF => is_utf8_continuation(second),
        0xED => second >= 0x80 and second <= 0x9F,
        0xF0 => second >= 0x90 and second <= 0xBF,
        0xF1...0xF3 => is_utf8_continuation(second),
        0xF4 => second >= 0x80 and second <= 0x8F,
        else => false,
    };
}

test "byte-to-unicode table round trips all 256 byte values" {
    for (0..256) |value| {
        const byte: u8 = @intCast(value);
        try std.testing.expectEqual(@as(?u8, byte), unicode_to_byte(byte_to_unicode(byte)));
    }
}

test "byte fallback token parser accepts uppercase byte hex" {
    try std.testing.expectEqual(@as(?u8, 0), parse_byte_token("<0x00>"));
    try std.testing.expectEqual(@as(?u8, 10), parse_byte_token("<0x0A>"));
    try std.testing.expectEqual(@as(?u8, 255), parse_byte_token("<0xFF>"));
    try std.testing.expectEqual(@as(?u8, 255), parse_byte_token("<0xff>"));
}

test "byte-level lossy UTF-8 decoding uses maximal invalid subparts" {
    const allocator = std.testing.allocator;
    const incomplete_three = try lossy_utf8(allocator, &.{ 0xE2, 0x82 });
    defer allocator.free(incomplete_three);
    try std.testing.expectEqualStrings("�", incomplete_three);

    const incomplete_four = try lossy_utf8(allocator, &.{ 0xF0, 0x9F });
    defer allocator.free(incomplete_four);
    try std.testing.expectEqualStrings("�", incomplete_four);

    const interrupted = try lossy_utf8(allocator, &.{ 0xE2, 0x28, 0xA1 });
    defer allocator.free(interrupted);
    try std.testing.expectEqualStrings("�(�", interrupted);
}

test "Metaspace replaces spaces before checking its first prefix" {
    const allocator = std.testing.allocator;
    const leading_space = try normalize_metaspace(allocator, " world", true);
    defer allocator.free(leading_space);
    try std.testing.expectEqualStrings("▁world", leading_space);

    const text_after_added_token = try normalize_metaspace(allocator, "world", false);
    defer allocator.free(text_after_added_token);
    try std.testing.expectEqualStrings("world", text_after_added_token);
}

test "malformed nested pipeline JSON returns InvalidTokenizer" {
    const malformed = [_][]const u8{
        "{\"model\":{\"type\":\"BPE\"},\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":[null,null]}}",
        "{\"model\":{\"type\":\"BPE\"},\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[null,null]}}",
        "{\"model\":{\"type\":\"BPE\"},\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":null}},null]}}",
    };
    for (malformed) |bytes| {
        try std.testing.expectError(error.InvalidTokenizer, tokenizer_t.from_json(std.testing.allocator, bytes));
    }
}
