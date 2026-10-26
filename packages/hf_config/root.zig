//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Decoder-only LLM checkpoints: configuration parsing (config.zig) and the expected tensor
//! manifest compared with the checkpoint's safetensors headers (manifest.zig).

pub const config = @import("config.zig");
pub const manifest = @import("manifest.zig");

test {
    _ = config;
    _ = manifest;
    _ = @import("manifest_test.zig");
    _ = @import("fixtures_test.zig");
}
