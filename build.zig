//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Build script for ncmorph. The generic libraries in packages/ never import NCM code.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const bf16 = b.addModule("bf16", .{
        .root_source_file = b.path("packages/bf16/root.zig"),
        .target = target,
    });
    const kernels = b.addModule("kernels", .{
        .root_source_file = b.path("packages/kernels/root.zig"),
        .target = target,
        .link_libc = true,
        .imports = &.{.{ .name = "bf16", .module = bf16 }},
    });
    if (target.result.os.tag == .linux) kernels.linkSystemLibrary("m", .{});
    const unicode_categories = b.addModule("unicode_categories", .{
        .root_source_file = b.path("packages/unicode_categories/root.zig"),
        .target = target,
    });
    const safetensors = b.addModule("safetensors", .{
        .root_source_file = b.path("packages/safetensors/root.zig"),
        .target = target,
    });
    const hf_config = b.addModule("hf_config", .{
        .root_source_file = b.path("packages/hf_config/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "safetensors", .module = safetensors },
        },
    });
    const decoder = b.addModule("decoder", .{
        .root_source_file = b.path("src/decoder.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "hf_config", .module = hf_config },
            .{ .name = "kernels", .module = kernels },
        },
    });
    const ncm_norm = b.addModule("ncm_norm", .{
        .root_source_file = b.path("src/ncm_norm.zig"),
        .target = target,
        .imports = &.{.{ .name = "kernels", .module = kernels }},
    });
    const ncm_manifest = b.addModule("ncm_manifest", .{
        .root_source_file = b.path("src/ncm_manifest.zig"),
        .target = target,
    });
    const ncm_position = b.addModule("ncm_position", .{
        .root_source_file = b.path("src/ncm_position.zig"),
        .target = target,
        .imports = &.{.{ .name = "kernels", .module = kernels }},
    });
    const ncm_body = b.addModule("ncm_body", .{
        .root_source_file = b.path("src/ncm_body.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "decoder", .module = decoder },
            .{ .name = "hf_config", .module = hf_config },
            .{ .name = "kernels", .module = kernels },
            .{ .name = "ncm_norm", .module = ncm_norm },
        },
    });
    const ncm_mask = b.addModule("ncm_mask", .{
        .root_source_file = b.path("src/ncm_mask.zig"),
        .target = target,
    });
    const ncm_model = b.addModule("ncm_model", .{
        .root_source_file = b.path("src/ncm_model.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "kernels", .module = kernels },
            .{ .name = "ncm_mask", .module = ncm_mask },
            .{ .name = "ncm_norm", .module = ncm_norm },
            .{ .name = "ncm_position", .module = ncm_position },
        },
    });
    const loader = b.addModule("loader", .{
        .root_source_file = b.path("src/loader.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "bf16", .module = bf16 },
            .{ .name = "decoder", .module = decoder },
            .{ .name = "ncm_body", .module = ncm_body },
            .{ .name = "ncm_mask", .module = ncm_mask },
            .{ .name = "ncm_model", .module = ncm_model },
            .{ .name = "ncm_position", .module = ncm_position },
            .{ .name = "ncm_norm", .module = ncm_norm },
            .{ .name = "hf_config", .module = hf_config },
            .{ .name = "safetensors", .module = safetensors },
        },
    });

    const exe = b.addExecutable(.{
        .name = "ncmorph",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "safetensors", .module = safetensors },
                .{ .name = "hf_config", .module = hf_config },
                .{ .name = "decoder", .module = decoder },
                .{ .name = "loader", .module = loader },
            },
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run the ncmorph command-line tool");
    run_step.dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run all tests");
    const library_modules = [_]*std.Build.Module{ bf16, kernels, unicode_categories, safetensors, hf_config, decoder, ncm_norm, ncm_manifest, ncm_position, ncm_body, ncm_mask, ncm_model, loader };
    for (library_modules) |module| {
        const tests = b.addTest(.{ .root_module = module });
        test_step.dependOn(&b.addRunArtifact(tests).step);
    }
    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    test_step.dependOn(&b.addRunArtifact(exe_tests).step);
}
