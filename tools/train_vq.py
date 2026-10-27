#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Train an OWT VQ tokenizer and export its reference-compatible state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from vq_checkpoint import load_latest, save_generation

RepositoryRoot = Path(__file__).resolve().parent.parent
WindowFormat = struct.Struct("<I")
DefaultLevels = [index * index for index in range(1, 33)]
TorchVersion = "2.13.0"


def check_versions() -> None:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"V2 requires Python 3.13; found {sys.version.split()[0]}")
    try:
        actual = importlib.metadata.version("torch").split("+", maxsplit=1)[0]
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"V2 requires torch=={TorchVersion}; package is missing") from error
    if actual != TorchVersion:
        raise RuntimeError(f"V2 requires torch=={TorchVersion}; found {actual}")


def _plan_cascade_strides(length: int, base: int = 4) -> list[int]:
    strides = []
    remaining = length
    while remaining > 1:
        if remaining % base:
            raise ValueError(f"sequence length {length} must be divisible by cascade base {base}")
        strides.append(base)
        remaining //= base
    return strides


class _encoder_t(nn.Module):
    def __init__(self, vocab_size: int, length: int, width: int, dropout: float):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, width)
        self.wpe = nn.Embedding(length, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self.drop(self.wte(token_ids) + self.wpe(positions)[None, :, :])


class _phi_t(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, width, 3))
        self.bias = nn.Parameter(torch.empty(width))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value: Tensor) -> Tensor:
        return 0.5 * value + 0.5 * functional.conv1d(value.transpose(1, 2), self.weight, self.bias, padding=1).transpose(1, 2)


class _vq_t(nn.Module):
    def __init__(self, width: int, levels: list[int], codebook_size: int):
        super().__init__()
        if not levels or levels[-1] <= 0 or levels[0] != 1:
            raise ValueError("VQ levels must begin at 1 and end at the sequence length")
        self.levels = levels
        self.commitment_cost = 0.25
        self.decay = 0.99
        self.epsilon = 1e-5
        self.embeddings = nn.ModuleList()
        self.quant_resi = nn.ModuleList()
        for level_index in range(len(levels)):
            embedding = nn.Embedding(codebook_size, width)
            nn.init.normal_(embedding.weight)
            embedding.weight.requires_grad_(False)
            self.embeddings.append(embedding)
            self.register_buffer(f"ema_cluster_size_{level_index}", torch.zeros(codebook_size))
            self.register_buffer(f"ema_w_{level_index}", embedding.weight.detach().clone())
            self.register_buffer(f"codebook_hit_{level_index}", torch.zeros(codebook_size, dtype=torch.int64))
            self.quant_resi.append(_phi_t(width))
        self.learned_down_l1 = nn.ModuleList()
        self.learned_up_l1 = nn.ModuleList()
        strides = _plan_cascade_strides(levels[-1])
        for stride in strides:
            down = nn.Conv1d(width, width, stride, stride=stride)
            with torch.no_grad():
                down.weight.zero_()
                for channel in range(width):
                    down.weight[channel, channel, :] = 1 / stride
                down.bias.zero_()
            self.learned_down_l1.append(down)
        for stride in reversed(strides):
            up = nn.ConvTranspose1d(width, width, stride, stride=stride)
            with torch.no_grad():
                up.weight.zero_()
                for channel in range(width):
                    up.weight[channel, channel, :] = 1
                up.bias.zero_()
            self.learned_up_l1.append(up)

    def _downsample(self, residual: Tensor, level_index: int) -> Tensor:
        level_length = self.levels[level_index]
        if level_index == 0:
            down = residual.transpose(1, 2)
            for layer in self.learned_down_l1:
                down = layer(down)
            return down.transpose(1, 2)
        if level_length == self.levels[-1]:
            return residual
        return functional.interpolate(residual.transpose(1, 2), size=level_length, mode="area").transpose(1, 2)

    def _upsample(self, value: Tensor, level_index: int) -> Tensor:
        if level_index == 0:
            up = value.transpose(1, 2)
            for layer in self.learned_up_l1:
                up = layer(up)
            return up.transpose(1, 2)
        if self.levels[level_index] == self.levels[-1]:
            return value
        return functional.interpolate(value.transpose(1, 2), size=self.levels[-1], mode="linear").transpose(1, 2)

    @torch.no_grad()
    def _ema_update(self, level_index: int, vectors: Tensor, indices: Tensor) -> None:
        counts = torch.bincount(indices.reshape(-1), minlength=self.embeddings[level_index].num_embeddings).to(vectors.dtype)
        sums = torch.zeros_like(self.embeddings[level_index].weight)
        sums.index_add_(0, indices.reshape(-1), vectors.reshape(-1, vectors.shape[-1]))
        cluster = getattr(self, f"ema_cluster_size_{level_index}")
        ema_w = getattr(self, f"ema_w_{level_index}")
        cluster.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        total = cluster.sum()
        smoothed = (cluster + self.epsilon) / (total + cluster.numel() * self.epsilon) * total
        ema_w.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        self.embeddings[level_index].weight.copy_(ema_w / smoothed.unsqueeze(-1).clamp_min(torch.finfo(ema_w.dtype).tiny))
        getattr(self, f"codebook_hit_{level_index}")[indices.unique()] = 1

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        with torch.autocast(device_type=value.device.type, enabled=False):
            return self._forward_float32(value.float())

    def _forward_float32(self, value: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        residual = value.detach().float()
        cumulative = torch.zeros_like(residual)
        losses = []
        cumulative_levels = []
        for level_index, embedding in enumerate(self.embeddings):
            down = self._downsample(residual, level_index)
            codebook = embedding.weight.float()
            distances = down.square().sum(-1, keepdim=True) + codebook.square().sum(-1)[None, None, :] - 2 * down @ codebook.t()
            indices = distances.argmin(dim=-1)
            if self.training:
                self._ema_update(level_index, down.detach(), indices)
            quantized = functional.embedding(indices, embedding.weight.float())
            if level_index == 0:
                quantized = down + (quantized - down).detach()
            up = self._upsample(quantized, level_index)
            contribution = self.quant_resi[level_index](up)
            cumulative = cumulative + contribution
            losses.append(self.commitment_cost * functional.mse_loss(cumulative.detach(), value.float()) + functional.mse_loss(cumulative, value.detach().float()))
            residual = residual - contribution.detach()
            cumulative_levels.append(cumulative)
        return cumulative, torch.stack(losses).mean(), cumulative_levels

    def _forward_with_indices(self, value: Tensor, indices_by_level: list[Tensor]) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Replay quantization with fixed assignments and codebooks, without EMA writes."""
        residual = value.detach().float()
        cumulative = torch.zeros_like(residual)
        losses = []
        cumulative_levels = []
        for level_index, (embedding, stored_indices) in enumerate(zip(self.embeddings, indices_by_level, strict=True)):
            down = self._downsample(residual, level_index)
            indices = stored_indices.to(device=value.device, dtype=torch.long)
            quantized = functional.embedding(indices, embedding.weight.float())
            if level_index == 0:
                quantized = down + (quantized - down).detach()
            contribution = self.quant_resi[level_index](self._upsample(quantized, level_index))
            cumulative = cumulative + contribution
            losses.append(self.commitment_cost * functional.mse_loss(cumulative.detach(), value.float())
                          + functional.mse_loss(cumulative, value.detach().float()))
            residual = residual - contribution.detach()
            cumulative_levels.append(cumulative)
        return cumulative, torch.stack(losses).mean(), cumulative_levels

    @torch.no_grad()
    def _ema_update_aggregated(self, level_index: int, counts: Tensor, sums: Tensor, touched: Tensor) -> None:
        cluster = getattr(self, f"ema_cluster_size_{level_index}")
        ema_w = getattr(self, f"ema_w_{level_index}")
        cluster.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        total = cluster.sum()
        smoothed = (cluster + self.epsilon) / (total + cluster.numel() * self.epsilon) * total
        ema_w.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        self.embeddings[level_index].weight.copy_(ema_w / smoothed.unsqueeze(-1).clamp_min(torch.finfo(ema_w.dtype).tiny))
        getattr(self, f"codebook_hit_{level_index}")[touched] = 1


class _rms_norm_t(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.epsilon) * self.weight


class _attention_t(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by decoder heads")
        self.heads = heads
        self.head_width = width // heads
        self.c_attn = nn.Linear(width, 3 * width, bias=True)
        self.c_proj = nn.Linear(width, width, bias=True)
        self.q_norm = _rms_norm_t(self.head_width)
        self.k_norm = _rms_norm_t(self.head_width)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        batch, length, width = value.shape
        query, key, content = self.c_attn(value).split(width, dim=-1)
        query = query.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        content = content.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        result = functional.scaled_dot_product_attention(query, key, content, dropout_p=self.attn_dropout if self.training else 0.0)
        result = result.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.c_proj(result))


class _block_t(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, bias=True, eps=1e-6)
        self.attn = _attention_t(width, heads, dropout)
        self.ln_2 = nn.LayerNorm(width, bias=True, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(width, 4 * width, bias=True)
        self.mlp.gelu = nn.GELU()
        self.mlp.c_proj = nn.Linear(4 * width, width, bias=True)
        self.mlp.resid_dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.attn(self.ln_1(value))
        hidden = self.mlp.gelu(self.mlp.c_fc(self.ln_2(value)))
        value = value + self.mlp.resid_dropout(self.mlp.c_proj(hidden))
        return value


class _decoder_t(nn.Module):
    def __init__(self, vocab_size: int, width: int, heads: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.middle_block = _block_t(width, heads, dropout)
        self.ln_f = nn.LayerNorm(width, bias=False, eps=1e-6)
        self.output_layer = nn.Linear(width, vocab_size, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.output_layer(self.ln_f(self.middle_block(value)))


class vqvae_t(nn.Module):
    def __init__(self, vocab_size: int, length: int, width: int = 384, levels: list[int] | None = None,
                 codebook_size: int = 4096, heads: int = 6, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.length = length
        self.width = width
        self.levels = list(levels if levels is not None else DefaultLevels)
        if not self.levels or self.levels[-1] != length or any(a >= b for a, b in zip(self.levels, self.levels[1:])):
            raise ValueError("levels must be strictly increasing and end at sequence length")
        self.encoder = _encoder_t(vocab_size, length, width, dropout)
        self.pre_quant_drop = nn.Dropout(0.0)
        self.pre_quant_norm = nn.GroupNorm(8, width, affine=False)
        self.vq = _vq_t(width, self.levels, codebook_size)
        self.decoder = _decoder_t(vocab_size, width, heads, dropout)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.encoder(token_ids)
        encoded = self.pre_quant_norm(self.pre_quant_drop(encoded).transpose(1, 2)).transpose(1, 2)
        _, vq_loss, cumulative_levels = self.vq(encoded)
        reconstruction = cumulative_levels[-1]
        if self.training and len(self.levels) > 1 and random.random() < 0.5:
            reconstruction = cumulative_levels[random.randrange(0, len(self.levels) - 1)]
        straight_through = encoded.float() + (reconstruction - encoded.float()).detach()
        logits = self.decoder(straight_through)
        reconstruction_loss = functional.cross_entropy(logits.reshape(-1, self.vocab_size), token_ids.reshape(-1))
        return logits, vq_loss + reconstruction_loss, vq_loss


class _packed_windows_t:
    """Memory-map A1's packed files and fetch only the batches used by training."""

    def __init__(self, paths: list[Path], length: int):
        self.length = length
        self.files: list[Tensor] = []
        self.offsets = [0]
        for path in paths:
            size = path.stat().st_size
            if size % (WindowFormat.size * length):
                raise ValueError(f"packed window file has a partial record: {path}")
            count = size // (WindowFormat.size * length)
            raw = torch.from_file(str(path), shared=False, size=count * length, dtype=torch.int32).view(count, length)
            self.files.append(raw)
            self.offsets.append(self.offsets[-1] + count)
        if not self.files or self.offsets[-1] == 0:
            raise ValueError("no packed training windows were found")

    @property
    def count(self) -> int:
        return self.offsets[-1]

    def batch(self, indices: Tensor) -> Tensor:
        ends = torch.tensor(self.offsets[1:], dtype=torch.int64)
        file_indices = torch.bucketize(indices, ends, right=True)
        result = torch.empty((indices.numel(), self.length), dtype=torch.int32)
        for file_index in file_indices.unique().tolist():
            mask = file_indices == file_index
            local_indices = indices[mask] - self.offsets[file_index]
            result[mask] = self.files[file_index][local_indices]
        return result.long()


def _learning_rate(step: int, total_steps: int, warmup_steps: int) -> float:
    peak = 3e-4
    floor = 3e-5
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def _train_batch_memory_bounded(model: vqvae_t, optimizer: torch.optim.Optimizer,
                                parameters: list[nn.Parameter], batch: Tensor, microbatch_size: int,
                                device: str) -> Tensor:
    """Run one recipe-sized batch while keeping activation memory to one piece."""
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    batch_size = batch.shape[0]
    chunks = [(start, min(start + microbatch_size, batch_size))
              for start in range(0, batch_size, microbatch_size)]
    device_type = device.split(":")[0]
    saved_rng: list[tuple[Tensor, Tensor | None]] = []
    encoded_chunks = []
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for start, end in chunks:
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
            saved_rng.append((cpu_state, cuda_state))
            token_ids = batch[start:end].to(device=device, dtype=torch.long)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                encoded = model.encoder(token_ids)
                encoded = model.pre_quant_norm(model.pre_quant_drop(encoded).transpose(1, 2)).transpose(1, 2)
            encoded_chunks.append(encoded.float().cpu())

        residual_chunks = encoded_chunks
        assignments: list[list[Tensor]] = [[] for _ in model.levels]
        for level_index, embedding in enumerate(model.vq.embeddings):
            counts = torch.zeros(embedding.num_embeddings, device=device, dtype=torch.float32)
            sums = torch.zeros_like(embedding.weight, device=device, dtype=torch.float32)
            touched = torch.zeros(embedding.num_embeddings, device=device, dtype=torch.bool)
            level_indices = []
            for residual_cpu in residual_chunks:
                residual = residual_cpu.to(device=device)
                down = model.vq._downsample(residual, level_index)
                codebook = embedding.weight.float()
                distances = down.square().sum(-1, keepdim=True) + codebook.square().sum(-1)[None, None, :] - 2 * down @ codebook.t()
                indices = distances.argmin(dim=-1)
                flat_indices = indices.reshape(-1)
                vectors = down.reshape(-1, down.shape[-1])
                counts.add_(torch.bincount(flat_indices, minlength=embedding.num_embeddings).float())
                sums.index_add_(0, flat_indices, vectors)
                touched[flat_indices.unique()] = True
                level_indices.append(indices.cpu())
            model.vq._ema_update_aggregated(level_index, counts, sums, touched)
            assignments[level_index] = level_indices
            updated_chunks = []
            for residual_cpu, indices in zip(residual_chunks, level_indices, strict=True):
                residual = residual_cpu.to(device=device)
                indices = indices.to(device=device, dtype=torch.long)
                quantized = functional.embedding(indices, embedding.weight.float())
                up = model.vq._upsample(quantized, level_index)
                contribution = model.vq.quant_resi[level_index](up)
                updated_chunks.append((residual - contribution.detach()).cpu())
            residual_chunks = updated_chunks

    # The cutoff is a property of the global batch. Pick it once, then replay each
    # piece from its encoder RNG state so encoder and decoder dropout match.
    cutoff = len(model.levels) - 1
    if model.training and len(model.levels) > 1 and random.random() < 0.5:
        cutoff = random.randrange(0, len(model.levels) - 1)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    for chunk_index, (start, end) in enumerate(chunks):
        cpu_state, cuda_state = saved_rng[chunk_index]
        replay_stream_cpu = torch.get_rng_state()
        replay_stream_cuda = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        token_ids = batch[start:end].to(device=device, dtype=torch.long)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            encoded = model.encoder(token_ids)
            encoded = model.pre_quant_norm(model.pre_quant_drop(encoded).transpose(1, 2)).transpose(1, 2)
        # The cached encoder RNG reproduces its first-pass dropout. Keep decoder
        # dropout on the advancing global stream so chunks do not reuse masks.
        torch.set_rng_state(replay_stream_cpu)
        if replay_stream_cuda is not None:
            torch.cuda.set_rng_state(replay_stream_cuda, device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            with torch.autocast(device_type=device_type, enabled=False):
                _, vq_loss, cumulative_levels = model.vq._forward_with_indices(
                    encoded.float(), [indices[chunk_index] for indices in assignments])
            reconstruction = cumulative_levels[cutoff]
            straight_through = encoded.float() + (reconstruction - encoded.float()).detach()
            logits = model.decoder(straight_through)
            reconstruction_loss = functional.cross_entropy(
                logits.reshape(-1, model.vocab_size), token_ids.reshape(-1))
            loss = vq_loss + reconstruction_loss
        (loss * ((end - start) / batch_size)).backward()
        total_loss += loss.detach().float() * ((end - start) / batch_size)
        del logits, reconstruction_loss, loss, vq_loss, cumulative_levels, reconstruction, straight_through, encoded
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    return total_loss


def _state_manifest(model: vqvae_t, raw_dir: Path) -> Path:
    entries = []
    for index, (name, tensor) in enumerate(sorted(model.state_dict().items())):
        if tensor.dtype not in (torch.float32, torch.int64):
            raise TypeError(f"unsupported trained VQ tensor dtype {tensor.dtype} for {name}")
        payload_name = f"tensor-{index:03d}.bin"
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        (raw_dir / payload_name).write_bytes(raw)
        entries.append({"name": name, "dtype": "F32" if tensor.dtype == torch.float32 else "I64",
                        "dtype_name": "F32" if tensor.dtype == torch.float32 else "I64",
                        "shape": list(tensor.shape), "payload": payload_name})
    manifest = raw_dir / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_with_w1(raw_dir: Path, manifest: Path, zig: str, source_repository: str, source_revision: str) -> Path:
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    payload_reads = []
    tensor_rows = []
    for index, entry in enumerate(entries):
        shape_text = ", ".join(str(dimension) for dimension in entry["shape"])
        payload_reads.append(
            f'    const payload_{index} = try cwd.readFileAlloc(init.io, {json.dumps(entry["payload"])}, arena, .unlimited);'
        )
        tensor_rows.append(
            "    .{ .name = " + json.dumps(entry["name"]) +
            f", .dtype = .{entry['dtype']}, .shape = &.{{ {shape_text} }}, .data = payload_{index} }},"
        )
    source_path = raw_dir / "writer.zig"
    source_path.write_text(
        "//\n// Ko, (Baczynski-Bruce) 2026; Apache 2.0\n//\n"
        'const std = @import("std");\nconst safetensors = @import("safetensors");\n'
        "pub fn main(init: std.process.Init) !void {\n"
        "    const arena = init.arena.allocator();\n    const cwd = std.Io.Dir.cwd();\n"
        + "\n".join(payload_reads) + "\n"
        + "    const tensors = [_]safetensors.tensor_data_t{\n"
        + "\n".join(tensor_rows) + "\n    };\n"
        + "    const metadata = [_]safetensors.metadata_entry_t{\n"
        + '        .{ .key = "ncmorph_format", .value = "1" },\n'
        + f'        .{{ .key = "source_repository", .value = {json.dumps(source_repository)} }},\n'
        + f'        .{{ .key = "source_revision", .value = {json.dumps(source_revision)} }},\n'
        + "    };\n"
        + "    const bytes = try safetensors.serialize(arena, &tensors, &metadata);\n"
        + '    try cwd.writeFile(init.io, .{ .sub_path = "tokenizer.safetensors", .data = bytes });\n}\n',
        encoding="utf-8",
    )
    command = [zig, "run", "--cache-dir", str(raw_dir / "zig-cache"), "--global-cache-dir",
               str(raw_dir / "zig-global-cache"), "--dep", "safetensors", f"-Mroot={source_path}",
               f"-Msafetensors={RepositoryRoot / 'packages' / 'safetensors' / 'root.zig'}"]
    result = subprocess.run(command, cwd=raw_dir, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout or "W1 safetensors writer failed")
    output = raw_dir / "tokenizer.safetensors"
    if not output.is_file():
        raise RuntimeError("W1 writer completed without producing tokenizer.safetensors")
    return output


def _optimizer_parameter_groups(parameters: list[nn.Parameter]) -> list[dict[str, Any]]:
    return [
        {"params": [parameter for parameter in parameters if parameter.ndim >= 2], "weight_decay": 0.01},
        {"params": [parameter for parameter in parameters if parameter.ndim < 2], "weight_decay": 0.0},
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported JSON checkpoint value: {type(value).__name__}")


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def _optimizer_checkpoint(optimizer: torch.optim.Optimizer, tensors: dict[str, Tensor]) -> dict[str, Any]:
    state_dict = optimizer.state_dict()
    saved_state = {}
    for parameter_id, values in state_dict["state"].items():
        serialized = {}
        for name, value in values.items():
            if isinstance(value, Tensor):
                key = f"optimizer.{parameter_id}.{name}"
                tensors[key] = value.detach().cpu().contiguous()
                serialized[name] = {"tensor": key}
            else:
                serialized[name] = _jsonable(value)
        saved_state[str(parameter_id)] = serialized
    groups = []
    for group in state_dict["param_groups"]:
        groups.append({key: _jsonable(value) for key, value in group.items()})
    return {"state": saved_state, "param_groups": groups}


def _restore_optimizer(optimizer: torch.optim.Optimizer, saved: dict[str, Any], tensors: dict[str, Tensor]) -> None:
    state = {}
    for parameter_id, values in saved["state"].items():
        restored = {}
        for name, value in values.items():
            if isinstance(value, dict) and set(value) == {"tensor"}:
                key = value["tensor"]
                if key not in tensors:
                    raise ValueError(f"VQ checkpoint is missing optimizer tensor {key}")
                restored[name] = tensors[key]
            else:
                restored[name] = value
        state[int(parameter_id)] = restored
    optimizer.load_state_dict({"state": state, "param_groups": saved["param_groups"]})


def _hash_data(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return result


def _training_data_provenance(data_summary: dict[str, Any], explicit_provenance: dict[str, Any] | None) -> dict[str, Any]:
    """Return recorded A1 identity, or require a separate record for legacy data."""
    summary_identity = data_summary.get("identity")
    if summary_identity is not None:
        if not isinstance(summary_identity, dict) or not summary_identity:
            raise ValueError("A1 summary identity must be a non-empty JSON object")
        if explicit_provenance is not None and explicit_provenance != summary_identity:
            raise ValueError("explicit data provenance does not match the A1 summary identity")
        return summary_identity
    if explicit_provenance is None:
        raise ValueError(
            "legacy A1 summary has no provenance identity; provide an independently recorded "
            "JSON object with --provenance-file"
        )
    if not isinstance(explicit_provenance, dict) or not explicit_provenance:
        raise ValueError("explicit data provenance must be a non-empty JSON object")
    return explicit_provenance


def _read_provenance_file(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read data provenance JSON {path}: {error}") from error
    if not isinstance(value, dict) or not value:
        raise ValueError("data provenance file must contain a non-empty JSON object")
    return value


def train(data_dir: Path, output_dir: Path, vocab_size: int, source_family: str, source_repository: str,
          source_revision: str,
          length: int = 1024, levels: list[int] | None = None, width: int = 384, codebook_size: int = 4096,
          heads: int = 6, batch_size: int = 768, updates: int = 78_432, warmup_steps: int = 1_000,
          seed: int = 42, max_steps: int | None = None, device: str | None = None, zig: str = "zig",
          microbatch_size: int = 8, resume: bool = False, checkpoint_every: int = 1_000,
          stop_after: int | None = None, data_provenance: dict[str, Any] | None = None) -> Path:
    check_versions()
    if output_dir.resolve() == RepositoryRoot or RepositoryRoot in output_dir.resolve().parents:
        raise ValueError("VQ output must be outside the ncmorph repository")
    level_sizes = list(levels if levels is not None else DefaultLevels)
    if not level_sizes or level_sizes[-1] != length:
        raise ValueError("last VQ level must equal sequence length")
    torch.manual_seed(seed)
    random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary_path = data_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"A1 summary.json is missing from {data_dir}")
    data_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data_summary, dict):
        raise ValueError("A1 summary.json must contain a JSON object")
    recorded_provenance = _training_data_provenance(data_summary, data_provenance)
    if data_summary.get("window_length") != length:
        raise ValueError(
            f"A1 data window_length is {data_summary.get('window_length')}; trainer length is {length}"
        )
    if data_summary.get("seed") != seed:
        raise ValueError(f"A1 data seed is {data_summary.get('seed')}; trainer seed is {seed}")
    paths = sorted(data_dir.glob("train_epoch_*.u32le"))
    dataset = _packed_windows_t(paths, length)
    expected_windows = data_summary.get("training_windows_per_epoch", 0) * data_summary.get("epochs", 0)
    if dataset.count != expected_windows:
        raise ValueError(f"A1 summary reports {expected_windows} training windows; found {dataset.count}")
    if dataset.count < batch_size:
        raise ValueError(f"training data has {dataset.count} windows, fewer than batch size {batch_size}")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    model = vqvae_t(vocab_size, length, width, level_sizes, codebook_size, heads).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(parameters),
        lr=3e-4,
        betas=(0.9, 0.95),
    )
    total_steps = max_steps if max_steps is not None else updates
    if total_steps <= 0 or checkpoint_every <= 0 or (stop_after is not None and stop_after <= 0):
        raise ValueError("total updates, checkpoint interval, and stop_after must be positive")
    checkpoint_dir = output_dir / "checkpoints"
    data_identity = {"summary": data_summary, "provenance": recorded_provenance, "files": _hash_data(paths)}
    identity = {
        "model": {"vocab_size": vocab_size, "length": length, "levels": level_sizes, "width": width,
                  "codebook_size": codebook_size, "heads": heads, "dropout": 0.3},
        "training": {"source_family": source_family, "source_repository": source_repository,
                     "source_revision": source_revision, "batch_size": batch_size, "updates": total_steps,
                     "warmup_steps": warmup_steps, "seed": seed, "microbatch_size": microbatch_size,
                     "device": device, "precision": "bf16_mixed_f32_master"},
        "data": data_identity,
    }
    model.train()
    step = 0
    shuffle_rng = torch.Generator().manual_seed(seed)
    epoch_index = 0
    epoch_order = torch.empty(0, dtype=torch.int64)
    batch_offset = 0
    generation_number = 0
    last_loss: float | None = None
    usable_count = dataset.count - dataset.count % batch_size

    if resume:
        saved_metadata, saved_tensors = load_latest(checkpoint_dir, identity)
        scalar = saved_metadata["scalar_state"]
        model_state = {name.removeprefix("model."): value for name, value in saved_tensors.items()
                       if name.startswith("model.")}
        expected_names = set(model.state_dict())
        if set(model_state) != expected_names:
            raise ValueError("VQ checkpoint model tensor manifest mismatch")
        model.load_state_dict(model_state, strict=True)
        _restore_optimizer(optimizer, scalar["optimizer"], saved_tensors)
        step = scalar["step"]
        epoch_index = scalar["epoch_index"]
        batch_offset = scalar["batch_offset"]
        last_loss = scalar.get("last_loss")
        epoch_order = saved_tensors["data.epoch_order"].long()
        if type(step) is not int or not 0 <= step <= total_steps:
            raise ValueError(f"VQ checkpoint update count is outside the schedule: {step!r}")
        if type(epoch_index) is not int or epoch_index < 0:
            raise ValueError(f"VQ checkpoint epoch index is invalid: {epoch_index!r}")
        if type(batch_offset) is not int or not 0 <= batch_offset <= usable_count or batch_offset % batch_size:
            raise ValueError(f"VQ checkpoint batch offset is invalid: {batch_offset!r}")
        if epoch_order.ndim != 1 or epoch_order.numel() != dataset.count:
            raise ValueError("VQ checkpoint data order does not match the training dataset")
        shuffle_rng.set_state(saved_tensors["rng.shuffle"].to(torch.uint8))
        torch.set_rng_state(saved_tensors["rng.torch_cpu"].to(torch.uint8))
        if device.startswith("cuda"):
            if "rng.torch_cuda" not in saved_tensors:
                raise ValueError("VQ checkpoint lacks CUDA random state")
            torch.cuda.set_rng_state(saved_tensors["rng.torch_cuda"].to(torch.uint8), device)
        random.setstate(_tuple_tree(scalar["rng_python"]))
        generation_number = int(saved_metadata["generation"].split("-")[-1])
    elif (checkpoint_dir / "latest.json").exists():
        raise ValueError(f"VQ checkpoints already exist in {checkpoint_dir}; pass resume=True to continue")

    def save_checkpoint() -> None:
        nonlocal generation_number
        generation_number += 1
        tensor_values = {f"model.{name}": value for name, value in model.state_dict().items()}
        tensor_values.update({"rng.shuffle": shuffle_rng.get_state(),
                              "rng.torch_cpu": torch.get_rng_state(),
                              "data.epoch_order": epoch_order})
        if device.startswith("cuda"):
            tensor_values["rng.torch_cuda"] = torch.cuda.get_rng_state(device)
        optimizer_values = _optimizer_checkpoint(optimizer, tensor_values)
        scalar_values = {"step": step, "epoch_index": epoch_index, "batch_offset": batch_offset,
                         "last_loss": last_loss, "rng_python": _jsonable(random.getstate()),
                         "optimizer": optimizer_values}
        save_generation(checkpoint_dir, identity, tensor_values, scalar_values, generation_number)

    stop_step = total_steps if stop_after is None else min(total_steps, step + stop_after)
    while step < stop_step:
        if epoch_order.numel() == 0 or batch_offset >= usable_count:
            epoch_order = torch.randperm(dataset.count, generator=shuffle_rng)
            epoch_index += 1
            batch_offset = 0
        batch = dataset.batch(epoch_order[batch_offset : batch_offset + batch_size])
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step, total_steps, warmup_steps)
        loss = _train_batch_memory_bounded(model, optimizer, parameters, batch, microbatch_size, device)
        step += 1
        batch_offset += batch_size
        last_loss = float(loss.detach().float().item())
        if step % 100 == 0 or step == total_steps:
            print(f"step {step}/{total_steps} loss={loss.detach().float().item():.6f}", flush=True)
        if step % checkpoint_every == 0 or step == stop_step or step == total_steps:
            save_checkpoint()

    model.cpu().eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ncmorph-vq-w1-") as temporary:
        raw_dir = Path(temporary)
        manifest = _state_manifest(model, raw_dir)
        written = _write_with_w1(raw_dir, manifest, zig, source_repository, source_revision)
        destination = output_dir / "tokenizer.safetensors"
        shutil.copyfile(written, destination)
    specs = [{"name": name, "shape": list(tensor.shape), "dtype": "F32" if tensor.dtype == torch.float32 else "I64"}
             for name, tensor in sorted(model.state_dict().items())]
    metadata = {
        "format": 1,
        "tokenizer_route": "trained",
        "source": {"family": source_family, "repository": source_repository, "revision": source_revision},
        "tokenizer_config": {"K": len(level_sizes), "L": length, "level_sizes": level_sizes,
                             "codebook_sizes": [codebook_size] * len(level_sizes), "width": width,
                             "vocabulary_size": vocab_size},
        "recipe": {"updates": step, "global_batch": batch_size, "peak_learning_rate": 3e-4,
                   "learning_rate_floor": 3e-5, "warmup_updates": warmup_steps, "seed": seed,
                   "microbatch_size": microbatch_size, "precision": "bf16_mixed_f32_master",
                   "weight_decay": 0.01, "gradient_clip": 1.0},
        "tensor_schema": specs,
        "tokenizer": {"filename": destination.name, "size_bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "stored_dtype": "F32"},
    }
    (output_dir / "ncm.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--source-family", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--updates", type=int, default=78_432)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="resume from the latest complete checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--stop-after", type=int, help="stop after this many updates in this invocation")
    parser.add_argument("--provenance-file", type=Path,
                        help="independently recorded provenance JSON required for legacy A1 summaries")
    parser.add_argument("--device")
    parser.add_argument("--zig", default="zig")
    args = parser.parse_args()
    try:
        path = train(args.data_dir, args.output_dir, args.vocab_size, args.source_family,
                     args.source_repository, args.source_revision, args.length, args.levels, args.width,
                     args.codebook_size, args.heads, args.batch_size, args.updates,
                     args.warmup_steps, args.seed, args.max_steps, args.device, args.zig, args.microbatch_size,
                     args.resume, args.checkpoint_every, args.stop_after,
                     _read_provenance_file(args.provenance_file))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"VQ training failed: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
