# Software sources: masks, tokenizers, safetensors, tokenizer heads

Read-only research, 2026-10-26. Raw files are cached under `/private/tmp/claude-501/-Users-jessicabruce--claude-spares-spare-e7phlw/19e02558-926e-417a-8cc2-e33c5915494f/scratchpad/research/` in `src_transformers/`, `src_tokenizers_v0.23.3/`, `src_safetensors_v0.8.0/`, `src_oniguruma_94877ac/`, `hf_data/<repo>/{tokenizer_head.json,config.json}`; scripts in `scripts/`. Line numbers refer to those copies. `UNVERIFIED:` marks anything not checked in source or JSON.

## 0. Pins

| Item | Pin | Evidence |
|---|---|---|
| transformers | v5.20.2 = `18329f1065382e54db698e242cf98bd3f3e5a766` | committed 2026-10-19T17:34:25Z, "Release: v5.20.2" |
| tokenizers range | `"tokenizers": "tokenizers>=0.23.1,<0.24.0",` | dependency_versions_table.py:72 |
| tokenizers tag | **v0.23.3**, commit `301277072263fb3830eb66eec7e47adadf76f1dc` (annotated tag object `9ab7691f…`), published 2026-10-09 | Newest final tag in range (`git ls-remote --tags`: v0.23.1, v0.23.2, v0.23.3; next is v1.0.0-rc.0). Release page HEAD: 200. Title is "Release v0.23.3 (python only)" and `tokenizers/Cargo.toml:8` still reads `version = "0.23.2"`. The 17 `tokenizers/src/...` files cited below are byte-identical in v0.23.1, v0.23.2 and v0.23.3 (diff: 0 lines each); `Cargo.toml` and `bindings/` were not diffed. |
| safetensors range | `"safetensors": "safetensors>=0.8.0",` | dependency_versions_table.py:58 (no upper bound) |
| safetensors tag | **v0.8.0**, commit `a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6`, published 2026-06-09 | Newest non-prerelease. v0.9.0-rc.0 and rc.1 are `prerelease: true`; `safetensors/src/tensor.rs` is identical in v0.8.0 and v0.9.0-rc.1 (diff: 0 lines). Release page HEAD: 301 to `github.com/safetensors/safetensors/...`, then 200. |

## A. Sliding-window masks (transformers 18329f10, `src/transformers/masking_utils.py`)

Functions: `causal_mask_function` (72-76), `sliding_window_overlay` (88-97), `sliding_window_causal_mask_function` (130-134), `create_sliding_window_causal_mask` (1078-1209), intersection helper `and_masks` (44-55).

```python
72  def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
76      return kv_idx <= q_idx
88  def sliding_window_overlay(sliding_window: int) -> Callable:
94      def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
95          return kv_idx > q_idx - sliding_window
130 def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
134     return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)
```

Rule for window W: query position q sees key position kv iff `q - W < kv <= q`, that is kv in `[q-W+1, q]`. The current position is included; W-1 earlier positions are visible; W positions in total (fewer near the start). The docstring example for `sliding_window=3` (lines 449-466) has row 3 = `[False,  True,  True,  True, False]`.

Other backends agree. Flash path, `modeling_flash_attention_utils.py`:

```python
683     if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
684         # The flash attention API sets inclusive boundaries, i.e. (4, 0) would take 4 tokens to the left
685         # and the current token for a total size of 5. However, we usually define our window sizes by
688         flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)
```

The SDPA no-mask shortcut is off once the window bites (`masking_utils.py:265-266`: `if local_attention_size is not None and kv_length >= local_attention_size:` then `return False`). A bidirectional variant, used only when `config.is_causal` is false (1130-1139), is `return abs(q_idx - kv_idx) <= sliding_window` (line 145); not relevant to decoders.

Mistral mapping:

```python
configuration_mistral.py:80   sliding_window: int | None = 4096
configuration_utils.py:1073   config = cls(**config_dict)
modeling_mistral.py:353       mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
modeling_mistral.py:171       sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
masking_utils.py:1154-1156    sliding_window = getattr(config, "sliding_window", None)
                              if sliding_window is None:
                                  raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")
```

So: key absent in config.json gives 4096; explicit `null` gives None, which selects the plain causal mask (`create_causal_mask`); an integer W gives the sliding mask in every layer (no per-layer pattern: `configuration_mistral.py:88-91` only warns when `layer_types` is passed). Observed: Mistral-7B-v0.1 `"sliding_window": 4096`; Mistral-7B-v0.3 `"sliding_window": null`. Llama: `grep -n sliding configuration_llama.py` finds nothing (no field). Qwen3: `use_sliding_window: bool = False` (77), `sliding_window: int | None = 4096` (78), and line 87 `self.sliding_window = self.sliding_window if self.use_sliding_window else None`.

## B. tokenizers v0.23.3 (`tokenizers/src/...`)

**Byte-to-unicode table**, `pre_tokenizers/byte_level.rs:15-39` (`bytes_char`):

```rust
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');
    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }
```

I ran this rule (Python transcription): 188 bytes map to themselves (0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF); the other 68 map in byte order to 256+n: 0x00-0x20 to U+0100-U+0120 (0x0A to U+010A, 0x20 to U+0120), 0x7F-0xA0 to U+0121-U+0142, 0xAD to U+0143. SmolLM2 vocab agrees: `Ġ` (U+0120) id 216, `Ċ` (U+010A) id 198.

**GPT-2 regex**, lines 43-44, verbatim:

```rust
static RE: LazyLock<SysRegex> = LazyLock::new(|| {
    SysRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
```

Engine: `SysRegex` is Oniguruma through the `onig` crate (`utils/onig.rs:16-22`; `Cargo.toml:106` `default = ["progressbar", "onig", "esaxx_fast"]`; the Python bindings take default features, `bindings/python/Cargo.toml:27-28`). The Python lockfile resolves onig 6.5.3, onig_sys 69.9.3 (`bindings/python/Cargo.lock:721-736`); the rust-onig v6.5.3 submodule `onig_sys/oniguruma` points at kkos/oniguruma commit `94877ac5901c30cc7022f4f464da393b04cd3375` = Oniguruma 6.9.10 (`oniguruma.h:37-39`), Unicode 16.0.0 (`doc/UNICODE_PROPERTIES:1`). `Regex::new` uses `REGEX_OPTION_NONE` and `Syntax::default()` = `OnigDefaultSyntax` (rust-onig `syntax.rs:96-97`) = `ONIG_SYNTAX_ONIGURUMA` (`regparse.c:164`, options `ONIG_OPTION_NONE`, line 110). Documented `\s` in the Unicode case (`doc/RE:66-69`): U+0009, U+000A, U+000B, U+000C, U+000D, U+0085(NEL), and General_Category Line_Separator, Paragraph_Separator, Space_Separator. UNVERIFIED: no regex was executed here.

**ByteLevel flags** (`byte_level.rs:57-68`, default 74-82: all three true; a missing `use_regex` deserializes to true, tests 573-578):

```rust
123            if self.add_prefix_space && !normalized.get().starts_with(' ') {
124                normalized.prepend(" ");
126            if self.use_regex {
127                normalized.split(re_ref, SplitDelimiterBehavior::Isolated)
128            } else {
129                Ok(vec![normalized])
```

`use_regex: true` splits each incoming piece with the regex, `Isolated` (every match is its own piece). `use_regex: false` keeps the piece whole; then every UTF-8 byte is mapped through the table (132-146). Test, line 282: `use_regex(false)` on "Hello my friend, how is your day going?" gives one piece `ĠHelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?`. This runs per existing split, so `add_prefix_space: true` prepends a space to every piece that does not already start with one. Decoder (155-172): each token char goes through the inverse table; if any char is not in it, the token's raw UTF-8 bytes are used; all bytes are joined and `String::from_utf8_lossy` applied. `decoders/byte_level.rs` does not exist (HTTP 404); the decoder lives in the file above.

**Split / Sequence**: `Split` (`split.rs:96-104`) calls `normalized.split(&self.regex, self.behavior)`, or `Invert(&self.regex)` when `invert`; pattern is `String` (regex-escaped, line 83) or `Regex`. `Sequence` applies each pre-tokenizer in order to the same `PreTokenizedString` (`sequence.rs:39-46`). Behaviors (`tokenizer/normalizer.rs:74-80`, input `the-final--countdown`, delimiter `-`): Removed `[the, final, countdown]`; Isolated `[the, -, final, -, -, countdown]`; MergedWithPrevious `[the-, final-, -, countdown]`; MergedWithNext `[the, -final, -, -countdown]`; Contiguous `[the, -, final, --, countdown]`.

**Digits** (`digits.rs:29-37`):

```rust
        if self.individual_digits {
            pretokenized.split(|_, normalized| {
                normalized.split(char::is_numeric, SplitDelimiterBehavior::Isolated)
            })
        } else {
            pretokenized.split(|_, normalized| {
                normalized.split(char::is_numeric, SplitDelimiterBehavior::Contiguous)
```

`Digits::default()` is `individual_digits: false` (21-25). `individual_digits: true`: every numeric char is its own piece, text between stays as pieces. `false`: runs of numeric chars stay together (test: "Hey 123 friend!" gives `Hey `, `123`, ` friend!`). `char::is_numeric` is `'0'..='9' => true,` then `c => c > '\x7f' && unicode::N(c),` and its doc names the general categories `Nd`, `Nl`, `No` (local Rust 1.88.0 std, `char/methods.rs:965-969,1001-1006`; Unicode data 16.0.0 there; the wheels' toolchain file says `stable`, so unpinned), so it also splits non-ASCII digits and numerics such as "²" and "①".

**Metaspace** (`metaspace.rs`). Default replacement `'▁'` (U+2581), prepend `Always`, `split` true (118). `prepend_scheme` serde values `"first"`, `"never"`, `"always"` (5-14); missing gives `always` (44-46); missing `split` gives true (74); legacy `add_prefix_space: false` is accepted only with `never` (63-70).

```rust
125             normalized.replace(' ', &self.str_rep)?;
127                 PrependScheme::Always => {
128                     if !normalized.get().starts_with(self.replacement) {
132                 PrependScheme::First => {
133                     if !normalized.get().starts_with(self.replacement)
134                         && normalized.offsets_original().0 == 0
139                 PrependScheme::Never => {}
141             if self.split {
142                 normalized.split(self.replacement, SplitDelimiterBehavior::MergedWithNext)
```

Only ASCII space is replaced. `Always` prepends to every piece, `First` only to the piece that starts at original offset 0, in both cases only if the piece does not already start with the replacement. `split: false` leaves one piece. Decoder (150-173): replacement char becomes `' '`, except at token index 0 when `prepend_scheme != Never` (dropped).

**NFC**: `normalizers/unicode.rs:27-31` calls `normalized.nfc()`; `tokenizer/normalizer.rs:461-464`: `self.transform(self.get().to_owned().nfc(), 0)` using `unicode_normalization_alignments::UnicodeNormalization` (line 4); `Cargo.toml:82` `unicode-normalization-alignments = "0.1"`, locked to 0.1.12, the only published version (`bindings/python/Cargo.lock:1260-1263`; crates.io checksum equal). Its `src/tables.rs` (n1t0/unicode-normalization `9ca696060d7bedd591f903b6285c29abaab716fa`, Cargo.toml version 0.1.12, committed 2019-12-30T16:58:38Z, crates.io publish 17:10:56Z) has `pub const UNICODE_VERSION: (u64, u64, u64) = (9, 0, 0);`. So NFC here follows Unicode 9.0 tables, not current Unicode. UNVERIFIED: the published .crate tarball was not unpacked.

**Legacy normalizers used by Llama 2**: `Prepend` (`prepend.rs:19-21`) prepends only `if !normalized.is_empty()`; `Replace` with `String` pattern replaces every match of the regex-escaped string (`replace.rs:70,82-85`). Order in the JSON below: Prepend `▁`, then Replace `" "` with `▁`. A leading space therefore gives `▁▁`, unlike Metaspace `first`/`always` (derived from the code, not run).

**Byte fallback and unknowns**, `models/bpe/model.rs:466-550`. Lookup unit is each character of the pre-tokenized piece (`w.char_indices()`, 466):

```rust
494            if let Some(id) = self.vocab.get(s.as_ref()) {
501                if self.byte_fallback {
502                    let tokens: Option<Vec<_>> = s
503                        .bytes()
504                        .map(|b| -> Option<&u32> {
505                            let code = format!("<{b:#04X}>");
507                            self.vocab.get(&code)
510                    if let Some(tokens) = tokens {
511                        for t in tokens {
512                            word.add(*t, 1);
514                        continue;
517                if let Some(unk_token) = &self.unk_token {
518                    unk = match (unk, self.fuse_unk) {
519                        (Some((unk_id, unk_len)), true) => {
520                            // Fuse unk
521                            Some((unk_id, unk_len + byte_len))
```

If a character is not in the vocab and `byte_fallback` is true, all its UTF-8 bytes are looked up as `<0xNN>`; if all exist, they replace it (each with byte length 1). `{b:#04X}` verified by compiling a test: `<0x00>`, `<0x0A>`, `<0xFF>` (lowercase `0x`, two uppercase hex digits). If any byte token is missing, or fallback is off, `unk_token` is used (fused into one token when `fuse_unk`); with neither, the character is silently dropped. A `continuing_subword_prefix` or `end_of_word_suffix` would be inside `s` before the byte loop (481-492); both are null in the readable JSON. `merge_all` then runs on these ids (547). BPE details: rank is the index in `merges` (273); the new token `a+b` must exist in the vocab (270-272); smallest rank first, ties by leftmost position (`word.rs:28-35`); `ignore_merges` true returns the whole piece as one token if it is in the vocab (559-567); empty input gives no tokens (602-604). Merges may be `[["a","b"],...]` or legacy `["a b",...]` strings split on a single space, exactly two parts, `#version` lines dropped (`serialization.rs:85-91`; `model.rs:369-386`). Missing `byte_fallback` or `ignore_merges` means false (`serialization.rs:119-128`). Decoder `ByteFallback` (`decoders/byte_fallback.rs:30-47`): token of length 6 starting `<0x` and ending `>`, hex via `u8::from_str_radix(&token[3..5], 16)`; a run of byte tokens is decoded with `String::from_utf8`, else one `�` per byte token. `Fuse` joins tokens (`fuse.rs:26`); `Strip{content,start,stop}` removes up to `start` leading and `stop` trailing `content` chars per token (`strip.rs:27-60`); `Replace` decoder substitutes matches (`replace.rs:88-106`).

## C. safetensors v0.8.0 (`safetensors/src/tensor.rs`; format text `README.md:74-104`)

```rust
10  const MAX_HEADER_SIZE: usize = 100_000_000;
11  const N_LEN: usize = size_of::<u64>();
227     // Make sure we're sorting by descending dtype alignment
228     // Then by name
230     data.sort_by(|(lname, left), (rname, right)| {
231         right.dtype().cmp(&left.dtype()).then(lname.cmp(rname))
240         let tensor_info = TensorInfo {
241             dtype: tensor.dtype(),
242             shape: tensor.shape().to_vec(),
243             data_offsets: (offset, offset + n),
244         };
245         offset += n;
251     let mut metadata_buf = serde_json::to_string(&metadata)?.into_bytes();
253     // Force alignment to 8 bytes.
254     let aligned_metadata_len = metadata_buf.len().next_multiple_of(N_LEN);
255     metadata_buf.resize(aligned_metadata_len, b' ');
291     buffer.extend(n.to_le_bytes());
292     buffer.extend(header_bytes);
398         let n: usize = u64::from_le_bytes(arr)
```

- **Layout**: 8-byte little-endian u64 `n` (= padded header length), then `n` bytes of compact JSON (`serde_json::to_string`), padded with `0x20` to a multiple of 8, then the data. No per-tensor padding (offsets are contiguous). Alignment comes from the 8-byte header multiple plus the sort order (largest alignment first); that inference is mine. Test `test_serialization_forced_alignement` (1161-1185): a 65-byte JSON gets 7 bytes of `32` and header length 72.
- **Order**: tensors sorted by dtype descending in declaration order (below), then name ascending (`String` order); header entries and data follow that order; `__metadata__` is written first (`Serialize for Metadata`, 574-597). Reading re-sorts by `data_offsets` (557) and `validate` (627-664) requires first start 0, each start equal to the previous end, `end >= start`, `end - start == numel * bitsize / 8`, bits divisible by 8; the last end plus 8 plus `n` must equal the file length (420).
- **Entry JSON**: `{"name":{"dtype":"F32","shape":[1,2,3],"data_offsets":[0,24]}}`, field order dtype, shape, data_offsets (798-806). Test vector (1066-1088): header length bytes `64,0,0,0,0,0,0,0`, JSON `{"attn.0":{"dtype":"F32","shape":[1,2,3],"data_offsets":[0,24]}}` (64 bytes, no padding).
- **`__metadata__`**: optional (`HashMetadata`, 539-546: `#[serde(rename = "__metadata__")] metadata: Option<HashMap<String, String>>`, then `#[serde(flatten)]` tensors). README 83: "A special key `__metadata__` is allowed to contain free form string-to-string map. Arbitrary JSON is not allowed, all values must be strings." `None` omits the key; `Some(empty)` writes `"__metadata__":{}` (derived from lines 584-589, not run). Keys inside it come from a std `HashMap` (`tensor.rs:533`, `lib.rs:29`), so their order is not deterministic when there are several.
- **Read checks**: fewer than 8 bytes gives `HeaderTooSmall`; `n > MAX_HEADER_SIZE` gives `HeaderTooLarge`; header must be UTF-8 JSON. README 78-79 says the header "MUST begin with a `{`" and "MAY be trailing padded with whitespace (0x20)", but `read_metadata` has no `{` check and the test at 1541-1549 says leading whitespace is accepted ("This is intentional").
- **Dtype strings**, serde names equal the variant names, declaration order = alignment order (808-863, `Display` 905-928), bit sizes in brackets: `BOOL`[8], `F4`[4], `F6_E2M3`[6], `F6_E3M2`[6], `U8`, `I8`, `F8_E5M2`, `F8_E4M3`, `F8_E8M0`, `F8_E4M3FNUZ`, `F8_E5M2FNUZ` [8 each], `I16`, `U16`, `F16`, `BF16` [16], `I32`, `U32`, `F32` [32], `C64`, `F64`, `I64`, `U64` [64]. Row-major, little-endian (796-797).

## D. Tokenizer heads (first 32768 bytes, HTTP 206 each) and config.json

| Repo @ full revision | tokenizer.json bytes | added_tokens in head | config vocab_size | sliding_window / tie / rope_theta |
|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-135M @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 2104556 | 17, complete, ids 0-16, all special, normalized false | 49152 | absent / true / 100000 |
| NousResearch/Llama-2-7b-hf @ `8efe6c9b93655b934e27bd9981e3ec13e55aee9d` | 1842764 | 3, complete (`<unk>`,`<s>`,`</s>`), normalized true | 32000 | absent / false / absent |
| NousResearch/Meta-Llama-3-8B @ `315b20096dc791d381d514deb5f8bd9c8d6d3061` | 9085698 | CUT OFF: 158 complete entries (128000-128157, last content `reserved_special_token_152`), entry 128158 partial | 128256 | absent / false / 500000.0 |
| unsloth/Llama-3.2-1B @ `9535bd9b1d1dea6acafbdc4813b728796aeb28da` | 17209920 (LFS) | CUT OFF: 158 complete entries (128000-128157, last content `reserved_special_token_149`), entry 128158 partial | 128256 | absent / true / 500000.0 |
| mistralai/Mistral-7B-v0.1 @ `27d67f1b5f57dc0953326b2601d68371d40ea8da` | 1795188 | 3, complete, normalized false | 32000 | 4096 / false / 10000.0 |
| mistralai/Mistral-7B-v0.3 @ `caa1feb0e54d415e2df31207e5f4e273e33509b1` | 1961548 | CUT OFF: 176 complete (ids 0-175, last `[control_173]`) | 32768 | null / false / 1000000.0 |

Short ids from the brief resolve to these full ids via `huggingface.co/api/models/<repo>/revision/<short>`; each equals the current `main` head. Sizes come from `.../api/models/<repo>/tree/<sha>`.

Sections below are the parsed objects re-serialized compactly (key order kept); `raw_decode` started at each top-level key. Complete in head for SmolLM2, Llama-2, Mistral v0.1. **Cut off for Meta-Llama-3-8B, unsloth/Llama-3.2-1B and Mistral-7B-v0.3: the head ends inside `added_tokens`, so `normalizer`, `pre_tokenizer`, `post_processor`, `decoder` and `model` are not present (UNVERIFIED).**

SmolLM2-135M:

```json
"normalizer": null
"pre_tokenizer": {"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}
"post_processor": null
"decoder": {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}
"model": {"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":...}
```

Llama-2-7b-hf:

```json
"normalizer": {"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]}
"pre_tokenizer": null
"post_processor": {"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<s>","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}}}
"decoder": {"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]}
"model": {"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,"vocab":...}   (key "ignore_merges" ABSENT)
```

Mistral-7B-v0.1: `post_processor` and `decoder` are identical to Llama-2's (checked by equality); the rest:

```json
"normalizer": null
"pre_tokenizer": {"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":false}
"model": {"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,"ignore_merges":false,"vocab":...}
```

Vocab evidence in the heads: Llama-2 (1526 visible entries) and Mistral v0.1 (1528) have `<0x00>`..`<0xFF>` at ids 3..258 (256 contiguous, uppercase hex), `<0x0A>` = 13; 521 and 524 visible tokens contain `▁`, none contain `Ġ`. SmolLM2 (1483 visible) has no `<0xNN>` tokens and 622 tokens containing `Ġ`.

## E. Family classification (from the JSON above)

| Tokenizer | Result |
|---|---|
| Llama 2 | SentencePiece-style BPE, not byte-level: normalizer Prepend `▁` + Replace `" "` to `▁`, `pre_tokenizer` null (no regex; whole text is one BPE piece), `byte_fallback` true, `fuse_unk` true, decoder Replace/ByteFallback/Fuse/Strip. |
| Mistral v0.1 | SentencePiece-style BPE with Metaspace (`first`, `split` false) instead of the normalizer; `byte_fallback` true, same decoder; no regex. |
| Llama 3 | UNVERIFIED from JSON: the head ends before the pipeline. Circumstantial only: the repo tree lists no `tokenizer.model`, while Llama-2 and Mistral repos do; vocab_size 128256. |
| SmolLM2 (extra) | Byte-level BPE: Digits(individual) then ByteLevel with the GPT-2 regex; `byte_fallback` false. |

## F. Gaps and a follow-up that needs approval

The brief limits tokenizer.json to the first 32768 bytes, so I did not read further. For the two Llama-3 files, the head layout is regular (my layout model matched all 158 visible entries with 0 mismatches), so if `added_tokens` ends at id 128255 with consecutive `<|reserved_special_token_N|>` names (assumption), the key `"normalizer"` starts near byte 52899 (Meta-Llama-3-8B) and 52858 (unsloth Llama-3.2-1B). One extra window such as `curl -sSL --fail --max-filesize 20000 -r 52000-61999 https://huggingface.co/NousResearch/Meta-Llama-3-8B/resolve/315b20096dc791d381d514deb5f8bd9c8d6d3061/tokenizer.json` would show the pipeline and model flags (check `ignore_merges`). For Mistral-7B-v0.3 the array length is not visible, so no offset can be estimated. Also unchecked: whether unsloth Llama-3.2-1B stores merges as pairs (size 17.2 MB vs 9.1 MB suggests a different form; both forms are accepted by the deserializer); regex behavior was not executed; Rust std Unicode version for the shipped wheels.
