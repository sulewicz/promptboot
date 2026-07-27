# PBTQW25 v1 frozen container format

PBTQW25 v1 is the fixed container format used by promptboot. PBTQW25 is not a general model format: it is the little-endian runtime image for exactly the checksum-pinned `Qwen2.5-0.5B-Instruct` GGUF described in `assets.lock.json`: 24 Qwen2 decoder blocks, hidden width 896, feed-forward width 4864, 14 query heads, two KV heads, 151,936 tokens, and the model's native 32,768-token context. A reader must reject every other version, architecture, shape, count, role, dtype, order, or identity.

All offsets and lengths are unsigned, checked before arithmetic, absolute from the start of the container, and little-endian. SHA-256 fields store the 32 digest bytes in their usual digest order. The full-file SHA-256 is external because an embedded full-file digest would be circular.

## Header

The header is exactly 256 bytes.

| Offset | Type | Field | Required value |
| ---: | --- | --- | --- |
| `0x00` | `char[8]` | magic | `PBTQW25` followed by NUL |
| `0x08` | `u32` | version | 1 |
| `0x0c` | `u32` | header bytes | 256 |
| `0x10` | `u32` | endian tag | `0x01020304` |
| `0x14` | `u32` | section count | 7 |
| `0x18` | `u32` | tensor count | 291 |
| `0x1c` | `u32` | vocabulary count | 151936 |
| `0x20` | `u32` | merge count | 151387 |
| `0x24` | `u32` | runtime context limit | 32768 |
| `0x28` | `u32` | block count | 24 |
| `0x2c` | `u32` | embedding width | 896 |
| `0x30` | `u32` | feed-forward width | 4864 |
| `0x34` | `u32` | query heads | 14 |
| `0x38` | `u32` | KV heads | 2 |
| `0x3c` | `u32` | head dimension | 64 |
| `0x40` | `u32` | EOS token | 151645 |
| `0x44` | `u32` | BOS token | 151643 |
| `0x48` | `u32` | padding token | 151643 |
| `0x4c` | `u32` | add BOS | 0 |
| `0x50` | `u64` | source bytes | 428730208 |
| `0x58` | `u64` | container bytes | exact file length |
| `0x60` | `u64` | section-directory offset | 256 |
| `0x68` | `u64` | section-directory bytes | 448 |
| `0x70` | `u64` | source tensor-data offset | 5947744 |
| `0x78` | `u64` | source tensor bytes | 422782464 |
| `0x80` | `u8[32]` | source SHA-256 | `7671c0c3…edaf6ed` |
| `0xa0` | `u8[32]` | content SHA-256 | exact bytes `[256,file_size)` |
| `0xc0` | `u8[64]` | reserved | all zero |

## Section directory

Seven fixed 64-byte entries begin at offset 256. An entry contains `u32 section_id` at `+0x00`, `u32 element_type` at `+0x04`, `u64 count` at `+0x08`, `u64 offset` at `+0x10`, `u64 byte_length` at `+0x18`, and the section SHA-256 at `+0x20`. Element types are 1=`U32LE`, 2=`U8`, 3=`U32LE_TRIPLE`, and 4=`TENSOR_ENTRY_V1`.

| ID | Section | Element type | Count | Byte length |
| ---: | --- | ---: | ---: | ---: |
| 1 | token offsets | 1 | 151937 | 607748 |
| 2 | concatenated token UTF-8 | 2 | 1372758 | 1372758 |
| 3 | token types | 2 | 151936 | 151936 |
| 4 | merge rules | 3 | 151387 | 1816644 |
| 5 | source chat template | 2 | 2509 | 2509 |
| 6 | tensor directory | 4 | 291 | 27936 |
| 7 | tensor data | 2 | byte length | includes internal zero alignment padding |

Sections occur in this exact order. Every section begins at a 64-byte boundary, sections never overlap, every gap is zero, every section hash covers exactly its declared bytes, and the final section ends at the file length.

## Tensor entries and payload

Each 96-byte tensor entry contains: `u32 tensor_id` at `+0x00`; `u16 layer` at `+0x04`; `u16 role` at `+0x06`; `u32 dtype` at `+0x08`; `u32 rank` at `+0x0c`; four `u32` dimensions at `+0x10`; `u64 data_offset` at `+0x20`; `u64 data_length` at `+0x28`; raw-tensor SHA-256 at `+0x30`; and 16 zero reserved bytes at `+0x50`.

Dimensions preserve GGUF `ne` order, with `ne0` fastest-changing and unused dimensions equal to 1. Global tensors use layer `0xffff`; decoder layers are 0 through 23. Dtypes are 1=F32, 2=legacy Q4_0, and 3=legacy Q8_0. Checked byte lengths are `elements*4`, `elements/32*18`, and `elements/32*34`, respectively, and both quantized element counts are divisible by 32.

Roles are 1 token embedding, 2 output norm, 3 output, 10 attention norm, 11 FFN down, 12 FFN gate, 13 FFN up, 14 FFN norm, 15 K bias, 16 K weight, 17 attention output, 18 Q bias, 19 Q weight, 20 V bias, and 21 V weight. ID 0 is `token_embd.weight`; IDs 1–288 are layers 0–23 with roles 10–21 in that order; ID 289 is `output_norm.weight`; ID 290 is `output.weight`.

Each tensor begins at a 64-byte boundary. Inter-tensor bytes are zero. Payload bytes are copied unchanged from the immutable GGUF: no transpose, repack, conversion, or requantization occurs.

## Tokenizer contract

Token offsets are monotonic `u32` values beginning at zero and ending at 1,372,758. Token bytes are exact source UTF-8. Token types are the source I32 values narrowed only after proving each fits `u8`. Each merge is `(left_id,right_id,result_id)` in source-rank order; all IDs are in range, the pair is unique, and result bytes are exactly left bytes followed by right bytes.

The source canonical digests use, for every token or merge string, a little-endian `u64` byte length followed by exact UTF-8, without a leading array count. Token types use each source I32 value represented as little-endian `u64`. The expected digests are:

- tokens: `8656e079473b857729cf444f772368cd6428dcd64513d46aac3f694b2f695282`
- merges: `907981a313a6e78ef2223229042674a39cd4995ee4c4ed40a31b58754e82c26c`
- token types: `f3760e7fbfc96d388f5b8d7cd67c5cd46535c8682ff84ebc9817cda61bc98f4d`
- chat template: `d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4`

The exact Qwen2 pre-tokenization expression is:

```text
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

GPT-2 byte-to-Unicode preserves bytes `0x21..0x7e`, `0xa1..0xac`, and `0xae..0xff`; every remaining byte in ascending order maps to `U+0100+n`. Decode applies the inverse and errors on an unmapped code point. BPE starts with byte-token IDs, repeatedly selects the lowest merge rank, breaks equal-rank occurrences leftmost, and replaces the pair with its recorded result until none applies.

The only renderer is:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{PRINTABLE_ASCII_USER}<|im_end|>
<|im_start|>assistant
```

Renderer markers are trusted IDs 151644 and 151645; user bytes are always ordinary text with special parsing disabled. BOS is not added. BOS/PAD alias 151643 and EOS is 151645. The deterministic inference oracle selects the lowest token ID on an exact logit tie; the interactive runtime uses the fixed sampling policy documented in the README and terminates on EOS. General Jinja and arbitrary special-token parsing are outside the format.

## Measured canonical packing

For the locked asset, the implementation emits 426,762,944 bytes, below the 469,762,048-byte ceiling. The canonical full-file SHA-256 is `b0f98ed6e0557ca35e1bced1000c950b3c84414251df65290315a7969981d42d`; the content SHA-256 is `7239e4ac4cde7b0e34427420a964d2df868773f35cb00a147162df4b29a87987`. See [development.md](development.md) for packing and inspection commands.
