# MATRIX Ingest Spec v1

## Purpose

MATRIX Ingest v1 defines a binary-first pipeline for π-LM ingestion. It standardizes:

- **π-LM symbol tokenization** (deterministic, byte-safe)
- **ATOM headers** (versioned, hashable, aligned)
- **SVG-Tensor atom packing** (grid metadata + token payload)
- **GGUF embedding compatibility** (optional embeddings export)

This spec is optimized for sequential reads, mmap access, and low-overhead decoding.

## 1) Pipeline Overview

```
[ HTML | JSON | MD ]
        ↓ (one-time)
  CLEAN + NORMALIZE
        ↓
 TOKENIZE (π symbol rules)
        ↓
  PACK → ATOMIC-DOM
        ↓
   mmap / seek / stream
        ↓
   π-LM / Embedding / Geometry
```

### Key rules

- **No parsing in the hot loop**: text decoding happens once, offline.
- **Fixed-width tokens**: tokens are uint16 or uint32.
- **Sequential layout**: atom payloads are contiguous and aligned.

## 2) π-LM Symbol Tokenizer

### Token classes

| Token ID | Meaning |
| --- | --- |
| 0 | PAD |
| 1 | BOS |
| 2 | EOS |
| 3 | UNK |
| 16–271 | Raw UTF-8 bytes (byte + 16) |
| 272+ | Symbol table entries |

### Symbol selection

Symbols are longest-match, greedy. A default starter set:

- `\r\n`, `\n`, `\t`, `  `
- Markdown operators: `` ``` ``, `---`, `===`, `#`, `*`, `_`
- Structural tokens: `</`, `/>`, `<`, `>`
- Operators: `::`, `->`, `=>`, `==`, `!=`, `<=`, `>=`, `//`

### Determinism

When a symbol match is not found, input is encoded as UTF-8 bytes. This guarantees:

- byte stability
- reversible decoding
- fixed token ID allocation

## 3) ATOM Header Format

ATOM headers are fixed-size (aligned) and precede the payload.

### Header layout (little-endian)

```
Offset  Size  Field
0x00    4     magic = "MATR"
0x04    2     version (uint16)
0x06    2     header_bytes (uint16)
0x08    4     atom_size (uint32)
0x0C    4     vocab_size (uint32)
0x10    4     dtype_code (uint32)   # 1=uint16, 2=uint32
0x14    4     flags (uint32)
0x18    32    payload_hash (sha256)
0x38    ...   padding to header_bytes
```

### Alignment

- `header_bytes` **MUST** be >= 64.
- Recommended: 128 bytes for cache-line alignment.

### Hashing

`payload_hash` is the SHA-256 of the payload bytes following the header.

## 4) SVG-Tensor Atom Packing

Each atom is stored as a small SVG-Tensor record containing the grid size and payload.

### Record layout

```
Offset  Size  Field
0x00    2     width (uint16)
0x02    2     height (uint16)
0x04    4     token_count (uint32)
0x08    ...   tokens (dtype * token_count)
```

### Grid rule

- `width * height = atom_size`
- Layout is row-major (left-to-right, top-to-bottom).

### Benefits

- Spatial layout metadata enables geometry-aware renderers.
- Atom reads remain contiguous and cache-friendly.

## 5) GGUF Embedding Compatibility

GGUF tensors can be ingested via a minimal reader that exports a named tensor to a
raw float32 payload. This is useful for:

- seeding embeddings
- cross-referencing GGUF tokens with MATRIX token streams

## 6) Reference Implementation

See:

- `tools/matrix_ingest.py` for the tokenizer, header, SVG-Tensor packer, and GGUF reader.

