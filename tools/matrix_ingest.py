"""MATRIX ingest utilities for π-LM.

Provides:
- π-LM symbol tokenizer (drop-in replacement for placeholder mapping)
- ATOM header format (versioned, hashable, aligned)
- SVG-Tensor packing (atom grid metadata + tokens)
- GGUF embedding ingestion (minimal reader for embedding tensors)
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------
# π-LM symbol tokenizer
# ---------------------------

@dataclasses.dataclass(frozen=True)
class SymbolToken:
    token_id: int
    symbol: str


class PiSymbolTokenizer:
    """Deterministic symbol tokenizer with byte-fallback.

    Strategy:
    - Prefer longest-match on known symbols.
    - Fallback to UTF-8 bytes for any other content.
    - Fixed token layout for stable binary streams.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, symbols: Sequence[str], vocab_size: int = 65536) -> None:
        if vocab_size < 1024:
            raise ValueError("vocab_size must be >= 1024")
        self.vocab_size = vocab_size
        self._byte_base = 16
        self._symbol_base = self._byte_base + 256
        self._symbols = list(dict.fromkeys(symbols))
        self._symbol_trie = self._build_trie(self._symbols)
        self._symbol_to_id = {
            symbol: self._symbol_base + idx for idx, symbol in enumerate(self._symbols)
        }
        self._id_to_symbol = {
            token_id: symbol for symbol, token_id in self._symbol_to_id.items()
        }
        max_id = self._symbol_base + len(self._symbols)
        if max_id >= vocab_size:
            raise ValueError("symbol table exceeds vocab size")

    @staticmethod
    def default_symbols() -> List[str]:
        return [
            "\r\n",
            "\n",
            "\t",
            "  ",
            "```",
            "---",
            "===",
            "::",
            "->",
            "=>",
            "</",
            "/>",
            "<",
            ">",
            "==",
            "!=",
            "<=",
            ">=",
            "//",
            "#",
            "*",
            "_",
        ]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.BOS)
        idx = 0
        while idx < len(text):
            match = self._longest_symbol_match(text, idx)
            if match:
                symbol, length = match
                tokens.append(self._symbol_to_id[symbol])
                idx += length
                continue
            char = text[idx]
            byte_tokens = self._encode_bytes(char)
            tokens.extend(byte_tokens)
            idx += 1
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        out: List[str] = []
        for token in tokens:
            if token in (self.PAD, self.BOS, self.EOS):
                continue
            if token == self.UNK:
                out.append("\uFFFD")
                continue
            if token in self._id_to_symbol:
                out.append(self._id_to_symbol[token])
                continue
            if token >= self._byte_base and token < self._byte_base + 256:
                out.append(bytes([token - self._byte_base]).decode("utf-8", errors="replace"))
                continue
            out.append("\uFFFD")
        return "".join(out)

    def _encode_bytes(self, char: str) -> List[int]:
        bytes_value = char.encode("utf-8", errors="replace")
        return [self._byte_base + b for b in bytes_value]

    def _longest_symbol_match(self, text: str, start: int) -> Optional[Tuple[str, int]]:
        node = self._symbol_trie
        match: Optional[str] = None
        length = 0
        idx = start
        while idx < len(text) and text[idx] in node:
            node = node[text[idx]]
            idx += 1
            if "" in node:
                match = node[""]
                length = idx - start
        if match:
            return match, length
        return None

    @staticmethod
    def _build_trie(symbols: Sequence[str]) -> Dict[str, Dict]:
        root: Dict[str, Dict] = {}
        for symbol in symbols:
            node = root
            for char in symbol:
                node = node.setdefault(char, {})
            node[""] = symbol
        return root


# ---------------------------
# ATOM header format
# ---------------------------

@dataclasses.dataclass(frozen=True)
class AtomHeader:
    """Aligned header for MATRIX atom streams."""

    magic: bytes = b"MATR"
    version: int = 1
    header_bytes: int = 128
    atom_size: int = 256
    vocab_size: int = 65536
    dtype_code: int = 1  # 1: uint16, 2: uint32
    flags: int = 0
    payload_hash: bytes = b"" * 32

    def pack(self) -> bytes:
        payload = struct.pack(
            "<4sHHIIII32s",
            self.magic,
            self.version,
            self.header_bytes,
            self.atom_size,
            self.vocab_size,
            self.dtype_code,
            self.flags,
            self.payload_hash.ljust(32, b"\x00"),
        )
        if len(payload) > self.header_bytes:
            raise ValueError("header_bytes too small for packed header")
        padding = b"\x00" * (self.header_bytes - len(payload))
        return payload + padding


def compute_payload_hash(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


# ---------------------------
# SVG-Tensor packing
# ---------------------------

@dataclasses.dataclass(frozen=True)
class SvgTensorAtom:
    width: int
    height: int
    tokens: np.ndarray

    def pack(self, dtype: np.dtype) -> bytes:
        if self.tokens.size != self.width * self.height:
            raise ValueError("tokens size does not match width*height")
        header = struct.pack("<HHI", self.width, self.height, self.tokens.size)
        return header + self.tokens.astype(dtype).tobytes(order="C")


def pack_svg_tensor_atoms(
    atoms: Sequence[SvgTensorAtom],
    out_path: Path,
    header: AtomHeader,
    dtype: np.dtype,
) -> None:
    payload = b"".join(atom.pack(dtype) for atom in atoms)
    payload_hash = compute_payload_hash(payload)
    header_with_hash = dataclasses.replace(header, payload_hash=payload_hash)
    out_path.write_bytes(header_with_hash.pack() + payload)


# ---------------------------
# GGUF embedding ingestion
# ---------------------------

_GGUF_MAGIC = b"GGUF"


class GgufReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = path.read_bytes()
        self._buf = io.BytesIO(self._data)

    def _read(self, fmt: str) -> Tuple:
        size = struct.calcsize(fmt)
        data = self._buf.read(size)
        if len(data) != size:
            raise ValueError("unexpected EOF")
        return struct.unpack(fmt, data)

    def _read_string(self) -> str:
        (length,) = self._read("<I")
        value = self._buf.read(length)
        return value.decode("utf-8")

    def _read_array(self, value_type: int) -> List:
        (length,) = self._read("<I")
        return [self._read_value(value_type) for _ in range(length)]

    def _read_value(self, value_type: int):
        if value_type == 0:
            return self._read("<B")[0]
        if value_type == 1:
            return self._read("<b")[0]
        if value_type == 2:
            return self._read("<H")[0]
        if value_type == 3:
            return self._read("<h")[0]
        if value_type == 4:
            return self._read("<I")[0]
        if value_type == 5:
            return self._read("<i")[0]
        if value_type == 6:
            return self._read("<Q")[0]
        if value_type == 7:
            return self._read("<q")[0]
        if value_type == 8:
            return self._read("<f")[0]
        if value_type == 9:
            return self._read("<d")[0]
        if value_type == 10:
            return bool(self._read("<B")[0])
        if value_type == 11:
            return self._read_string()
        if value_type == 12:
            (inner_type,) = self._read("<I")
            return self._read_array(inner_type)
        raise ValueError(f"unsupported GGUF value type {value_type}")

    def read_header(self) -> Dict:
        magic = self._buf.read(4)
        if magic != _GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        (version,) = self._read("<I")
        (tensor_count,) = self._read("<Q")
        (kv_count,) = self._read("<Q")
        metadata = {}
        for _ in range(kv_count):
            key = self._read_string()
            (value_type,) = self._read("<I")
            metadata[key] = self._read_value(value_type)
        tensors = []
        for _ in range(tensor_count):
            name = self._read_string()
            (n_dims,) = self._read("<I")
            dims = [self._read("<Q")[0] for _ in range(n_dims)]
            (ggml_type,) = self._read("<I")
            (offset,) = self._read("<Q")
            tensors.append(
                {"name": name, "dims": dims, "type": ggml_type, "offset": offset}
            )
        data_offset = self._buf.tell()
        return {
            "version": version,
            "metadata": metadata,
            "tensors": tensors,
            "data_offset": data_offset,
        }

    def load_tensor(self, tensor_info: Dict) -> np.ndarray:
        ggml_type = tensor_info["type"]
        dtype = gguf_dtype_to_numpy(ggml_type)
        dims = tensor_info["dims"]
        offset = tensor_info["offset"]
        base = tensor_info.get("data_offset")
        if base is None:
            base = self.read_header()["data_offset"]
        byte_offset = base + offset
        count = int(np.prod(dims))
        byte_count = count * np.dtype(dtype).itemsize
        raw = self._data[byte_offset : byte_offset + byte_count]
        return np.frombuffer(raw, dtype=dtype).reshape(dims)


def gguf_dtype_to_numpy(ggml_type: int) -> np.dtype:
    mapping = {
        0: np.float32,
        1: np.float16,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.uint8,
        7: np.uint16,
        8: np.uint32,
        9: np.uint64,
    }
    if ggml_type not in mapping:
        raise ValueError(f"Unsupported GGML type {ggml_type}")
    return mapping[ggml_type]


def ingest_gguf_embeddings(
    gguf_path: Path,
    tensor_name: str,
    out_path: Path,
) -> None:
    reader = GgufReader(gguf_path)
    header = reader.read_header()
    tensor_info = next(
        (tensor for tensor in header["tensors"] if tensor["name"] == tensor_name),
        None,
    )
    if tensor_info is None:
        raise ValueError(f"tensor {tensor_name} not found")
    tensor_info["data_offset"] = header["data_offset"]
    data = reader.load_tensor(tensor_info)
    payload = data.astype(np.float32).tobytes(order="C")
    out_path.write_bytes(payload)


# ---------------------------
# Directory packing workflow
# ---------------------------

@dataclasses.dataclass
class PackConfig:
    atom_size: int = 256
    vocab_size: int = 65536
    dtype: np.dtype = np.uint16


def load_and_clean(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".json":
        try:
            obj = json.loads(text)
            text = json.dumps(obj, separators=(",", ":"))
        except json.JSONDecodeError:
            pass
    text = text.replace("<", " ").replace(">", " ")
    return text


def iter_text_files(input_dir: Path) -> Iterator[Path]:
    for path in input_dir.rglob("*"):
        if path.suffix.lower() in (".txt", ".md", ".html", ".json"):
            yield path


def pack_directory_svg_tensor(
    input_dir: Path,
    out_file: Path,
    config: PackConfig,
    grid_width: int = 16,
) -> None:
    tokenizer = PiSymbolTokenizer(PiSymbolTokenizer.default_symbols(), config.vocab_size)
    tokens: List[int] = []
    for path in iter_text_files(input_dir):
        text = load_and_clean(path)
        tokens.extend(tokenizer.encode(text))

    pad = (-len(tokens)) % config.atom_size
    if pad:
        tokens.extend([PiSymbolTokenizer.PAD] * pad)

    arr = np.array(tokens, dtype=config.dtype)
    atoms: List[SvgTensorAtom] = []
    for i in range(0, len(arr), config.atom_size):
        atom_tokens = arr[i : i + config.atom_size]
        height = config.atom_size // grid_width
        if grid_width * height != config.atom_size:
            raise ValueError("grid width must divide atom size")
        atoms.append(
            SvgTensorAtom(width=grid_width, height=height, tokens=atom_tokens)
        )

    header = AtomHeader(
        atom_size=config.atom_size,
        vocab_size=config.vocab_size,
        dtype_code=1 if config.dtype == np.uint16 else 2,
    )
    pack_svg_tensor_atoms(atoms, out_file, header, config.dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pack directory into MATRIX atoms")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("out_file", type=Path)
    parser.add_argument("--atom-size", type=int, default=256)
    parser.add_argument("--grid-width", type=int, default=16)
    args = parser.parse_args()

    cfg = PackConfig(atom_size=args.atom_size)
    pack_directory_svg_tensor(args.input_dir, args.out_file, cfg, args.grid_width)
    print(f"[OK] Packed MATRIX atoms to {args.out_file}")
