```python
#!/usr/bin/env python3
"""
ASX Local Python Compiler v1.1
==============================

Compiles:
  brain.xml (SVG lane tensors) + optional data pack → results.json (lane-aware weights.v2)

Features (v1.1):
- SVG → lane scalar streams (geometry → scalars)
- Lane-specific knobs (temp.<lane>, top_p.<lane>, style.<lane>)
- Lane merge semantics (add/avg/ema/max/override)
- Proof hash per lane + pack hash (anti-drift)

No external deps. Deterministic. Portable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# VERSIONS / CONSTANTS
# ============================================================

COMPILER_VERSION = "ASX_LOCAL_COMPILER_v1.1.0"
WEIGHTS_VERSION = "weights.v2"
CANON = "ASX_CANON_V1"

DEFAULT_DIMS = 64

# Frozen mixing weights for scalar extraction
MIX_W_LEN = 0.55
MIX_W_TURN = 0.25
MIX_W_BBOX = 0.10
MIX_W_CENT = 0.10


# ============================================================
# HELPERS (deterministic math / rounding / hashing)
# ============================================================

def clamp01(x: float) -> float:
  return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def f6(x: float) -> float:
  # Stable 6-decimal rounding compatible with JS round6
  return float(Decimal(str(x)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def round_vec6(vec: List[float]) -> List[float]:
  return [f6(v) for v in vec]


def stable_hash32(s: str) -> int:
  # FNV-1a-like, deterministic across runs
  h = 2166136261
  for ch in s:
    h ^= ord(ch)
    h = (h + ((h << 1) & 0xFFFFFFFF) + ((h << 4) & 0xFFFFFFFF) + ((h << 7) & 0xFFFFFFFF)
         + ((h << 8) & 0xFFFFFFFF) + ((h << 24) & 0xFFFFFFFF)) & 0xFFFFFFFF
  return h


def canon_dumps(obj: Any) -> str:
  # Sorted keys, no whitespace (canonical string)
  return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
  return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ============================================================
# XML PARSING (brain.xml)
# ============================================================

@dataclass
class BrainDoc:
  brain_id: str
  brain_name: str
  svg_scale: float
  lanes: Dict[str, List[str]]   # lane_id -> list of path d strings
  threshold: float
  knobs: Dict[str, Dict[str, float]]  # knob -> { laneId or "*": value }


def _strip_ns(tag: str) -> str:
  # ElementTree includes namespace in tag like "{...}svg"
  return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_brain_xml(path: str) -> BrainDoc:
  if not os.path.exists(path):
    raise FileNotFoundError(f"brain.xml not found: {path}")

  tree = ET.parse(path)
  root = tree.getroot()

  if _strip_ns(root.tag) != "brain":
    raise ValueError("Root element must be <brain>")

  brain_id = root.attrib.get("id", "brain_unnamed")
  brain_name = root.attrib.get("name", brain_id)

  svg_el = None
  for child in root:
    if _strip_ns(child.tag) == "svg":
      svg_el = child
      break
  if svg_el is None:
    raise ValueError("Missing <svg> in brain.xml")

  scale = derive_svg_scale(svg_el)

  tensor_el = None
  for child in root:
    if _strip_ns(child.tag) == "tensor":
      tensor_el = child
      break
  if tensor_el is None:
    raise ValueError("Missing <tensor> in brain.xml")

  # Parse knobs (lane-aware), prefer <knobs><temp lane="x" value="0.7"/></knobs> style,
  # but also support attributes temp/top_p/style on <knobs>.
  knobs = parse_knobs(root)

  # Parse threshold
  threshold = 0.6
  for child in root:
    if _strip_ns(child.tag) == "rules":
      for r in child:
        if _strip_ns(r.tag) == "threshold":
          threshold = float(r.attrib.get("value", "0.6"))
      break

  lanes = parse_tensor_lanes(tensor_el)

  return BrainDoc(
    brain_id=brain_id,
    brain_name=brain_name,
    svg_scale=scale,
    lanes=lanes,
    threshold=threshold,
    knobs=knobs
  )


def derive_svg_scale(svg_el: ET.Element) -> float:
  vb = svg_el.attrib.get("viewBox") or svg_el.attrib.get("viewbox")
  if vb:
    parts = re.split(r"\s+", vb.strip())
    if len(parts) == 4:
      try:
        W = abs(float(parts[2]))
        H = abs(float(parts[3]))
        S = max(W, H)
        return S if S > 0 else 1.0
      except Exception:
        pass
  # fallback width/height
  def parse_len(s: Optional[str]) -> float:
    if not s:
      return 0.0
    # strip common units
    s = s.strip().replace("px", "")
    try:
      return abs(float(s))
    except Exception:
      return 0.0

  w = parse_len(svg_el.attrib.get("width"))
  h = parse_len(svg_el.attrib.get("height"))
  S = max(w, h)
  return S if S > 0 else 1.0


def parse_tensor_lanes(tensor_el: ET.Element) -> Dict[str, List[str]]:
  lanes: Dict[str, List[str]] = {}

  # Detect <lane> children
  lane_children = [c for c in tensor_el if _strip_ns(c.tag) == "lane"]

  if not lane_children:
    # backward compat: <tensor><path/></tensor> => policy lane
    ds: List[str] = []
    for p in tensor_el.iter():
      if _strip_ns(p.tag) == "path":
        d = p.attrib.get("d", "")
        if d.strip():
          ds.append(d.strip())
    if not ds:
      raise ValueError("No <path> found in <tensor>")
    lanes["policy"] = ds
    return lanes

  for lane_el in lane_children:
    lane_id = lane_el.attrib.get("id")
    if not lane_id:
      raise ValueError("<lane> missing id attribute")
    ds: List[str] = []
    for p in lane_el.iter():
      if _strip_ns(p.tag) == "path":
        d = p.attrib.get("d", "")
        if d.strip():
          ds.append(d.strip())
    if not ds:
      raise ValueError(f"Lane '{lane_id}' has no <path> d data")
    lanes[lane_id] = ds

  return lanes


def parse_knobs(root: ET.Element) -> Dict[str, Dict[str, float]]:
  """
  Canonical knobs format (recommended):

  <knobs>
    <temp lane="attention" value="0.78"/>
    <temp lane="policy" value="0.60"/>
    <temp lane="*" value="0.70"/>
    <top_p lane="*" value="0.90"/>
    <style lane="style" value="0.55"/>
  </knobs>

  Backward compat:
  <knobs temp="0.75" top_p="0.92" style="0.40"/>
  => applies to "*" lane for each knob.
  """
  knobs: Dict[str, Dict[str, float]] = {}

  knobs_el = None
  for child in root:
    if _strip_ns(child.tag) == "knobs":
      knobs_el = child
      break
  if knobs_el is None:
    return knobs

  # attribute compat
  for k in ("temp", "top_p", "style"):
    if k in knobs_el.attrib:
      knobs.setdefault(k, {})
      knobs[k]["*"] = clamp01(float(knobs_el.attrib.get(k, "0.0")))

  # child entries
  for child in knobs_el:
    name = _strip_ns(child.tag)  # temp/top_p/style
    if name not in ("temp", "top_p", "style"):
      continue
    lane = child.attrib.get("lane", "*")
    val = child.attrib.get("value")
    if val is None:
      continue
    knobs.setdefault(name, {})
    knobs[name][lane] = clamp01(float(val))

  return knobs


def resolve_knob(knobs: Dict[str, Dict[str, float]], name: str, lane_id: str, fallback: float) -> float:
  lane_map = knobs.get(name)
  if not lane_map:
    return fallback
  if lane_id in lane_map:
    return clamp01(float(lane_map[lane_id]))
  if "*" in lane_map:
    return clamp01(float(lane_map["*"]))
  return fallback


# ============================================================
# SVG PATH → LANE SCALAR STREAMS (geometry → scalars)
# ============================================================

_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def parse_numbers_from_path_d(d: str) -> List[float]:
  return [float(m.group(0)) for m in _NUM_RE.finditer(d)]


def nums_to_points(nums: List[float]) -> List[Tuple[float, float]]:
  pts: List[Tuple[float, float]] = []
  for i in range(0, len(nums) - 1, 2):
    pts.append((nums[i], nums[i + 1]))
  return pts


def path_to_scalar_stream(d: str, scale: float) -> List[float]:
  nums = parse_numbers_from_path_d(d)
  pts = nums_to_points(nums)
  if len(pts) < 2:
    return []

  # bbox + centroid
  min_x = max_x = pts[0][0]
  min_y = max_y = pts[0][1]
  sum_x = 0.0
  sum_y = 0.0
  for x, y in pts:
    min_x = min(min_x, x)
    max_x = max(max_x, x)
    min_y = min(min_y, y)
    max_y = max(max_y, y)
    sum_x += x
    sum_y += y
  cx = sum_x / len(pts)
  cy = sum_y / len(pts)

  bbox_w = (max_x - min_x) / scale
  bbox_h = (max_y - min_y) / scale
  bbox_n = clamp01((abs(bbox_w) + abs(bbox_h)) * 0.5)

  cent_n = clamp01((abs(cx / scale) + abs(cy / scale)) * 0.5)

  stream: List[float] = []
  prev_ang: Optional[float] = None

  for i in range(len(pts) - 1):
    x0, y0 = pts[i]
    x1, y1 = pts[i + 1]
    dx = (x1 - x0)
    dy = (y1 - y0)

    seg_len = math.sqrt(dx * dx + dy * dy)
    len_n = clamp01(seg_len / scale)

    ang = math.atan2(dy, dx)
    turn_n = 0.0
    if prev_ang is not None:
      da = abs(ang - prev_ang)
      if da > math.pi:
        da = (2 * math.pi) - da
      turn_n = clamp01(da / math.pi)
    prev_ang = ang

    s = clamp01(
      MIX_W_LEN * len_n +
      MIX_W_TURN * turn_n +
      MIX_W_BBOX * bbox_n +
      MIX_W_CENT * cent_n
    )
    stream.append(f6(s))

  return stream


def extract_lane_scalar_streams(brain: BrainDoc) -> Dict[str, List[float]]:
  out: Dict[str, List[float]] = {}
  for lane_id, paths in brain.lanes.items():
    scalars: List[float] = []
    for d in paths:
      scalars.extend(path_to_scalar_stream(d, brain.svg_scale))
    out[lane_id] = scalars
  return out


# ============================================================
# FOLDING (scalar stream → fixed dims tensor per lane)
# ============================================================

def fold_stream_to_dims(stream: List[float], dims: int) -> List[float]:
  vec = [0.0] * dims
  for i, v in enumerate(stream):
    vec[i % dims] += float(v)
  return [f6(x) for x in vec]


def fold_lanes_to_dims(lanes_streams: Dict[str, List[float]], dims: int) -> Dict[str, List[float]]:
  return {lane_id: fold_stream_to_dims(stream, dims) for lane_id, stream in lanes_streams.items()}


# ============================================================
# SYNTHESIS (folded tensors → lane weights)
# ============================================================

def synthesize_lane_weights(folded_lanes: Dict[str, List[float]], seed: str, amplitude: float) -> Dict[str, Dict[str, List[float]]]:
  seed_hash = stable_hash32(seed)
  out: Dict[str, Dict[str, List[float]]] = {}

  for lane_id, folded in folded_lanes.items():
    base: List[float] = []
    phase = (seed_hash % 997) * 0.0001
    for i, fv in enumerate(folded):
      w = math.sin(i * 0.37 + phase) * amplitude * (1.0 + float(fv))
      base.append(f6(w))
    out[lane_id] = {"W": base}

  return out


def apply_lane_knobs_to_weights(weights_pack: Dict[str, Any], knobs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
  out = json.loads(json.dumps(weights_pack))
  lanes = out.get("lanes", {})
  for lane_id, lane_obj in lanes.items():
    W = lane_obj.get("W")
    if not isinstance(W, list):
      continue

    temp = resolve_knob(knobs, "temp", lane_id, 0.70)
    top_p = resolve_knob(knobs, "top_p", lane_id, 0.90)
    style = resolve_knob(knobs, "style", lane_id, 0.40)

    mag = 0.65 + temp * 0.70          # 0.65..1.35
    damp = 1.25 - top_p * 0.55        # 1.25..0.70

    styleW = lane_obj.get("styleW") if isinstance(lane_obj.get("styleW"), list) else None

    newW: List[float] = []
    for i, w in enumerate(W):
      ww = float(w) * mag * damp
      if styleW is not None and i < len(styleW):
        ww = (1.0 - style) * ww + style * float(styleW[i])
      newW.append(f6(ww))

    lane_obj["W"] = newW

  return out


# ============================================================
# MERGE SEMANTICS (lane-aware pack merging)
# ============================================================

def merge_scalar(a: float, b: float, mode: str, alpha: float) -> float:
  a = float(a or 0.0)
  b = float(b or 0.0)
  if mode == "override":
    return f6(b)
  if mode == "add":
    return f6(a + b)
  if mode == "avg":
    return f6((a + b) * 0.5)
  if mode == "max":
    return b if abs(b) >= abs(a) else a
  # ema default
  return f6((1.0 - alpha) * a + alpha * b)


def merge_vec(A: List[float], B: List[float], mode: str, alpha: float) -> List[float]:
  n = min(len(A), len(B))
  out: List[float] = []
  for i in range(n):
    a = float(A[i]); b = float(B[i])
    if mode == "override":
      v = b
    elif mode == "add":
      v = a + b
    elif mode == "avg":
      v = (a + b) * 0.5
    elif mode == "max":
      v = b if abs(b) >= abs(a) else a
    else:
      v = (1.0 - alpha) * a + alpha * b
    out.append(f6(v))
  return out


def zero_vec(n: int) -> List[float]:
  return [0.0] * n


def merge_packs(A: Dict[str, Any], B: Dict[str, Any], mode: str = "ema", alpha: float = 0.15) -> Dict[str, Any]:
  if int(A.get("dims")) != int(B.get("dims")):
    raise ValueError("dims mismatch")
  dims = int(A["dims"])

  out: Dict[str, Any] = {
    "version": WEIGHTS_VERSION,
    "dims": dims,
    "bias": merge_scalar(A.get("bias", 0.0), B.get("bias", 0.0), mode, alpha),
    "lanes": {},
    "meta": {
      "law": "lane-aware",
      "compiler": COMPILER_VERSION,
      "merged_from": [A.get("meta", {}).get("source", "A"), B.get("meta", {}).get("source", "B")],
      "merge": {"mode": mode, "alpha": alpha}
    }
  }

  lane_ids = set()
  lane_ids.update((A.get("lanes") or {}).keys())
  lane_ids.update((B.get("lanes") or {}).keys())

  for lane_id in sorted(lane_ids):
    WA = (A.get("lanes") or {}).get(lane_id, {}).get("W")
    WB = (B.get("lanes") or {}).get(lane_id, {}).get("W")
    if not isinstance(WA, list):
      WA = zero_vec(dims)
    if not isinstance(WB, list):
      WB = zero_vec(dims)
    out["lanes"][lane_id] = {"W": merge_vec(WA, WB, mode, alpha)}

  return out


# ============================================================
# PROOFS (per-lane + pack hash) — anti-drift
# ============================================================

def canon_lane_obj(pack: Dict[str, Any], lane_id: str) -> Dict[str, Any]:
  lane = pack["lanes"][lane_id]
  W = [f6(float(v)) for v in lane["W"]]
  return {"dims": int(pack["dims"]), "lane": lane_id, "version": pack["version"], "W": W}


def add_lane_proofs(pack: Dict[str, Any]) -> Dict[str, Any]:
  out = json.loads(json.dumps(pack))
  out.setdefault("proof", {})
  out["proof"]["algo"] = "sha256"
  out["proof"]["canonicalization"] = CANON
  out["proof"]["lanes"] = {}

  for lane_id in sorted((out.get("lanes") or {}).keys()):
    lane_obj = canon_lane_obj(out, lane_id)
    h = sha256_hex(canon_dumps(lane_obj))
    out["proof"]["lanes"][lane_id] = {"hash": h}

  bias = f6(float(out.get("bias", 0.0)))
  pack_obj = {
    "bias": bias,
    "dims": int(out["dims"]),
    "lanes": {k: out["proof"]["lanes"][k]["hash"] for k in sorted(out["proof"]["lanes"].keys())},
    "version": out["version"]
  }
  out["proof"]["pack_hash"] = sha256_hex(canon_dumps(pack_obj))
  return out


def verify_lane_proofs(pack: Dict[str, Any]) -> bool:
  proof = pack.get("proof") or {}
  if proof.get("algo") != "sha256":
    return False
  if proof.get("canonicalization") != CANON:
    return False
  lane_proofs = proof.get("lanes") or {}
  for lane_id in sorted((pack.get("lanes") or {}).keys()):
    expected = lane_proofs.get(lane_id, {}).get("hash")
    if not expected:
      return False
    lane_obj = canon_lane_obj(pack, lane_id)
    got = sha256_hex(canon_dumps(lane_obj))
    if got != expected:
      return False

  bias = f6(float(pack.get("bias", 0.0)))
  pack_obj = {
    "bias": bias,
    "dims": int(pack["dims"]),
    "lanes": {k: lane_proofs[k]["hash"] for k in sorted(lane_proofs.keys())},
    "version": pack["version"]
  }
  got_pack = sha256_hex(canon_dumps(pack_obj))
  return got_pack == proof.get("pack_hash")


# ============================================================
# COMPILATION PIPELINE
# ============================================================

def compile_from_brain(brain: BrainDoc, dims: int, seed: str, bias: float, amplitude: float,
                       knobs_override: Optional[Dict[str, Dict[str, float]]] = None,
                       meta_source: str = "local") -> Dict[str, Any]:
  # 1) SVG → scalar streams per lane
  lane_streams = extract_lane_scalar_streams(brain)

  # 2) Fold to dims per lane
  folded = fold_lanes_to_dims(lane_streams, dims)

  # 3) Synthesize lane weights
  lanes = synthesize_lane_weights(folded, seed=seed, amplitude=amplitude)

  # 4) Build pack
  pack: Dict[str, Any] = {
    "version": WEIGHTS_VERSION,
    "dims": dims,
    "bias": f6(bias),
    "lanes": lanes,
    "knobs": brain.knobs,  # embed knobs used/available
    "meta": {
      "law": "lane-aware",
      "compiler": COMPILER_VERSION,
      "source": meta_source,
      "brain_id": brain.brain_id,
      "brain_name": brain.brain_name
    }
  }

  # 5) Apply knobs (brain + optional override)
  knobs = brain.knobs
  if knobs_override:
    knobs = merge_knobs(knobs, knobs_override)
    pack["knobs"] = knobs

  pack = apply_lane_knobs_to_weights(pack, knobs)

  # 6) Proofs
  pack = add_lane_proofs(pack)
  return pack


def merge_knobs(base: Dict[str, Dict[str, float]], override: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
  out = json.loads(json.dumps(base or {}))
  for knob_name, lane_map in (override or {}).items():
    out.setdefault(knob_name, {})
    for lane_id, v in lane_map.items():
      out[knob_name][lane_id] = clamp01(float(v))
  return out


# ============================================================
# CLI
# ============================================================

def load_json_file(path: str) -> Dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def save_json_file(path: str, obj: Any) -> None:
  with open(path, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_knobs_json_arg(s: Optional[str]) -> Optional[Dict[str, Dict[str, float]]]:
  if not s:
    return None
  # expects JSON string
  obj = json.loads(s)
  if not isinstance(obj, dict):
    raise ValueError("--knobs must be a JSON object")
  # basic shape validation
  out: Dict[str, Dict[str, float]] = {}
  for k, v in obj.items():
    if not isinstance(v, dict):
      raise ValueError("knobs entries must be objects (lane->value)")
    out[k] = {}
    for lane, val in v.items():
      out[k][lane] = clamp01(float(val))
  return out


def main():
  ap = argparse.ArgumentParser(description="ASX Local Python Compiler v1.1 (lane-aware weights.v2)")
  ap.add_argument("--brain", default="brain.xml", help="Path to brain.xml")
  ap.add_argument("--out", default="results.json", help="Output results.json")
  ap.add_argument("--dims", type=int, default=DEFAULT_DIMS, help="Weight dims")
  ap.add_argument("--seed", default="ASX::LOCAL", help="Deterministic seed")
  ap.add_argument("--bias", type=float, default=0.05, help="Bias scalar")
  ap.add_argument("--amplitude", type=float, default=0.08, help="Base amplitude")
  ap.add_argument("--knobs", default=None, help='JSON string for lane knobs override, e.g. \'{"temp":{"policy":0.6,"*":0.7}}\'')
  ap.add_argument("--merge_with", default=None, help="Optional path to an existing results.json to merge with")
  ap.add_argument("--merge_mode", default="ema", choices=["override","add","avg","ema","max"], help="Merge mode")
  ap.add_argument("--merge_alpha", type=float, default=0.15, help="EMA alpha")
  ap.add_argument("--verify", action="store_true", help="Verify proofs after compile/merge")

  args = ap.parse_args()

  brain = parse_brain_xml(args.brain)
  knobs_override = parse_knobs_json_arg(args.knobs)

  pack = compile_from_brain(
    brain=brain,
    dims=int(args.dims),
    seed=str(args.seed),
    bias=float(args.bias),
    amplitude=float(args.amplitude),
    knobs_override=knobs_override,
    meta_source="local_python"
  )

  if args.merge_with:
    other = load_json_file(args.merge_with)
    # If other is old weights.v1, you should upgrade externally; we enforce dims/version here
    if int(other.get("dims", -1)) != int(pack["dims"]):
      raise ValueError("merge dims mismatch")
    merged = merge_packs(other, pack, mode=args.merge_mode, alpha=float(args.merge_alpha))
    merged = add_lane_proofs(merged)
    pack = merged

  if args.verify:
    ok = verify_lane_proofs(pack)
    if not ok:
      raise SystemExit("Proof verification FAILED")
    print("[✓] proofs verified")

  save_json_file(args.out, pack)
  print(f"[✓] wrote {args.out}")
  print(f"    version={pack['version']} dims={pack['dims']} lanes={len(pack.get('lanes', {}))} compiler={COMPILER_VERSION}")


if __name__ == "__main__":
  main()
```

### How to use

Compile `brain.xml` → `results.json`:

```bash
python asx_local_compiler_v1_1.py --brain brain.xml --out results.json --seed "LOCAL_DOOM_BRAIN" --verify
```

Override lane knobs:

```bash
python asx_local_compiler_v1_1.py \
  --brain brain.xml \
  --out results.json \
  --knobs '{"temp":{"attention":0.82,"policy":0.58,"*":0.70},"top_p":{"*":0.90},"style":{"style":0.55}}' \
  --verify
```

Merge with an existing pack:

```bash
python asx_local_compiler_v1_1.py \
  --brain brain.xml \
  --out merged.json \
  --merge_with existing.json \
  --merge_mode ema \
  --merge_alpha 0.2 \
  --verify
```

If you want, I can also emit the **matching local JS compiler v1.1** (same laws) so your PWA and local compiler stay bitwise-aligned.
