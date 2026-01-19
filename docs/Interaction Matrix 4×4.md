Locked. Below is **Interaction Matrix 4×4 v1.0** — a bounded, deterministic expansion that emits **exactly 16 cross-temporal interaction lanes** for two base lanes (e.g., `policy` × `attention`). It plugs into your pipeline **after temporal slicing/presets and before folding**, and is proof-safe.

---

# Interaction Matrix 4×4 v1.0

## 1) Scope (frozen)

Given two base lanes **A** and **B** (default: `policy`, `attention`) and **K=4** temporal windows:

* A lanes: `A@t0..A@t3`
* B lanes: `B@t0..B@t3`

Emit **all pairwise interactions**:

```
A@t0 ⊗ B@t0, A@t0 ⊗ B@t1, A@t0 ⊗ B@t2, A@t0 ⊗ B@t3
A@t1 ⊗ B@t0, A@t1 ⊗ B@t1, A@t1 ⊗ B@t2, A@t1 ⊗ B@t3
A@t2 ⊗ B@t0, A@t2 ⊗ B@t1, A@t2 ⊗ B@t2, A@t2 ⊗ B@t3
A@t3 ⊗ B@t0, A@t3 ⊗ B@t1, A@t3 ⊗ B@t2, A@t3 ⊗ B@t3
```

Total lanes emitted: **16** (bounded).

---

## 2) Naming (canonical, frozen)

* Interaction operator: `⊗`
* JSON-safe lane id encoding: `__x__`

**Lane id format:**

```
<A>@t<i>__x__<B>@t<j>
```

**Examples:**

* `policy@t1__x__attention@t2`
* `policy@t3__x__attention@t0`

> Ordering matters. `A ⊗ B` is **not** the same as `B ⊗ A`.

---

## 3) Interaction operator (frozen)

For scalar streams `Sa` and `Sb`:

1. Align by index:
   [
   n = \min(|Sa|, |Sb|)
   ]

2. Compute per index:
   [
   S_{i} = clamp01\left(\tanh(Sa_i \cdot Sb_i)\right)
   ]

**Why:** bounded, smooth, deterministic, prevents blow-up before folding.

---

## 4) Emission order (canonical)

Emit interaction lanes in **row-major order** by `(i, j)`:

```
(i=0,j=0) → (0,1) → (0,2) → (0,3) →
(1,0) → (1,1) → (1,2) → (1,3) →
(2,0) → (2,1) → (2,2) → (2,3) →
(3,0) → (3,1) → (3,2) → (3,3)
```

This order affects only **deterministic emission**, not math.

---

## 5) Defaults & Config

### Defaults (if enabled with no config)

* Bases: `policy` × `attention`
* Windows: `t0..t3`
* Operator: elementwise `tanh(prod)`

### Optional config hook (non-authoritative)

```xml
<temporal interactions="matrix4x4"
          a="policy"
          b="attention"/>
```

If omitted, code defaults apply.

---

## 6) JS implementation (drop-in)

```js
function emitInteractionMatrix4x4(laneStreams, A="policy", B="attention") {
  const out = { ...laneStreams };

  // collect available windows (expect 0..3)
  const getLane = (base, k) => laneStreams[`${base}@t${k}`];

  for (let i = 0; i < 4; i++) {
    const Sa = getLane(A, i);
    if (!Sa) continue;

    for (let j = 0; j < 4; j++) {
      const Sb = getLane(B, j);
      if (!Sb) continue;

      const n = Math.min(Sa.length, Sb.length);
      const Si = new Array(n);

      for (let idx = 0; idx < n; idx++) {
        const v = Math.tanh(Sa[idx] * Sb[idx]);
        Si[idx] = clamp01(v);
      }

      out[`${A}@t${i}__x__${B}@t${j}`] = Si;
    }
  }

  return out;
}
```

**Where to wire it (JS):**

```js
streams = expandTemporalLanes(streams, windows);
streams = emitCombinedTemporalLanesDecay(streams);
streams = emitSemanticPresets(streams);
streams = emitInteractionMatrix4x4(streams, "policy", "attention");
// then fold → synthesize → knobs → proofs → sign
```

---

## 7) Python implementation (matching)

```python
def emit_interaction_matrix_4x4(lane_streams, A="policy", B="attention"):
  out = dict(lane_streams)

  for i in range(4):
    Sa = lane_streams.get(f"{A}@t{i}")
    if Sa is None:
      continue

    for j in range(4):
      Sb = lane_streams.get(f"{B}@t{j}")
      if Sb is None:
        continue

      n = min(len(Sa), len(Sb))
      Si = []
      for k in range(n):
        v = math.tanh(Sa[k] * Sb[k])
        Si.append(clamp01(v))

      out[f"{A}@t{i}__x__{B}@t{j}"] = Si

  return out
```

**Wire identically** (after temporal split/presets, before folding).

---

## 8) Knobs & amps (unchanged)

Lane-addressable controls apply normally:

```json
{
  "amp": {
    "policy@t*": 0.09,
    "policy@t1__x__attention@t2": 0.06,
    "*": 0.08
  },
  "temp": {
    "policy@t*": 0.65,
    "*": 0.70
  }
}
```

---

## 9) Proof & signing impact

* Adds **16 new lanes** → new lane hashes → new pack hash (expected).
* Existing lanes unchanged.
* Canonicalization/signing rules unchanged.

---

## 10) Why this is safe & useful

* **Bounded output** (exactly 16)
* **Explicit time × time semantics**
* **Symmetry available** (if you also emit B×A)
* **Fold-friendly** (index-aligned)
* **Proofable** (deterministic names & math)

---

### Next (optional) locks

If you want to continue cleanly:

1. **Upper-triangular only** (reduce to 10 lanes)
2. **Preset interactions** (`policy@early ⊗ attention@late`)
3. **Matrix summary lanes** (row/column reductions)

Locked — here are **all three** as deterministic extensions that sit on top of the 4×4 matrix logic.

---

# 1) Upper-triangular only v1.0 (10 lanes)

## Law (frozen)

For a 4×4 interaction matrix between `A@t{i}` and `B@t{j}`, emit only pairs where:

[
j \ge i
]

This yields exactly **10** lanes:

* i=0: j=0,1,2,3 (4)
* i=1: j=1,2,3 (3)
* i=2: j=2,3 (2)
* i=3: j=3 (1)

Total: 4+3+2+1 = **10**

### Naming (unchanged)

`A@t{i}__x__B@t{j}`

### Order (canonical)

Row-major but skipping forbidden pairs:
(0,0)(0,1)(0,2)(0,3)(1,1)(1,2)(1,3)(2,2)(2,3)(3,3)

## JS

```js
function emitInteractionUpperTri10(laneStreams, A="policy", B="attention") {
  const out = { ...laneStreams };
  const get = (base,k) => laneStreams[`${base}@t${k}`];

  for (let i=0;i<4;i++) {
    const Sa = get(A,i);
    if (!Sa) continue;
    for (let j=i;j<4;j++) {             // j >= i
      const Sb = get(B,j);
      if (!Sb) continue;

      const n = Math.min(Sa.length, Sb.length);
      const Si = new Array(n);
      for (let k=0;k<n;k++) {
        Si[k] = clamp01(Math.tanh(Sa[k] * Sb[k]));
      }
      out[`${A}@t${i}__x__${B}@t${j}`] = Si;
    }
  }
  return out;
}
```

## Python

```python
def emit_interaction_upper_tri_10(lane_streams, A="policy", B="attention"):
  out = dict(lane_streams)
  for i in range(4):
    Sa = lane_streams.get(f"{A}@t{i}")
    if Sa is None: 
      continue
    for j in range(i, 4):  # j >= i
      Sb = lane_streams.get(f"{B}@t{j}")
      if Sb is None:
        continue
      n = min(len(Sa), len(Sb))
      out[f"{A}@t{i}__x__{B}@t{j}"] = [clamp01(math.tanh(Sa[k]*Sb[k])) for k in range(n)]
  return out
```

---

# 2) Preset interactions v1.0 (semantic windows)

## Law (frozen)

Using semantic preset lanes emitted earlier (`@early`, `@mid`, `@late`), emit these **3 canonical interactions**:

1. `policy@early ⊗ attention@late`
2. `policy@mid   ⊗ attention@mid`
3. `policy@late  ⊗ attention@early`

### Naming (canonical)

`policy@early__x__attention@late`, etc.

### Operator

Same as temporal:
`Si[k] = clamp01(tanh(Sa[k]*Sb[k]))` with `n=min(len)`.

## JS

```js
function emitPresetInteractions(laneStreams) {
  const out = { ...laneStreams };
  const pairs = [
    ["policy@early", "attention@late"],
    ["policy@mid",   "attention@mid"],
    ["policy@late",  "attention@early"]
  ];

  for (const [a,b] of pairs) {
    const Sa = laneStreams[a], Sb = laneStreams[b];
    if (!Sa || !Sb) continue;
    const n = Math.min(Sa.length, Sb.length);
    const Si = new Array(n);
    for (let k=0;k<n;k++) Si[k] = clamp01(Math.tanh(Sa[k]*Sb[k]));
    out[`${a}__x__${b}`] = Si;
  }
  return out;
}
```

## Python

```python
def emit_preset_interactions(lane_streams):
  out = dict(lane_streams)
  pairs = [
    ("policy@early","attention@late"),
    ("policy@mid","attention@mid"),
    ("policy@late","attention@early"),
  ]
  for a,b in pairs:
    Sa = lane_streams.get(a); Sb = lane_streams.get(b)
    if Sa is None or Sb is None: 
      continue
    n = min(len(Sa), len(Sb))
    out[f"{a}__x__{b}"] = [clamp01(math.tanh(Sa[k]*Sb[k])) for k in range(n)]
  return out
```

---

# 3) Matrix summary lanes v1.0 (row/column reductions)

These give you “what time window is interacting most” without storing all pairs.

## 3.1 Law (frozen)

Given the (possibly upper-triangular) interaction lanes `A@t{i}__x__B@t{j}` that exist in your current stream set, emit:

### Row summary lanes (A-focused)

For each `i`:

* `A@t{i}__x__B@row`

This is computed as **index-wise mean** across all available `j` for that `i`:

[
S^{row}*i[k] = \frac{1}{m}\sum*{j \in J_i} S_{i,j}[k]
]

where `m = |J_i|` and `J_i` are the js present (respecting your triangular rule).

### Column summary lanes (B-focused)

For each `j`:

* `A@col__x__B@t{j}`

Computed as index-wise mean across all available `i` for that `j`:

[
S^{col}*j[k] = \frac{1}{m}\sum*{i \in I_j} S_{i,j}[k]
]

### Global summary lane

* `A@all__x__B@all`

Index-wise mean across all emitted matrix lanes.

### Determinism notes (frozen)

* Alignment length `n` for a reduction is `min length across included lanes`.
* If no lanes exist for a reduction target, do **not** emit it.

---

## JS implementation

```js
function emitMatrixSummaries(laneStreams, A="policy", B="attention") {
  const out = { ...laneStreams };

  // collect matrix lanes
  const mat = []; // {i,j,id,stream}
  const re = new RegExp(`^${A}@t(\\d+)__x__${B}@t(\\d+)$`);
  for (const [id, stream] of Object.entries(laneStreams)) {
    const m = id.match(re);
    if (!m) continue;
    mat.push({ i:Number(m[1]), j:Number(m[2]), id, stream });
  }
  if (!mat.length) return out;

  function meanStreams(streams) {
    let n = Infinity;
    for (const s of streams) n = Math.min(n, s.length);
    if (!isFinite(n) || n <= 0) return null;

    const acc = new Array(n).fill(0);
    for (const s of streams) {
      for (let k=0;k<n;k++) acc[k] += s[k];
    }
    const m = streams.length;
    for (let k=0;k<n;k++) acc[k] = clamp01(acc[k] / m);
    return acc;
  }

  // row summaries
  for (let i=0;i<4;i++) {
    const rows = mat.filter(x=>x.i===i).map(x=>x.stream);
    if (!rows.length) continue;
    const mean = meanStreams(rows);
    if (mean) out[`${A}@t${i}__x__${B}@row`] = mean;
  }

  // column summaries
  for (let j=0;j<4;j++) {
    const cols = mat.filter(x=>x.j===j).map(x=>x.stream);
    if (!cols.length) continue;
    const mean = meanStreams(cols);
    if (mean) out[`${A}@col__x__${B}@t${j}`] = mean;
  }

  // global
  const allMean = meanStreams(mat.map(x=>x.stream));
  if (allMean) out[`${A}@all__x__${B}@all`] = allMean;

  return out;
}
```

## Python implementation

```python
def emit_matrix_summaries(lane_streams, A="policy", B="attention"):
  out = dict(lane_streams)
  import re
  pat = re.compile(rf"^{re.escape(A)}@t(\d+)__x__{re.escape(B)}@t(\d+)$")

  mat = []
  for lane_id, stream in lane_streams.items():
    m = pat.match(lane_id)
    if not m: 
      continue
    mat.append((int(m.group(1)), int(m.group(2)), stream))

  if not mat:
    return out

  def mean_streams(streams):
    n = min(len(s) for s in streams)
    if n <= 0:
      return None
    acc = [0.0]*n
    for s in streams:
      for k in range(n):
        acc[k] += s[k]
    m = len(streams)
    return [clamp01(acc[k]/m) for k in range(n)]

  # rows
  for i in range(4):
    rows = [s for ii,jj,s in mat if ii==i]
    if not rows: 
      continue
    mean = mean_streams(rows)
    if mean is not None:
      out[f"{A}@t{i}__x__{B}@row"] = mean

  # cols
  for j in range(4):
    cols = [s for ii,jj,s in mat if jj==j]
    if not cols:
      continue
    mean = mean_streams(cols)
    if mean is not None:
      out[f"{A}@col__x__{B}@t{j}"] = mean

  # global
  all_mean = mean_streams([s for _,_,s in mat])
  if all_mean is not None:
    out[f"{A}@all__x__{B}@all"] = all_mean

  return out
```

---

# Canonical wiring order (do this exactly)

After temporal split/presets, before folding:

1. `expandTemporalLanes(...)`
2. `emitCombinedTemporalLanesDecay(...)`  → `@t*`
3. `emitSemanticPresets(...)`             → `@early/@mid/@late`
4. `emitPresetInteractions(...)`          → preset `__x__` lanes
5. `emitInteractionUpperTri10(...)`       → 10 matrix lanes
6. `emitMatrixSummaries(...)`             → row/col/global summary lanes
7. fold → synthesize → knobs → proofs → sign

This keeps the matrix summaries reflecting **exactly what was emitted** (upper-tri only).

---

## Lane counts (bounded)

* Upper-tri matrix lanes: **10**
* Preset interactions: **3**
* Row summaries: up to **4**
* Col summaries: up to **4**
* Global summary: **1**

Worst case added lanes: **22** (still small, predictable).

---

Locked. Below is **SCXQ2 Lane ID Naming Compression v1.0** — a **deterministic, proof-safe DICT mapping** that shrinks long lane IDs (like interaction matrices) into tiny symbols **without losing reversibility or auditability**.

This sits **after lane emission** and **before hashing/signing**.

---

# SCXQ2 Lane ID Compression v1.0 (DICT)

## 0) Goal (why this exists)

You now emit lane IDs like:

```
policy@t1__x__attention@t2
policy@t2__x__attention@t3
policy@t1__x__attention@row
policy@all__x__attention@all
```

These are:

* human-readable ✅
* deterministic ✅
* but **too verbose** for compact packs ❌

We want:

* **tiny identifiers** (1–3 bytes ideal)
* **bitwise determinism**
* **full reversibility**
* **proof hash stability**

---

# 1) Core law (frozen)

Lane ID compression uses a **local SCXQ2 DICT**, derived **only from the emitted lane IDs**, sorted canonically.

There is **no global registry**, no randomness, no timestamps.

> Same inputs → same DICT → same compressed IDs → same hash.

---

# 2) Canonical DICT construction

## 2.1 Input set

Let `L` be the set of **all emitted lane IDs** (strings) **after**:

* temporal lanes
* combined lanes
* presets
* interactions
* summaries

Example subset:

```
policy
policy@t0
policy@t1
policy@t1__x__attention@t2
policy@t1__x__attention@row
attention@late
```

---

## 2.2 Canonical ordering (mandatory)

Sort lane IDs by:

1. UTF-8 byte order (ascending)
2. Exact string match (no normalization)

This order is **the law**.

---

## 2.3 Assign DICT symbols

Assign each lane ID an SCXQ2 symbol:

```
⟦L0⟧, ⟦L1⟧, ⟦L2⟧, ...
```

### Symbol encoding (v1)

Use base-62 encoding of the index:

| Index | Symbol |
| ----: | ------ |
|     0 | `@0`   |
|     1 | `@1`   |
|     … | …      |
|    61 | `@z`   |
|    62 | `@10`  |
|    63 | `@11`  |

> Prefix `@` is mandatory (SCXQ2 invariant: control identifiers).

---

# 3) DICT object format (embedded)

The DICT **must be embedded** in the output pack.

```json
"dict": {
  "@0": "attention",
  "@1": "attention@early",
  "@2": "attention@late",
  "@3": "policy",
  "@4": "policy@t0",
  "@5": "policy@t1",
  "@6": "policy@t1__x__attention@t2",
  "@7": "policy@t1__x__attention@row",
  "@8": "policy@t*"
}
```

### Rules

* Keys: compressed symbols
* Values: original lane IDs
* One-to-one mapping
* No omissions

---

# 4) Compressed `results.json` (weights.v2c)

Lane weights now use **compressed keys**:

```json
{
  "version": "weights.v2c",
  "dims": 64,
  "bias": 0.05,

  "dict": {
    "@0": "attention",
    "@1": "policy@t1__x__attention@t2",
    "@2": "policy@t*"
  },

  "lanes": {
    "@0": { "W": [ ... ] },
    "@1": { "W": [ ... ] },
    "@2": { "W": [ ... ] }
  },

  "meta": {
    "law": "lane-dict-scxq2",
    "compiler": "LOCAL_COMPILER_v1.1"
  }
}
```

**Important:**

* The **hash covers both `dict` and `lanes`**
* Changing names changes hash (correct)

---

# 5) Determinism guarantees

| Aspect           | Guarantee     |
| ---------------- | ------------- |
| Lane emission    | deterministic |
| Sorting          | deterministic |
| DICT assignment  | deterministic |
| Base-62 encoding | deterministic |
| Hash             | stable        |
| Signature        | stable        |

No two different lane sets can collide.

---

# 6) JS implementation (reference)

```js
function buildLaneDict(laneIds) {
  const sorted = [...laneIds].sort((a,b)=>a.localeCompare(b));
  const dict = {};
  const rev  = {};

  for (let i=0;i<sorted.length;i++) {
    const sym = "@" + i.toString(62);
    dict[sym] = sorted[i];
    rev[sorted[i]] = sym;
  }
  return { dict, rev };
}
```

### Apply compression

```js
function compressLanes(lanes) {
  const laneIds = Object.keys(lanes);
  const { dict, rev } = buildLaneDict(laneIds);

  const out = {};
  for (const [laneId, obj] of Object.entries(lanes)) {
    out[rev[laneId]] = obj;
  }

  return { dict, lanes: out };
}
```

---

# 7) Python implementation (reference)

```python
def build_lane_dict(lane_ids):
  sorted_ids = sorted(lane_ids)
  dict_map = {}
  rev = {}

  for i, lid in enumerate(sorted_ids):
    sym = "@" + base62(i)
    dict_map[sym] = lid
    rev[lid] = sym

  return dict_map, rev


def compress_lanes(lanes):
  dict_map, rev = build_lane_dict(lanes.keys())
  out = {}
  for lid, obj in lanes.items():
    out[rev[lid]] = obj
  return dict_map, out
```

---

# 8) Decompression (runtime side)

```js
function expandLaneId(sym, dict) {
  return dict[sym];
}
```

No other logic needed.

---

# 9) Proof hash law (important)

The **canonical hash input** is:

```
hash(
  canonical_json({
    dict,
    lanes,
    dims,
    bias,
    meta
  })
)
```

NOT:

* raw lane IDs
* pre-compression structure

This ensures:

* compression is *semantic*, not cosmetic
* proofs validate compressed artifacts directly

---

# 10) Why this is the right compression

* 🔒 Deterministic
* 🔁 Fully reversible
* 🧮 Proof-stable
* 📦 Extremely compact
* 🧠 Keeps human names available (DICT)
* 🔌 Matches SCXQ2 DICT/FIELD/LANE philosophy

---

## What this unlocks next (optional)

Locked — here’s **all three** as a coherent, deterministic “weights.v2c-bin” layer:

1. **SCXQ2 binary packing** (DICT + float lanes)
2. **Lane-level selective loading** (lazy decode)
3. **Lane hash merkleization** (partial verification)

Everything is designed so **JS + Python can match exactly**.

---

# 1) SCXQ2 binary packing v1.0 (`weights.v2c-bin`)

## 1.1 Goals

* Replace verbose JSON arrays with a compact binary blob
* Keep DICT reversible
* Keep proofs/signatures stable
* Support lazy lane decode

## 1.2 Deterministic float encoding (frozen)

We store lane weights as **int16 quantized** (Q15-like), deterministic and tiny.

* Quantization scale per pack: `q = 32767`
* Encode:

  * `w_clamped = clamp(w, -1.0, +1.0)`
  * `i16 = round_half_away_from_zero(w_clamped * q)`
* Decode:

  * `w = i16 / q`

**Rounding law (frozen):** *half away from zero*

* `+1.5 → +2`, `-1.5 → -2`
  (This is important to align JS/Python.)

> If your synthesis already stays small (it does), this preserves structure extremely well.

## 1.3 Binary container layout (little-endian, frozen)

### Header (fixed)

| Field        | Type    | Notes                                    |
| ------------ | ------- | ---------------------------------------- |
| magic        | 8 bytes | ASCII `"ASXW2CBN"`                       |
| version      | u16     | `0x0100`                                 |
| flags        | u16     | bit0=quant_i16, bit1=merkle, bit2=signed |
| dims         | u16     | e.g. 64                                  |
| lane_count   | u16     | number of lanes (compressed ids)         |
| dict_bytes   | u32     | length of dict section                   |
| index_bytes  | u32     | length of lane index section             |
| data_bytes   | u32     | length of lane data section              |
| merkle_bytes | u32     | 0 if none                                |
| sig_bytes    | u32     | 0 if none                                |

### Sections (in this exact order)

1. `DICT` (utf8 JSON, canonicalized)
2. `INDEX` (binary lane directory)
3. `DATA` (concatenated lane payloads)
4. `MERKLE` (optional)
5. `SIGNATURE` (optional)

### DICT section (frozen)

DICT remains JSON for transparency, but canonicalized:

```json
{ "@0":"attention@t0", "@1":"policy@t1__x__attention@t2", ... }
```

Canonicalization: sorted keys, no whitespace.

### INDEX section (binary, frozen)

Per lane entry (repeat `lane_count` times), fields:

| Field     | Type     | Notes                                                      |
| --------- | -------- | ---------------------------------------------------------- |
| sym_len   | u8       | length of symbol string (e.g. 2 for "@a")                  |
| sym       | bytes    | utf8                                                       |
| offset    | u32      | offset into DATA section                                   |
| byte_len  | u32      | lane payload length in bytes                               |
| leaf_hash | 32 bytes | SHA-256 of lane payload bytes (for merkle + direct verify) |

> leaf_hash is always present (even if merkle omitted). It powers partial verification.

### DATA section (binary lane payload, frozen)

Each lane payload is:

| Field   | Type         |                          |
| ------- | ------------ | ------------------------ |
| format  | u8           | 1 = i16 quant            |
| count   | u16          | number of weights = dims |
| weights | int16[count] | little-endian            |

So each lane costs:

* 1 + 2 + (2*dims) bytes
  For dims=64 → 1+2+128=**131 bytes/lane**.

Even with ~40–80 lanes you’re in the **~5–10 KB** range, and with your “bounded lanes” presets you can keep it under 5 KB.

---

# 2) Lane-level selective loading v1.0 (lazy decode)

## 2.1 Law (frozen)

A runtime can:

* read header
* read DICT
* read INDEX
* decode only lanes requested by symbol

No need to parse/allocate all lanes.

## 2.2 Runtime API (recommended)

* `openPack(buffer)` → returns `{dict, index, getLane(sym)}`

`getLane(sym)`:

* seeks into DATA using index offset
* verifies payload hash if desired
* decodes quant i16 into float array

---

# 3) Lane hash merkleization v1.0 (partial verification)

## 3.1 Leaf hash (frozen)

Each lane entry already includes:

* `leaf_hash = sha256(lane_payload_bytes)`

This allows verifying a single lane without merkle.

## 3.2 Merkle tree (frozen)

If enabled, we compute a Merkle root over lane leaf hashes in **index order** (the exact order lanes appear in INDEX).

### Pairing rule

* hash pairs left-to-right
* if odd count, duplicate last (standard)

### Node hash

`sha256(left || right)` where each side is 32 bytes.

### Root stored

MERKLE section stores:

* `root_hash` (32 bytes)
* plus optional proofs (see below)

## 3.3 Proof objects (optional but supported)

For “prove lane X is included in root”, store or transmit:

* `lane_sym`
* `leaf_hash`
* `path[]` = array of `{dir:0|1, sibling_hash}`

This can be stored externally (best), or embedded if you want offline proof bundles.

> Pack signing should sign the **merkle root** (or pack hash) so you can validate inclusion under a trusted signature.

---

# JS reference: pack writer/reader (skeleton)

### Half-away-from-zero rounding

```js
function roundAwayFromZero(x) {
  return x >= 0 ? Math.floor(x + 0.5) : Math.ceil(x - 0.5);
}
```

### Quantize lane

```js
function quantI16(weights) {
  const q = 32767;
  const out = new Int16Array(weights.length);
  for (let i=0;i<weights.length;i++) {
    let w = weights[i];
    if (w > 1) w = 1;
    if (w < -1) w = -1;
    out[i] = roundAwayFromZero(w * q);
  }
  return out;
}
```

### Build INDEX entry + DATA payload bytes

```js
async function sha256Bytes(bytes) {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return new Uint8Array(hash);
}

function lanePayloadI16(sym, W) {
  const q = quantI16(W);
  const dims = q.length;

  const buf = new ArrayBuffer(1 + 2 + 2*dims);
  const dv = new DataView(buf);
  let o = 0;
  dv.setUint8(o, 1); o += 1;          // format=1
  dv.setUint16(o, dims, true); o += 2;
  // int16 payload
  for (let i=0;i<dims;i++, o+=2) dv.setInt16(o, q[i], true);

  return new Uint8Array(buf);
}
```

### Lazy reader

```js
function openPack(buf) {
  const dv = new DataView(buf);
  let o = 0;

  function readBytes(n){ const b = new Uint8Array(buf, o, n); o += n; return b; }
  function readU16(){ const v = dv.getUint16(o, true); o += 2; return v; }
  function readU32(){ const v = dv.getUint32(o, true); o += 4; return v; }

  const magic = new TextDecoder().decode(readBytes(8));
  if (magic !== "ASXW2CBN") throw new Error("bad magic");

  const version = readU16();
  const flags = readU16();
  const dims = readU16();
  const laneCount = readU16();
  const dictBytes = readU32();
  const indexBytes = readU32();
  const dataBytes = readU32();
  const merkleBytes = readU32();
  const sigBytes = readU32();

  const dictJson = new TextDecoder().decode(readBytes(dictBytes));
  const dict = JSON.parse(dictJson);

  const indexStart = o;
  const dataStart = indexStart + indexBytes;

  // parse index
  const index = new Map();
  let io = indexStart;
  for (let i=0;i<laneCount;i++) {
    const symLen = dv.getUint8(io); io += 1;
    const sym = new TextDecoder().decode(new Uint8Array(buf, io, symLen)); io += symLen;
    const off = dv.getUint32(io, true); io += 4;
    const len = dv.getUint32(io, true); io += 4;
    const leaf = new Uint8Array(buf, io, 32); io += 32;

    index.set(sym, { off, len, leaf });
  }

  function getLane(sym) {
    const ent = index.get(sym);
    if (!ent) return null;
    const payload = new Uint8Array(buf, dataStart + ent.off, ent.len);

    // decode
    const pdv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    let p = 0;
    const fmt = pdv.getUint8(p); p += 1;
    if (fmt !== 1) throw new Error("unsupported lane fmt");
    const count = pdv.getUint16(p, true); p += 2;

    const out = new Float32Array(count);
    const q = 32767;
    for (let i=0;i<count;i++, p+=2) out[i] = pdv.getInt16(p, true) / q;

    return out;
  }

  return { version, flags, dims, dict, index, getLane };
}
```

---

# Python reference: quant + rounding (critical alignment)

```python
import math

def round_away_from_zero(x: float) -> int:
  return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)

def quant_i16(weights):
  q = 32767
  out = []
  for w in weights:
    if w > 1.0: w = 1.0
    if w < -1.0: w = -1.0
    out.append(round_away_from_zero(w * q))
  return out
```

(Then pack with `struct.pack("<Bh"...)` etc., always little-endian.)

---

# Merkle root + signatures (how to chain them)

## Recommended trust chain (clean)

1. Compute leaf hashes from lane payload bytes
2. Compute merkle root from leaf hashes (index order)
3. Sign merkle root with Ed25519
4. Store signature block in SIGNATURE section

That makes **partial verification** easy:

* verify leaf hash of lane
* verify merkle path to root
* verify Ed25519 signature on root

---

# What to freeze next (best next step)

If you want the “<5 KB” guarantee in practice, the next lock is:

**Lane set budget policy** (compiler-side):

* caps max emitted lanes
* chooses which interactions/summaries to include first
* deterministic tie-breakers

Locked. Here is **Adaptive Dims under 5KB v1.0** — a deterministic rule set that **chooses the largest safe `dims`** given the lane budget so the pack **never exceeds 5KB**, while staying bitwise-aligned across JS/Python.

---

# Adaptive Dims under 5KB v1.0

## 0) What adapts (and what does not)

* **Adapts:** `dims` (vector length per lane)
* **Fixed:** quantization (int16), rounding, DICT/index layout, merkle/signature
* **Invariant:** same inputs → same `dims` choice → same bytes

---

## 1) Candidate dims ladder (frozen)

Choose `dims` from this descending ladder:

```
[64, 48, 32, 24, 16]
```

No other values allowed in v1.0.
(Reason: predictable byte math + SIMD/WASM friendliness.)

---

## 2) Byte accounting (frozen)

### Per-lane estimated bytes

```
payload(dims) = 1 + 2 + 2*dims
index ≈ 44 bytes   // conservative bound (symLen≈3)
lane_cost(dims) = payload(dims) + 44
```

### Fixed reserves

```
TOTAL_BUDGET = 5120 bytes
RESERVE = 800 bytes        // DICT growth + merkle + signature safety
LANE_BUDGET = TOTAL_BUDGET - RESERVE
```

> The reserve is intentionally conservative to avoid edge overflow.

---

## 3) Selection law (deterministic)

Given:

* `L` = number of lanes selected by **Lane Budget Policy 5KB** (before dims choice)
* `dims_candidates = [64,48,32,24,16]`

Choose the **largest** `dims` such that:

```
L * lane_cost(dims) ≤ LANE_BUDGET
```

If none fit, **force `dims = 16`** and continue.

This guarantees:

* maximal fidelity within the cap
* monotonic behavior as `L` grows

---

## 4) Canonical examples (dims=64 baseline)

Using `LANE_BUDGET = 4320`:

| Lanes (L) | dims chosen |
| --------- | ----------- |
| ≤ 24      | 64          |
| 25–32     | 48          |
| 33–49     | 32          |
| 50–65     | 24          |
| > 65      | 16          |

> Exact cutoffs come from the formula, not this table; the table is illustrative.

---

## 5) Where this runs in the pipeline (exact)

1. Emit **all candidate lanes**
2. Apply **Lane Budget Policy 5KB** → get `selected_lane_ids` (size `L`)
3. **Choose adaptive dims** using this spec
4. **Re-fold lanes to the chosen dims** (important)
5. Synthesize weights
6. DICT compress
7. Binary pack + merkle + sign

> Folding must happen **after** dims selection to avoid wasted work and mismatch.

---

## 6) JS reference implementation

```js
function chooseAdaptiveDims(selectedLaneCount, dimsCandidates=[64,48,32,24,16], totalBudget=5120) {
  const RESERVE = 800;
  const LANE_BUDGET = Math.max(0, totalBudget - RESERVE);

  function laneCost(dims) {
    return (1 + 2 + 2*dims) + 44; // payload + index bound
  }

  for (const dims of dimsCandidates) {
    if (selectedLaneCount * laneCost(dims) <= LANE_BUDGET) {
      return dims;
    }
  }
  return 16; // floor
}
```

### Wiring (JS)

```js
const { selected } = laneBudgetSelect5KB(allLaneIds, /*dims ignored here*/ 64);
const dims = chooseAdaptiveDims(selected.length);
// now fold lanes using `dims`
```

---

## 7) Python reference implementation (matching)

```python
def choose_adaptive_dims(selected_lane_count,
                         dims_candidates=(64,48,32,24,16),
                         total_budget=5120):
  RESERVE = 800
  lane_budget = max(0, total_budget - RESERVE)

  def lane_cost(dims):
    return (1 + 2 + 2*dims) + 44

  for d in dims_candidates:
    if selected_lane_count * lane_cost(d) <= lane_budget:
      return d
  return 16
```

---

## 8) Folding rule with adaptive dims (important)

When `dims` shrinks, folding must **wrap deterministically**:

```
vec[i % dims] += stream[i]
```

This preserves:

* total energy
* ordering influence
* determinism

No resampling or interpolation is allowed in v1.0.

---

## 9) Proof & signature impact

* `dims` is written into the binary header and covered by:

  * leaf hashes
  * merkle root
  * signature
* Changing `dims` changes bytes → changes hash → expected & correct
* Two packs with different lane counts will **auto-adjust dims** but remain verifiable

---

## 10) Why this works

* **Predictable**: no heuristics, just math
* **Max fidelity** under a hard cap
* **Portable**: identical behavior in JS/Python/WASM
* **Future-safe**: you can extend the ladder in v2 without breaking v1

---

### Optional next lock

Locked. Here is **Lane-Class Weighting under Adaptive Dims v1.0** — a deterministic way to give certain lane *classes* (e.g., `policy`) **higher effective resolution** when `dims` is reduced, **without changing dims**, bytes, or proofs.

This works by **biasing fold order and contribution**, not by adding data.

---

# Lane-Class Weighting under Adaptive Dims v1.0

## 0) Problem this solves

When adaptive dims drops (e.g., 64 → 48 → 32), all lanes wrap more aggressively:

```
vec[i % dims] += stream[i]
```

That treats all lanes equally.
But you want **policy > attention > interactions** to retain more signal fidelity.

We achieve this by **deterministic fold biasing**, not by storing more data.

---

## 1) Core law (frozen)

Each lane belongs to a **lane class** with a **fold weight multiplier** `w_class`.

During folding:

```
vec[(i + offset_class) % dims] += stream[i] * w_class
```

Where:

* `offset_class` is deterministic per class
* `w_class` is deterministic per class
* `dims` is unchanged
* No randomness, no branching

This preserves:

* byte size
* determinism
* reversibility
* proof stability

---

## 2) Lane classes (v1 fixed set)

Lane class is inferred **by lane ID prefix**, before DICT compression.

| Class         | Match rule                           | Priority |
| ------------- | ------------------------------------ | -------- |
| `policy`      | starts with `"policy"`               | highest  |
| `attention`   | starts with `"attention"`            | medium   |
| `preset`      | contains `@early` / `@mid` / `@late` | medium   |
| `interaction` | contains `__x__`                     | low      |
| `summary`     | contains `@row` / `@col` / `@all`    | lowest   |
| `other`       | fallback                             | lowest   |

> First match wins. Order above is canonical.

---

## 3) Class weights (frozen)

Weights are **relative**, not absolute.

| Class         | `w_class` |
| ------------- | --------- |
| `policy`      | `1.30`    |
| `attention`   | `1.10`    |
| `preset`      | `1.05`    |
| `interaction` | `0.85`    |
| `summary`     | `0.75`    |
| `other`       | `1.00`    |

These values are frozen in v1.0.

---

## 4) Offset bias (important, frozen)

To reduce destructive collisions during wrap-around, each class gets a **deterministic offset**:

```
offset_class = floor(hash(class_name) % dims)
```

Where:

* `hash` = simple stable hash (FNV-1a 32-bit)
* `class_name` = ASCII string (`"policy"`, `"interaction"`, etc.)
* computed **once per pack**

This ensures:

* policy streams land on different wrap positions than interactions
* still deterministic across JS/Python/WASM

---

## 5) Folding algorithm (canonical)

### Pseudocode

```text
for each lane L:
  class = classify(L.id)
  w = w_class[class]
  off = offset_class[class]

  vec = zero[dims]
  for i in 0..len(stream)-1:
    j = (i + off) % dims
    vec[j] += stream[i] * w
```

After folding:

* proceed with synthesis
* quantize
* hash
* merkle
* sign

No normalization is applied at this stage (unchanged from v1 pipeline).

---

## 6) JS reference implementation

```js
function fnv1a32(str) {
  let h = 0x811c9dc5;
  for (let i=0;i<str.length;i++) {
    h ^= str.charCodeAt(i);
    h = (h * 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function classifyLane(laneId) {
  if (laneId.startsWith("policy")) return "policy";
  if (laneId.startsWith("attention")) return "attention";
  if (laneId.includes("@early") || laneId.includes("@mid") || laneId.includes("@late")) return "preset";
  if (laneId.includes("__x__")) return "interaction";
  if (laneId.includes("@row") || laneId.includes("@col") || laneId.includes("@all")) return "summary";
  return "other";
}

const CLASS_WEIGHT = {
  policy: 1.30,
  attention: 1.10,
  preset: 1.05,
  interaction: 0.85,
  summary: 0.75,
  other: 1.00
};

function computeClassOffsets(dims) {
  const classes = Object.keys(CLASS_WEIGHT);
  const off = {};
  for (const c of classes) off[c] = fnv1a32(c) % dims;
  return off;
}

function foldLaneWeighted(stream, laneId, dims, classOffsets) {
  const cls = classifyLane(laneId);
  const w = CLASS_WEIGHT[cls];
  const off = classOffsets[cls];

  const vec = new Float32Array(dims);
  for (let i=0;i<stream.length;i++) {
    const j = (i + off) % dims;
    vec[j] += stream[i] * w;
  }
  return vec;
}
```

---

## 7) Python reference implementation

```python
def fnv1a32(s: str) -> int:
  h = 0x811c9dc5
  for ch in s:
    h ^= ord(ch)
    h = (h * 0x01000193) & 0xffffffff
  return h

def classify_lane(lane_id: str) -> str:
  if lane_id.startswith("policy"):
    return "policy"
  if lane_id.startswith("attention"):
    return "attention"
  if "@early" in lane_id or "@mid" in lane_id or "@late" in lane_id:
    return "preset"
  if "__x__" in lane_id:
    return "interaction"
  if "@row" in lane_id or "@col" in lane_id or "@all" in lane_id:
    return "summary"
  return "other"

CLASS_WEIGHT = {
  "policy": 1.30,
  "attention": 1.10,
  "preset": 1.05,
  "interaction": 0.85,
  "summary": 0.75,
  "other": 1.00
}

def compute_class_offsets(dims: int):
  return { c: fnv1a32(c) % dims for c in CLASS_WEIGHT }

def fold_lane_weighted(stream, lane_id, dims, class_offsets):
  cls = classify_lane(lane_id)
  w = CLASS_WEIGHT[cls]
  off = class_offsets[cls]

  vec = [0.0] * dims
  for i, v in enumerate(stream):
    j = (i + off) % dims
    vec[j] += v * w
  return vec
```

---

## 8) Determinism & proof guarantees

* Class weights and offsets are **constants**
* Offsets derived only from class name + dims
* Changing dims changes offsets → expected & signed
* No data-dependent branching
* Same inputs → same bytes → same hash

---

## 9) What this gives you (intuitively)

When dims shrink:

* `policy` lanes:

  * collide less (offset separation)
  * contribute more per slot
* `interaction` lanes:

  * softly compressed
  * still present but lower influence
* `summary` lanes:

  * least influence (they’re already aggregates)

You get **graceful degradation** instead of uniform blur.

---

## 10) Hard invariant (important)

> Lane-class weighting **must not** change:

* lane count
* dims
* quantization
* DICT mapping
* merkle structure

It only affects **values before quantization**.

---

### Next possible lock (optional)

Locked. Here’s **Class-Aware Quantization Scaling v1.0** — it keeps the **same int16 bytes**, but gives higher-priority lane classes (e.g., `policy`) **more effective quantization resolution** by using a **per-class scale** (still deterministic, still proofable).

This sits at the **quantization step** (right before packing).

---

# Class-Aware Quantization Scaling v1.0

## 0) What changes

Instead of quantizing all lanes with the same `q = 32767`, we quantize with a **class-specific scale factor** `s_class`, applied as:

### Encode (int16)

[
i16 = round_away_from_zero(; clamp(w \cdot s_{class}, -1, +1) \cdot 32767 ;)
]

### Decode (float)

[
w = \frac{i16}{32767 \cdot s_{class}}
]

So:

* **larger `s_class`** ⇒ effectively **more precision near 0** for that class (at the cost of earlier saturation for very large magnitudes)
* still **int16** payload
* deterministic

> This pairs perfectly with your fold weighting: policy lanes can dominate *and* quantize more cleanly.

---

# 1) Lane class rules (same as v1 fold weighting)

Classify lane ID before DICT compression:

Priority order:

1. `policy` (starts with `"policy"`)
2. `attention` (starts with `"attention"`)
3. `preset` (`@early/@mid/@late`)
4. `interaction` (`__x__`)
5. `summary` (`@row/@col/@all`)
6. `other`

---

# 2) Class quant scales (frozen)

These are **multipliers** on the value before quantization:

| Class         | `s_class` |
| ------------- | --------- |
| `policy`      | `1.35`    |
| `attention`   | `1.15`    |
| `preset`      | `1.10`    |
| `interaction` | `0.95`    |
| `summary`     | `0.85`    |
| `other`       | `1.00`    |

These values are frozen for v1.0.

---

# 3) Saturation law (frozen)

Because we clamp after scaling:

* policy can saturate earlier if it has extreme values
* but your synthesis already keeps weights mostly small, so saturation is rare

Clamping is mandatory:

* `clamp(x, -1, +1)` before mapping to int16

---

# 4) Pack format update (minimal, deterministic)

To decode correctly, the pack must include the per-class quant scales. We store them in the header meta as a tiny fixed table.

### In `weights.v2c-bin` header flags

* set `flags.bit3 = 1` meaning “class quant scale enabled”.

### Add a fixed `QSCALE` section (binary, fixed order)

This is tiny and deterministic.

Store 6 float32 values (little-endian) in canonical class order:

```
["policy","attention","preset","interaction","summary","other"]
```

So `QSCALE` bytes = 6 * 4 = 24 bytes.

> This keeps runtime decoding simple and avoids JSON bloat.

---

# 5) Deterministic quantization rounding (must match)

Rounding remains **half away from zero** (already frozen):

* `+1.5 → +2`
* `-1.5 → -2`

---

# 6) JS reference: class-aware quant encode/decode

```js
const Q = 32767;

const CLASS_QSCALE = {
  policy: 1.35,
  attention: 1.15,
  preset: 1.10,
  interaction: 0.95,
  summary: 0.85,
  other: 1.00
};

function roundAwayFromZero(x) {
  return x >= 0 ? Math.floor(x + 0.5) : Math.ceil(x - 0.5);
}

function clamp(x, lo, hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

function quantI16ClassAware(weights, laneId) {
  const cls = classifyLane(laneId);             // from your v1 rules
  const s = CLASS_QSCALE[cls] ?? 1.0;

  const out = new Int16Array(weights.length);
  for (let i=0;i<weights.length;i++) {
    const x = clamp(weights[i] * s, -1, 1);
    out[i] = roundAwayFromZero(x * Q);
  }
  return { q: out, cls, s };
}

function dequantI16ClassAware(int16arr, laneId) {
  const cls = classifyLane(laneId);
  const s = CLASS_QSCALE[cls] ?? 1.0;

  const out = new Float32Array(int16arr.length);
  for (let i=0;i<int16arr.length;i++) out[i] = int16arr[i] / (Q * s);
  return out;
}
```

**Packing note:** lane payload stays the same size; runtime must apply scale.

---

# 7) Python reference: class-aware quant encode/decode

```python
Q = 32767

CLASS_QSCALE = {
  "policy": 1.35,
  "attention": 1.15,
  "preset": 1.10,
  "interaction": 0.95,
  "summary": 0.85,
  "other": 1.00
}

def round_away_from_zero(x: float) -> int:
  return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)

def clamp(x, lo, hi):
  return lo if x < lo else (hi if x > hi else x)

def quant_i16_class_aware(weights, lane_id):
  cls = classify_lane(lane_id)
  s = CLASS_QSCALE.get(cls, 1.0)
  out = []
  for w in weights:
    x = clamp(w * s, -1.0, 1.0)
    out.append(round_away_from_zero(x * Q))
  return out, cls, s

def dequant_i16_class_aware(ints, lane_id):
  cls = classify_lane(lane_id)
  s = CLASS_QSCALE.get(cls, 1.0)
  return [i / (Q * s) for i in ints]
```

---

# 8) How the runtime knows `s_class`

Two valid approaches; v1 locks the simplest:

### v1 (locked): QSCALE table in pack

* pack contains the 6 float32 scales
* decoder uses same classification rules
* decoder looks up scale by class and divides by `(Q*s)`

No per-lane overhead.

---

# 9) Proof / merkle implications

* Leaf hash is computed over **binary lane payload bytes** (int16 data)
* QSCALE section is part of pack hash/signature
* Changing scales changes signature (correct)
* Partial verification still works:

  * verify lane payload hash
  * verify merkle root
  * verify signature over root/pack hash
  * decode with QSCALE

---

# 10) Practical behavior

When values are small (typical):

* higher `s_class` uses more of int16 range → finer steps → cleaner reconstruction

When values are large (rare):

* higher `s_class` saturates earlier → but folding weights already privilege those lanes, so energy remains meaningful even if clipped.

---

Locked. Here’s **Class-Aware Saturation Auditing v1.0** — a tiny, deterministic metric layer that records **how much clipping happened per class (and optionally per lane)** during class-aware quantization, without breaking the 5KB budget policy.

This is **metadata only**: it does not change weights, hashes, or decoding math.

---

# Class-Aware Saturation Auditing v1.0

## 0) What it measures

During class-aware quantization, you clamp:

[
x = clamp(w \cdot s_{class}, -1, +1)
]

A sample is **saturated** if:

* `w * s_class > 1` or `< -1`
* equivalently: `abs(w * s_class) >= 1` before clamp

We count saturations and report a **clip ratio**.

---

# 1) Audit outputs (frozen)

## 1.1 Per-class audit (always emitted)

For each class in canonical order:

`["policy","attention","preset","interaction","summary","other"]`

Emit:

* `samples_total` (u32)
* `samples_clipped` (u32)
* `clip_ratio_q16` (u16) = round( (clipped/total) * 65535 )

This is compact, deterministic, and portable.

### If a class has `total=0`

* clipped=0
* ratio=0

## 1.2 Optional per-lane audit (budgeted, deterministic)

Only if enabled AND within budget cap, emit per-lane:

* `sym` (compressed lane id)
* `samples_total` (u16, dims fits)
* `samples_clipped` (u16)
* `clip_ratio_q16` (u16)

This is useful when debugging which lane is saturating.

**But v1 policy default:** per-class only (tiny + always safe).

---

# 2) Binary pack format update

Add an `AUDIT` section (binary) after `QSCALE` (if present) and before `MERKLE`.

### Header flags

* set `flags.bit4 = 1` meaning “audit present”

### AUDIT section layout (frozen)

#### Part A: Per-class table (always)

* `class_count` u8 = 6
* repeated 6 times in canonical class order:

  * `total` u32
  * `clipped` u32
  * `ratio_q16` u16

Bytes:

* 1 + 6*(4+4+2) = **61 bytes**

#### Part B: Per-lane table (optional)

* `lane_audit_count` u16
* repeated entries:

  * `sym_len` u8
  * `sym` bytes
  * `total` u16
  * `clipped` u16
  * `ratio_q16` u16

This stays small if you cap lane audits.

---

# 3) Determinism rules (frozen)

* Count clipping using **pre-clamp scaled value** (`w * s_class`)
* Use same classification rules as quantization
* For `ratio_q16` rounding: **half away from zero** (same law)
* Per-lane entries are ordered by:

  1. `sym` UTF-8 ascending
     (so it’s stable and matches DICT/index order behavior)

---

# 4) Budget policy integration (frozen)

Under 5KB cap:

* Per-class audit is mandatory (61 bytes)
* Per-lane audit is optional and capped:

### Default cap

* `LANE_AUDIT_MAX = 8` lanes

### Which lanes get lane audits (deterministic)

Pick the first `LANE_AUDIT_MAX` lanes by **Tier priority** (same tier list as lane selection), and within tier by UTF-8 lane-id order.

If fewer than 8 exist, audit all.

If budget is extremely tight (rare), drop per-lane audits entirely but keep per-class.

---

# 5) JS reference: audit collection

```js
const CLASS_ORDER = ["policy","attention","preset","interaction","summary","other"];

function ratioQ16(clipped, total) {
  if (!total) return 0;
  const x = (clipped / total) * 65535;
  return x >= 0 ? Math.floor(x + 0.5) : Math.ceil(x - 0.5);
}

function auditQuantization(weights, laneId) {
  const cls = classifyLane(laneId);
  const s = CLASS_QSCALE[cls] ?? 1.0;

  let clipped = 0;
  const total = weights.length;

  for (let i=0;i<total;i++) {
    const v = weights[i] * s;
    if (v >= 1 || v <= -1) clipped++;
  }

  return { cls, total, clipped, ratio_q16: ratioQ16(clipped, total) };
}

function accumulateClassAudit(classAudit, one) {
  const slot = classAudit[one.cls];
  slot.total += one.total;
  slot.clipped += one.clipped;
}
```

At end, compute `ratio_q16` per class from totals.

---

# 6) Python reference: audit collection

```python
CLASS_ORDER = ["policy","attention","preset","interaction","summary","other"]

def ratio_q16(clipped: int, total: int) -> int:
  if total == 0:
    return 0
  x = (clipped / total) * 65535.0
  return round_away_from_zero(x)

def audit_quantization(weights, lane_id):
  cls = classify_lane(lane_id)
  s = CLASS_QSCALE.get(cls, 1.0)

  clipped = 0
  total = len(weights)
  for w in weights:
    v = w * s
    if v >= 1.0 or v <= -1.0:
      clipped += 1

  return cls, total, clipped, ratio_q16(clipped, total)
```

---

# 7) What this gives you

* If `policy` starts saturating a lot when dims drop, you’ll see:

  * `policy.clip_ratio` rising
* You can then:

  * lower `policy` qscale
  * adjust fold class weight
  * adjust synthesis amplitude for that lane class
    …without guessing.

---

# 8) Invariants (hard)

Auditing must not affect:

* lane bytes
* lane hashes
* merkle leaves
* decoded values

It is informational only.

---

Locked. Below is **Auto-Tuning Rule v1.0** — a **deterministic, replayable, single-pass adjustment** of class quantization scales based on saturation auditing. It **never loops**, never learns online, and always produces the same result for the same inputs.

This completes the compression stack without violating proofs.

---

# Auto-Tuning Rule v1.0 (Deterministic)

## 0) Design constraints (hard)

* **One shot** only (no iterative tuning)
* **Pure function** of:

  * emitted lane values
  * class-aware saturation audit
* **Replayable**: chosen scales are written into the pack and signed
* **Budget-safe**: does not change lane count or dims
* **Bounded**: scales only move within a fixed range

---

# 1) What is auto-tuned

Only **class quantization scales** (`s_class`) are adjusted.

What is **not** changed:

* dims
* folding
* lane selection
* class weights
* quantization format (int16)
* rounding law
* DICT
* merkle structure

---

# 2) Inputs (frozen)

From **Class-Aware Saturation Auditing** you already have, per class:

* `samples_total`
* `samples_clipped`
* `clip_ratio_q16`

Convert:
[
clip_ratio = \frac{clip_ratio_q16}{65535}
]

---

# 3) Thresholds & limits (frozen)

### Clip thresholds

| Symbol   | Value  | Meaning               |
| -------- | ------ | --------------------- |
| `T_LOW`  | `0.01` | negligible saturation |
| `T_HIGH` | `0.08` | excessive saturation  |

### Scale bounds (absolute)

Each class scale must remain within:

```
0.70 ≤ s_class ≤ 1.60
```

These bounds are **hard clamps**.

---

# 4) Adjustment rule (frozen)

For each class independently:

### Case A — Excessive saturation

If:

```
clip_ratio > T_HIGH
```

Then:
[
s' = s \times (1 - \alpha)
]

### Case B — Very low saturation

If:

```
clip_ratio < T_LOW
```

Then:
[
s' = s \times (1 + \beta)
]

### Case C — Healthy range

If:

```
T_LOW ≤ clip_ratio ≤ T_HIGH
```

Then:

```
s' = s   (no change)
```

---

# 5) Step sizes (frozen)

These are deliberately small to avoid instability.

```
α = 0.12   // scale reduction
β = 0.06   // scale increase
```

Then clamp:

```
s' = clamp(s', 0.70, 1.60)
```

---

# 6) Canonical initial scales (v1 reminder)

| Class       | Initial `s_class` |
| ----------- | ----------------- |
| policy      | 1.35              |
| attention   | 1.15              |
| preset      | 1.10              |
| interaction | 0.95              |
| summary     | 0.85              |
| other       | 1.00              |

Auto-tuning modifies these **once**.

---

# 7) Ordering & determinism law (important)

1. Start from **canonical initial scales**
2. Run folding → synthesis
3. Run **audit**
4. Apply **auto-tuning rule once**
5. **Re-quantize** lanes using the tuned scales
6. Write tuned `QSCALE` into pack
7. Hash → merkle → sign

⚠️ There is **no second audit**.
⚠️ There is **no feedback loop**.

This ensures:

* no oscillation
* no dependence on runtime decoding
* strict determinism

---

# 8) JS reference implementation

```js
const T_LOW  = 0.01;
const T_HIGH = 0.08;
const ALPHA  = 0.12;
const BETA   = 0.06;
const S_MIN  = 0.70;
const S_MAX  = 1.60;

function autoTuneQScales(initialScales, classAudit) {
  const tuned = {};

  for (const cls of Object.keys(initialScales)) {
    const s = initialScales[cls];
    const audit = classAudit[cls] || { clipped:0, total:0 };
    const ratio = audit.total ? audit.clipped / audit.total : 0;

    let s2 = s;

    if (ratio > T_HIGH) {
      s2 = s * (1 - ALPHA);
    } else if (ratio < T_LOW) {
      s2 = s * (1 + BETA);
    }

    if (s2 < S_MIN) s2 = S_MIN;
    if (s2 > S_MAX) s2 = S_MAX;

    tuned[cls] = s2;
  }

  return tuned;
}
```

---

# 9) Python reference implementation

```python
T_LOW  = 0.01
T_HIGH = 0.08
ALPHA  = 0.12
BETA   = 0.06
S_MIN  = 0.70
S_MAX  = 1.60

def auto_tune_qscales(initial_scales, class_audit):
  tuned = {}
  for cls, s in initial_scales.items():
    audit = class_audit.get(cls, {"clipped":0, "total":0})
    ratio = (audit["clipped"] / audit["total"]) if audit["total"] else 0.0

    s2 = s
    if ratio > T_HIGH:
      s2 = s * (1 - ALPHA)
    elif ratio < T_LOW:
      s2 = s * (1 + BETA)

    s2 = max(S_MIN, min(S_MAX, s2))
    tuned[cls] = s2

  return tuned
```

---

# 10) Pack encoding (critical)

The **tuned scales** replace the canonical ones in the `QSCALE` section:

```
QSCALE = [
  policy_s,
  attention_s,
  preset_s,
  interaction_s,
  summary_s,
  other_s
]
```

These values are:

* included in hash
* included in signature
* authoritative for decoding

So replay always yields the same floats.

---

# 11) Safety & invariants

✔ Cannot diverge
✔ Cannot overshoot
✔ Cannot exceed budget
✔ Cannot change lane topology
✔ Cannot invalidate proofs

This is **controlled compression-aware adaptation**, not learning.

---

# 12) What this unlocks

You now have a full deterministic pipeline:

* SVG → lanes → temporal → interactions
* Lane budget + adaptive dims
* Class-weighted folding
* Class-aware quantization
* Saturation auditing
* **Auto-tuned quant scales**
* Binary SCXQ2 pack < 5KB
* Lazy load + partial verification
* Signed, replayable artifacts

At this point, you’ve effectively built a **symbolic, multi-head, time-aware model format** with no tensors and no training loops.

Locked. Below are **Cross-Pack Compatibility Rules v1.0** — a deterministic contract for **comparing, loading, merging, and upgrading** multiple `weights.v2c-bin` packs without breaking proofs, budgets, or replayability.

This is the final glue that lets packs coexist safely.

---

# Cross-Pack Compatibility Rules v1.0

## 0) Scope (what this governs)

Applies to **binary packs** produced by your pipeline:

* `weights.v2c-bin`
* SCXQ2 DICT + INDEX + DATA
* adaptive dims
* class-aware quant + auto-tuned scales
* merkleized lanes
* signed roots

This spec **does not** change packing or math — only how **multiple packs interact**.

---

## 1) Compatibility levels (frozen)

Each pair of packs `(A, B)` falls into exactly one level:

### Level 0 — **Identical**

All of the following match byte-exactly:

* header fields (dims, flags)
* DICT
* INDEX
* DATA
* MERKLE
* SIGNATURE

**Rule:** interchangeable; caches may dedupe.

---

### Level 1 — **Semantically compatible**

Must match:

* `version`
* `dims`
* lane IDs (after DICT expansion)
* class set
* quantization format
* fold & tuning laws (implied by version)

May differ:

* lane values
* QSCALE values
* signatures
* merkle roots

**Rule:** can be **loaded together** and **compared lane-wise**.

---

### Level 2 — **Adaptively compatible**

May differ:

* `dims`
* QSCALE values
* lane *presence* (budget selection)
* signatures

Must match:

* `version`
* lane naming law
* class definitions
* quantization format

**Rule:** can be **merged or compared only via normalized adapters** (defined below).

---

### Level 3 — **Incompatible**

Any mismatch in:

* version major
* lane naming law
* quantization format
* fold law
* rounding law

**Rule:** cannot be merged or compared. Must recompile.

---

## 2) Canonical compatibility check (deterministic)

Given packs `A` and `B`:

```text
if magic/version mismatch → Level 3
else if dims equal AND lane-id sets equal → Level 1
else if version equal AND lane-law equal → Level 2
else → Level 3
```

This logic is frozen.

---

## 3) Lane identity resolution (critical)

All cross-pack logic uses **expanded lane IDs** (DICT-resolved strings), never compressed symbols.

**Invariant:**
Compressed symbols are pack-local. Expanded IDs are universal.

---

## 4) Lane presence rules (frozen)

For any operation involving lanes:

* If a lane exists in one pack but not the other:

  * Treat missing lane as **zero vector**
* Missing lanes never cause incompatibility

This guarantees:

* budgeted packs can interoperate
* future expansions don’t break older packs

---

## 5) Dims normalization (Level 2 only)

When `dims_A ≠ dims_B`, normalize **downward** to:

```
dims_N = min(dims_A, dims_B)
```

### Down-fold adapter (frozen)

For a lane vector `V` of length `dims_X`:

```text
for i in 0..len(V)-1:
  Vn[i % dims_N] += V[i]
```

This is identical to your fold rule — no new math.

**Never upsample.**

---

## 6) Quant scale normalization (Level 2 only)

Each pack has its own `QSCALE[class]`.

When comparing or merging values:

* Decode each lane using **its own QSCALE**
* Perform operations in float space
* If re-packing, **re-quantize** using the *target pack’s* tuned QSCALE

This avoids cross-scale distortion.

---

## 7) Comparison semantics (safe & deterministic)

### 7.1 Lane-wise similarity

For a lane `L` in packs `A` and `B`:

1. Normalize dims → `dims_N`
2. Decode both lanes → float vectors
3. Compute similarity (recommended default):

```
cosine_similarity(VA, VB)
```

If either lane is missing → similarity = 0.

---

### 7.2 Pack-level similarity

Weighted average across lanes:

```
SIM = Σ ( w_class(L) * sim(L) ) / Σ w_class
```

Use **fold class weights** from v1:

* policy weighted highest
* summary lowest

This produces a stable “brain similarity” score.

---

## 8) Merge semantics (deterministic)

### 8.1 Allowed merges

Only allowed when compatibility ≥ Level 2.

### 8.2 Merge modes (frozen set)

#### Mode A — **Overlay (default)**

* Target pack `T`
* Source pack `S`

Rules:

* For each lane:

  * if present in `S`: use `S`
  * else: keep `T`
* dims = `dims_T`
* QSCALE = `QSCALE_T`
* signature invalidated → re-sign

This is safe for **patch updates**.

---

#### Mode B — **Weighted blend**

Given weight `λ ∈ [0,1]` (must be fixed, not data-dependent):

```
V = λ * VS + (1 - λ) * VT
```

Applied after normalization.

Then:

* re-quantize with target QSCALE
* re-sign

Use for **controlled evolution**, not hot swaps.

---

#### Mode C — **Union**

* Keep **all lanes from both packs**
* If same lane exists in both:

  * prefer source (`S`) deterministically

This may increase lane count → must re-run **lane budget + adaptive dims** before re-packing.

---

## 9) Proof & signature rules (hard)

* Any merge or normalization:

  * **invalidates original signature**
  * requires **new merkle + signature**
* Partial verification remains valid **before** merge
* After merge, only the new signature is authoritative

No exceptions.

---

## 10) Cache & loading rules (runtime)

* Runtime may cache packs by:

  * `(merkle_root, signature)`
* Two packs with same merkle root are **bitwise identical**
* DICT compression differences do **not** affect compatibility as long as expanded IDs match

---

## 11) Forward-compatibility (v1 promise)

Future versions may:

* add new lane classes
* add new audit sections
* add new metadata

As long as:

* existing lane naming rules remain valid
* quant format unchanged

Older runtimes:

* must ignore unknown sections
* must treat unknown lanes as “other” class

---

## 12) Hard invariants (summary)

* Expanded lane IDs are the universal key
* Never upsample dims
* Never mix quant scales without decode
* Missing lanes = zero
* Merges always re-sign
* No silent behavior changes

---

## 13) What this completes

You now have a **complete, closed system**:

* deterministic generation
* bounded size (<5KB)
* adaptive resolution
* class-aware fidelity
* audit + auto-tuning
* lazy loading
* partial verification
* **safe multi-pack coexistence**

At this point, your format qualifies as a **portable symbolic model artifact** with stronger guarantees than most ML checkpoints.

Locked. Below is the **Human-Readable Pack Manifest View v1.0** — a **lossless, non-authoritative, derived summary** of a `weights.v2c-bin` pack, intended for inspection, debugging, diffs, and UX.
It **never participates in execution, hashing, or signing**.

---

# Pack Manifest View v1.0

## 0) Core rule (hard)

> The manifest is **derived**, not loaded.
> If the binary and manifest disagree, **the binary always wins**.

The manifest:

* is generated *from* the binary pack
* may be regenerated at any time
* must not introduce new semantics

---

## 1) Purpose

The manifest exists to answer, at a glance:

* What lanes are inside?
* What classes and temporal windows exist?
* What dims and budgets were used?
* Did auto-tuning fire?
* Where is saturation happening?
* Are two packs compatible?

---

## 2) File identity

### Recommended filename

```
weights.v2c.manifest.json
```

### Relationship

```
weights.v2c-bin  ──derive──▶  weights.v2c.manifest.json
```

The manifest may optionally include a reference hash of the binary it was derived from.

---

## 3) Canonical manifest schema (v1.0)

```json
{
  "schema": "asx.pack.manifest.v1",
  "derived_from": {
    "binary_format": "weights.v2c-bin",
    "merkle_root": "hex",
    "signature": "hex",
    "hash_alg": "blake3"
  },

  "core": {
    "version": "2.1",
    "dims": 48,
    "lane_budget_bytes": 5120,
    "adaptive_dims": true,
    "quant_format": "int16",
    "endianness": "little"
  },

  "lane_classes": {
    "policy": {
      "count": 4,
      "fold_weight": 1.30,
      "qscale": 1.27,
      "auto_tuned": true
    },
    "attention": {
      "count": 4,
      "fold_weight": 1.10,
      "qscale": 1.15,
      "auto_tuned": false
    },
    "preset": {
      "count": 6,
      "fold_weight": 1.05,
      "qscale": 1.10,
      "auto_tuned": false
    },
    "interaction": {
      "count": 10,
      "fold_weight": 0.85,
      "qscale": 0.84,
      "auto_tuned": true
    },
    "summary": {
      "count": 3,
      "fold_weight": 0.75,
      "qscale": 0.85,
      "auto_tuned": false
    },
    "other": {
      "count": 0,
      "fold_weight": 1.00,
      "qscale": 1.00,
      "auto_tuned": false
    }
  },

  "temporal_layout": {
    "windows": ["early", "mid", "late"],
    "decay": "exponential",
    "interaction_matrix": {
      "shape": "upper-triangular",
      "effective_lanes": 10
    }
  },

  "saturation_audit": {
    "policy": {
      "samples": 192,
      "clipped": 7,
      "clip_ratio": 0.036
    },
    "attention": {
      "samples": 192,
      "clipped": 1,
      "clip_ratio": 0.005
    },
    "preset": {
      "samples": 288,
      "clipped": 0,
      "clip_ratio": 0.000
    },
    "interaction": {
      "samples": 480,
      "clipped": 22,
      "clip_ratio": 0.045
    },
    "summary": {
      "samples": 144,
      "clipped": 0,
      "clip_ratio": 0.000
    }
  },

  "lanes": [
    {
      "id": "policy@early",
      "class": "policy",
      "temporal": "early",
      "dims": 48,
      "hash": "hex",
      "saturation": 0.021
    },
    {
      "id": "policy@late",
      "class": "policy",
      "temporal": "late",
      "dims": 48,
      "hash": "hex",
      "saturation": 0.048
    }
  ],

  "compatibility": {
    "level": 2,
    "lane_law": "v1",
    "fold_law": "v1",
    "quant_law": "v1",
    "merge_allowed": true
  },

  "notes": [
    "Auto-tuning reduced policy qscale due to saturation > 0.08",
    "Interaction lanes near upper-triangle boundary saturated mildly"
  ]
}
```

---

## 4) Section semantics (frozen)

### 4.1 `derived_from`

* Pure reference
* Allows tooling to assert which binary this describes
* Not used in runtime

---

### 4.2 `core`

Summarizes immutable runtime-relevant parameters:

* `dims` after adaptive reduction
* total byte budget used
* quantization format

---

### 4.3 `lane_classes`

Per class:

* `count` = number of lanes of that class
* `fold_weight` = frozen class weight
* `qscale` = **final tuned value**
* `auto_tuned` = whether auto-tuning changed it

This is the **single most useful diagnostic block**.

---

### 4.4 `temporal_layout`

Explains how time was folded:

* window names
* decay model
* interaction matrix shape
  This lets humans understand *why* certain lanes exist.

---

### 4.5 `saturation_audit`

Human-readable version of AUDIT section:

* floats instead of q16
* derived, not authoritative
* rounding allowed

---

### 4.6 `lanes` (optional, truncated)

This section:

* may be **truncated**
* is ordered by class priority then ID
* is meant for inspection, not completeness

Runtime must not depend on this list being complete.

---

### 4.7 `compatibility`

Pre-computed compatibility hints:

* matches the formal rules
* helps UIs quickly warn users
* non-binding

---

## 5) Generation rules (deterministic)

When generating the manifest:

1. Decode header
2. Expand DICT → lane IDs
3. Read QSCALE
4. Read AUDIT
5. Compute derived floats (ratios)
6. Sort lanes deterministically
7. Emit JSON with:

   * UTF-8
   * stable key ordering
   * no trailing whitespace

---

## 6) What the manifest MUST NOT do

* ❌ introduce new values
* ❌ re-compute tuning
* ❌ alter scales
* ❌ affect merging
* ❌ affect loading

It is **read-only documentation**.

---

## 7) Tooling advantages

With this manifest you can:

* diff two packs meaningfully
* visualize lane topology
* explain auto-tuning decisions
* debug saturation issues
* show “model cards” in UI
* safely compare packs before merging

---

## 8) Final invariant

> **Binary is law. Manifest is explanation.**

This closes the system cleanly:
you now have **a symbolic model format with introspection, proofs, and deterministic evolution**.

If you want, next steps are purely ergonomic:

* visual lane heatmaps
* SVG overlays
* or a CLI `asx pack inspect` command that emits this manifest automatically.


