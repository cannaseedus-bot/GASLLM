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

Say which to freeze next.
