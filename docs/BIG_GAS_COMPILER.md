Perfect — the distinction is **correct and necessary**:

* **GASLLM** → cloud / Sheets / GAS-hosted compiler + packs
* **LOCAL branch** → same compiler **logic**, but **no GAS, no Sheets, no network**

So what you actually want is:

> **ONE canonical compiler design**
> with **two execution shells**
> (GAS + Local JS/Python)

Below is the **exact sketch of `BIG_GAS_COMPILER.gs`**, written so it can be:

* dropped into GAS **as-is**
* or mechanically ported to **local JS / Python** with zero semantic drift

This is **not pseudocode** — it’s a real, structured compiler file.

---

# `BIG_GAS_COMPILER.gs` (Canonical Compiler Kernel)

> **Invariant:**
> This file is a *compiler*, not a runtime, not a model.

---

## 0. File contract (top of file)

```js
/**
 * BIG_GAS_COMPILER.gs
 * ===================
 *
 * ASX Symbolic Weight Compiler
 *
 * ROLE:
 * - Compile structured data packs → results.json weights
 *
 * NON-GOALS:
 * - ❌ No inference
 * - ❌ No training
 * - ❌ No randomness
 * - ❌ No model execution
 *
 * DETERMINISM:
 * - Same inputs → same outputs (bitwise stable)
 *
 * PORTABILITY:
 * - GAS canonical
 * - Must port 1:1 to local JS / Python
 */
```

---

## 1. Global constants (frozen law)

```js
const COMPILER_VERSION = "ASX_BIG_COMPILER_v1.0.0";
const WEIGHTS_VERSION  = "weights.v1";

const DEFAULT_DIMS = 64;

const EPS = 1e-9;
```

---

## 2. Entry point (single public API)

This is the **only function external code should call**.

```js
function compileWeightPack(input) {
  // @Pop — ingest
  const pack = ingestInput(input);

  // @Wo — normalize state
  const normalized = normalizePack(pack);

  // @Sek — fold + synthesize
  const folded = foldSignals(normalized.signals, normalized.dims);
  const weights = synthesizeWeights(folded, normalized.config);

  // @Collapse — emit
  return emitResultsJson(weights, normalized.meta);
}
```

This maps cleanly to your ASX phases.

---

## 3. Ingest layer (local vs GAS friendly)

```js
function ingestInput(input) {
  /**
   * input can be:
   * - plain JS object (LOCAL)
   * - parsed JSON (GAS)
   * - sheet-derived object (GASLLM branch)
   */

  if (!input || typeof input !== "object") {
    throw new Error("Invalid compiler input");
  }

  return {
    signals: input.signals || {},
    config:  input.config  || {},
    meta:    input.meta    || {}
  };
}
```

> **Local model note:**
> For local builds, `input` comes from:
>
> * JSON file
> * CLI args
> * cached results.json
>   No changes needed.

---

## 4. Normalization layer (lawful math only)

```js
function normalizePack(pack) {
  const dims = Number(pack.config.dims || DEFAULT_DIMS);

  const normSignals = {};
  for (const key in pack.signals) {
    normSignals[key] = clamp01(pack.signals[key]);
  }

  return {
    dims,
    signals: normSignals,
    config: {
      bias: Number(pack.config.bias || 0.0),
      seed: String(pack.config.seed || "ASX::LOCAL"),
      amplitude: Number(pack.config.amplitude || 0.08),
      styleAmplitude: Number(pack.config.styleAmplitude || 0.06)
    },
    meta: {
      compiler: COMPILER_VERSION,
      source: pack.meta.source || "local",
      timestamp: pack.meta.timestamp || null
    }
  };
}
```

---

## 5. Folding layer (signal → vector)

This is the **heart of the system**.

```js
function foldSignals(signals, dims) {
  const vec = new Array(dims).fill(0);

  let i = 0;
  for (const key in signals) {
    const v = signals[key];
    const idx = i % dims;
    vec[idx] += v;
    i++;
  }

  return vec;
}
```

Deterministic. Order-preserving. Portable.

---

## 6. Weight synthesis (symbolic math only)

```js
function synthesizeWeights(folded, config) {
  const dims = folded.length;
  const base = new Array(dims);
  const style = new Array(dims);

  const seedHash = stableHash(config.seed);

  for (let i = 0; i < dims; i++) {
    const phase = (seedHash % 997) * 0.0001;

    base[i] =
      Math.sin(i * 0.37 + phase) *
      config.amplitude *
      (1 + folded[i]);

    style[i] =
      Math.cos(i * 0.21 + phase) *
      config.styleAmplitude;
  }

  return {
    version: WEIGHTS_VERSION,
    dims,
    bias: config.bias,
    W: roundVec(base),
    styleW: roundVec(style)
  };
}
```

---

## 7. Emit layer (results.json)

```js
function emitResultsJson(weights, meta) {
  return {
    ...weights,
    meta: {
      ...meta,
      law: "deterministic",
      note: "Compiled symbolic weights"
    }
  };
}
```

This object is **directly serializable**.

---

## 8. Math utilities (portable)

```js
function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function roundVec(vec) {
  return vec.map(v => Math.round(v * 1e6) / 1e6);
}

function stableHash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
  }
  return h >>> 0;
}
```

No crypto libs → works in GAS **and** local.

---

## 9. Example local invocation (conceptual)

```js
const pack = compileWeightPack({
  signals: {
    joy: 0.72,
    anger: 0.11,
    clarity: 0.84
  },
  config: {
    dims: 64,
    bias: 0.05,
    seed: "LOCAL_DOOM_BRAIN"
  },
  meta: {
    source: "local_json"
  }
});

JSON.stringify(pack, null, 2);
```

---

## Why this satisfies *both* branches

### GASLLM branch

* Replace `ingestInput()` source with Sheets adapters
* Everything else stays frozen

### Local model branch

* Feed JSON / SVG-derived stats
* Same compiler
* Same output shape
* Same results.json semantics

---

## Key invariant (important)

> **No branch divergence is allowed in compiler logic**
> Only the **data ingestion layer differs**

This prevents:

* drift
* training hell
* “why does local behave differently?”

---

## Next locks (recommended)

If you want to proceed cleanly, next we should lock **one** of these:

1. **Local data-pack schema** (`local_pack.v1.json`)
2. **SVG → signals adapter** (geometry → scalar fields)
3. **Proof hash envelope** for `results.json`
4. **Lane-aware weights** (`W.policy`, `W.attention`, …)

Tell me which one you want to freeze next and I’ll write it at the same level of precision.
