# SCXPI Model Pack (GAS)

This document captures a buildable, GAS-friendly **symbolic/compressed model** pattern that treats SCXQ2 + π + rules + geometry as a deterministic runtime.

## What “Symbolic / Compressed Model” means in practice

Instead of loading tensor weights, you ship:

1. **A tiny immutable codex** (SCXQ2 dictionary + field map)
2. **A policy / rule graph** (deterministic transforms, gates, routing)
3. **A geometry feature layer** (shape metrics, adjacency, containment, distances)
4. **A π-evaluator** (math-only scoring and collapse)

So “inference” becomes:

```
input → normalize → features → score → pick action → emit JSON
```

No tensors required.

## The minimal executable contract (Model ABI for GAS)

Think of the “model” as a pure function:

```js
output = model.eval({ input, state, codex, rules, geom })
```

Where output is always **API JSON**.

### Output shape (example)

- `intent`: what to do
- `confidence`: π score
- `actions`: list of deterministic ops
- `proof`: audit + hash inputs

## A small but real example (SCXQ2 + π + rules)

### 1) “Model pack” object (what gets stored / shipped)

- tiny, stable, replayable
- works in GAS

```js
const MODEL_PACK = {
  "@kind": "scxpi.model.v1",
  "@id": "asx://model/scxpi/v1",
  "@v": 1,
  "@codex": {
    // SCXQ2: dictionary lanes (tiny example)
    "dict": { "D0": ["ALLOW", "DENY", "ROUTE", "TAG", "RISK", "SAFE"] },
    "fields": { "intent": 0, "risk": 1, "route": 2, "tags": 3 }
  },
  "@pi": {
    // weights used by π scoring (not tensors, just constants)
    "weights": { "risk": 2.0, "match": 1.0, "penalty": 3.0 },
    "threshold": 2.5
  },
  "@rules": [
    // rule = predicate(input/features) -> emits atoms
    { "if": { "contains": ["text", "password"] }, "emit": ["RISK", 0.9] },
    { "if": { "contains": ["text", "apikey"] },   "emit": ["RISK", 0.9] },
    { "if": { "contains": ["text", "hello"] },    "emit": ["TAG", "greeting"] },
    { "if": { "always": true },                   "emit": ["ROUTE", "api"] }
  ]
};
```

### 2) GAS-side evaluator (pure, deterministic)

This is the “inference engine” — tiny and fast.

```js
function scxpi_eval(input, pack) {
  const text = String(input.text || "");
  const feats = { risk: 0, tags: [], route: "api", intent: "ALLOW" };

  // Rules
  for (const r of pack["@rules"]) {
    if (ruleMatch(r["if"], { text })) {
      const [op, val] = r.emit;
      if (op === "RISK") feats.risk = Math.max(feats.risk, Number(val));
      if (op === "TAG") feats.tags.push(String(val));
      if (op === "ROUTE") feats.route = String(val);
    }
  }

  // π scoring (toy but real): score = w.risk*risk + w.match*(tags>0) - w.penalty*(riskyTag)
  const w = pack["@pi"].weights;
  const hasTags = feats.tags.length > 0 ? 1 : 0;
  const riskyTag = feats.tags.includes("greeting") ? 0 : 0; // placeholder
  const score = (w.risk * feats.risk) + (w.match * hasTags) - (w.penalty * riskyTag);

  feats.intent = score >= pack["@pi"].threshold ? "ALLOW" : "DENY";

  return {
    intent: feats.intent,
    confidence: clamp01(score / (pack["@pi"].threshold * 2)),
    route: feats.route,
    tags: feats.tags,
    features: feats,
    proof: {
      model: pack["@id"],
      v: pack["@v"],
      score,
      threshold: pack["@pi"].threshold
    }
  };
}

function ruleMatch(cond, ctx) {
  if (cond.always) return true;
  if (cond.contains) {
    const [field, needle] = cond.contains;
    return String(ctx[field] || "").toLowerCase().includes(String(needle).toLowerCase());
  }
  return false;
}

function clamp01(x) { return Math.max(0, Math.min(1, x)); }
```

This is already enough to:

- route requests
- block unsafe content
- tag + classify
- produce a proofable score

…and it runs comfortably inside GAS.

## Where “Geometry” plugs in (the real power)

Geometry is where this becomes *not just rules*.

You define features like:

- `bbox.area`
- `path.count`
- `edge_density`
- `symmetry_score`
- `overlap_ratio`
- `adjacency_graph_hash`

Then rules can match on geometry:

- “if overlap_ratio > 0.85 → classify as dense”
- “if symmetry_score high → likely icon”
- “if path.count small and corners ~4 → cube”

And π collapses the best label/action.

That’s how you get **SVG intelligence** without tensors.

## Why this beats `.safetensors` for this stack

- **Deterministic**: every output is replayable
- **Auditable**: rule hits + π score + proof hash
- **Tiny**: KB–MB packs instead of GB weights
- **Upgradeable**: ship deltas as SCXQ2 streams
- **GAS-native**: runs where tokens can’t

## SCXPI model schema (v1)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-model/v1",
  "@v": 1,
  "title": "SCXPI_MODEL_v1 — Symbolic Model Pack for GAS",
  "type": "object",
  "required": ["@kind", "@id", "@v", "@codex", "@pi", "@rules"],
  "properties": {
    "@kind": { "const": "scxpi.model.v1" },
    "@id": { "type": "string", "minLength": 8 },
    "@v": { "type": "integer", "minimum": 1 },
    "@meta": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "author": { "type": "string" },
        "created_utc": { "type": "string" },
        "notes": { "type": "string" }
      },
      "additionalProperties": true
    },
    "@codex": {
      "type": "object",
      "required": ["dict", "fields"],
      "properties": {
        "dict": {
          "type": "object",
          "description": "SCXQ2-like dictionary lanes (D0..Dn) storing token strings.",
          "additionalProperties": {
            "type": "array",
            "items": { "type": "string" }
          }
        },
        "fields": {
          "type": "object",
          "description": "Field name -> integer slot mapping (optional, used by compressors).",
          "additionalProperties": { "type": "integer" }
        }
      },
      "additionalProperties": false
    },
    "@pi": {
      "type": "object",
      "required": ["weights", "threshold"],
      "properties": {
        "weights": {
          "type": "object",
          "description": "Named scalar weights for π scoring.",
          "additionalProperties": { "type": "number" }
        },
        "threshold": { "type": "number" },
        "clamp": { "type": "boolean", "default": true }
      },
      "additionalProperties": false
    },
    "@geom": {
      "type": "object",
      "description": "Optional geometry feature contract (names + expected ranges).",
      "properties": {
        "features": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name"],
            "properties": {
              "name": { "type": "string" },
              "min": { "type": "number" },
              "max": { "type": "number" }
            },
            "additionalProperties": false
          }
        }
      },
      "additionalProperties": false
    },
    "@rules": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["if", "emit"],
        "properties": {
          "id": { "type": "string" },
          "if": {
            "type": "object",
            "description": "Predicate block",
            "properties": {
              "always": { "type": "boolean" },
              "contains": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": { "type": "string" },
                "description": "[fieldName, needle]"
              },
              "eq": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {},
                "description": "[lhs, rhs] (primitive compare)"
              },
              "gt": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {},
                "description": "[lhs, rhs] numeric"
              },
              "lt": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {},
                "description": "[lhs, rhs] numeric"
              },
              "and": {
                "type": "array",
                "items": { "type": "object" }
              },
              "or": {
                "type": "array",
                "items": { "type": "object" }
              },
              "not": { "type": "object" }
            },
            "additionalProperties": false
          },
          "emit": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "description": "[OP, VALUE]",
            "items": {}
          },
          "priority": { "type": "integer", "default": 0 }
        },
        "additionalProperties": false
      }
    },
    "@seal": {
      "type": "object",
      "description": "Optional integrity seal. If present, runtime can enforce it.",
      "properties": {
        "alg": { "type": "string", "enum": ["sha256"] },
        "canonical": { "type": "string", "description": "Canonical JSON string hash input (optional)"},
        "hash": { "type": "string", "description": "hex sha256 of canonical form" }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

## SCXPI runtime (GAS)

```javascript
/**
 * SCXPI_MODEL_v1 — GAS Runtime
 * ------------------------------------------------------------
 * Deterministic symbolic "model" evaluator:
 * input (text + optional geom) -> rules -> π score -> intent/actions -> proof
 *
 * Drop this into Code.gs (or scxpi.runtime.gs).
 * Works in plain Google Apps Script.
 */

// ============================================================
// Public API
// ============================================================

/**
 * Evaluate an input against a SCXPI model pack.
 * @param {Object} pack SCXPI model pack (scxpi.model.v1)
 * @param {Object} input { text?: string, geom?: object, state?: object }
 * @param {Object=} opts { enforceSeal?: boolean, nowUtc?: string }
 * @returns {Object} result
 */
function SCXPI_EVAL(pack, input, opts) {
  opts = opts || {};
  SCXPI_VERIFY_PACK(pack, { enforceSeal: !!opts.enforceSeal });

  var ctx = SCXPI_BUILD_CTX(input || {});
  var hits = [];
  var feats = SCXPI_INIT_FEATURES();

  // Sort rules deterministically: priority desc, then id asc, then stable index asc
  var rules = (pack["@rules"] || []).map(function (r, i) {
    var rr = JSON.parse(JSON.stringify(r));
    rr.__i = i;
    rr.priority = (typeof rr.priority === "number") ? rr.priority : 0;
    rr.id = (typeof rr.id === "string" && rr.id.length) ? rr.id : ("rule_" + i);
    return rr;
  }).sort(function (a, b) {
    if (a.priority !== b.priority) return b.priority - a.priority;
    if (a.id !== b.id) return a.id < b.id ? -1 : 1;
    return a.__i - b.__i;
  });

  // Apply rules
  for (var i = 0; i < rules.length; i++) {
    var r = rules[i];
    if (SCXPI_RULE_MATCH(r["if"], ctx)) {
      var op = r.emit[0];
      var val = r.emit[1];
      SCXPI_APPLY_EMIT(feats, op, val);
      hits.push({ id: r.id, op: op, val: val, priority: r.priority });
    }
  }

  // π score + collapse
  var pi = SCXPI_SCORE(pack["@pi"], feats, ctx);

  // Result
  var nowUtc = (typeof opts.nowUtc === "string" && opts.nowUtc) ? opts.nowUtc : SCXPI_NOW_UTC();
  var result = {
    "@kind": "scxpi.result.v1",
    model: { id: pack["@id"], v: pack["@v"] },
    input: { text_len: (ctx.text || "").length, geom_present: !!ctx.geom },
    intent: pi.intent,
    confidence: pi.confidence,
    score: pi.score,
    threshold: pack["@pi"].threshold,
    route: feats.route,
    tags: feats.tags.slice(0),
    features: {
      risk: feats.risk,
      allow: feats.allow,
      deny: feats.deny,
      route: feats.route,
      tag_count: feats.tags.length
    },
    hits: hits,
    proof: SCXPI_PROOF(pack, ctx, feats, pi, nowUtc)
  };

  return result;
}

/**
 * Verify pack structure + optional seal.
 * @param {Object} pack
 * @param {Object=} opts { enforceSeal?: boolean }
 */
function SCXPI_VERIFY_PACK(pack, opts) {
  opts = opts || {};
  if (!pack || typeof pack !== "object") throw new Error("SCXPI: pack must be object");
  if (pack["@kind"] !== "scxpi.model.v1") throw new Error("SCXPI: pack.@kind must be scxpi.model.v1");
  if (typeof pack["@id"] !== "string" || pack["@id"].length < 8) throw new Error("SCXPI: pack.@id invalid");
  if (typeof pack["@v"] !== "number" || pack["@v"] < 1) throw new Error("SCXPI: pack.@v invalid");

  if (!pack["@codex"] || typeof pack["@codex"] !== "object") throw new Error("SCXPI: pack.@codex missing");
  if (!pack["@codex"].dict || typeof pack["@codex"].dict !== "object") throw new Error("SCXPI: pack.@codex.dict missing");
  if (!pack["@codex"].fields || typeof pack["@codex"].fields !== "object") throw new Error("SCXPI: pack.@codex.fields missing");

  if (!pack["@pi"] || typeof pack["@pi"] !== "object") throw new Error("SCXPI: pack.@pi missing");
  if (!pack["@pi"].weights || typeof pack["@pi"].weights !== "object") throw new Error("SCXPI: pack.@pi.weights missing");
  if (typeof pack["@pi"].threshold !== "number") throw new Error("SCXPI: pack.@pi.threshold missing/invalid");

  if (!Array.isArray(pack["@rules"])) throw new Error("SCXPI: pack.@rules must be array");

  // Optional seal enforcement
  if (opts.enforceSeal && pack["@seal"]) {
    var seal = pack["@seal"];
    if (seal.alg !== "sha256") throw new Error("SCXPI: seal.alg must be sha256");
    if (typeof seal.hash !== "string" || seal.hash.length < 32) throw new Error("SCXPI: seal.hash missing");
    // Canonicalize without @seal to compute expected hash
    var canonical = SCXPI_CANONICAL_STRING(SCXPI_STRIP_SEAL(pack));
    var expected = SCXPI_SHA256_HEX(canonical);
    if (expected !== seal.hash) {
      throw new Error("SCXPI: seal mismatch expected=" + expected + " got=" + seal.hash);
    }
  }
}

// ============================================================
// Core: Context / Features / Rule Matching
// ============================================================

function SCXPI_BUILD_CTX(input) {
  var text = (typeof input.text === "string") ? input.text : "";
  var geom = (input.geom && typeof input.geom === "object") ? input.geom : null;
  var state = (input.state && typeof input.state === "object") ? input.state : {};
  return {
    text: text,
    text_lc: text.toLowerCase(),
    geom: geom,
    state: state
  };
}

function SCXPI_INIT_FEATURES() {
  return {
    risk: 0,
    allow: 0,
    deny: 0,
    route: "api",
    tags: []
  };
}

/**
 * Apply emitted atom deterministically.
 * Supported ops: RISK, TAG, ROUTE, ALLOW, DENY
 */
function SCXPI_APPLY_EMIT(feats, op, val) {
  op = String(op || "");
  if (op === "RISK") {
    var r = Number(val);
    if (!isFinite(r)) r = 0;
    if (r > feats.risk) feats.risk = r;
    return;
  }
  if (op === "TAG") {
    var t = String(val);
    if (t && feats.tags.indexOf(t) === -1) feats.tags.push(t);
    return;
  }
  if (op === "ROUTE") {
    feats.route = String(val || "api");
    return;
  }
  if (op === "ALLOW") {
    feats.allow += 1;
    return;
  }
  if (op === "DENY") {
    feats.deny += 1;
    return;
  }
}

/**
 * Predicate evaluation (deterministic, no side effects).
 */
function SCXPI_RULE_MATCH(cond, ctx) {
  if (!cond || typeof cond !== "object") return false;

  if (cond.always === true) return true;

  if (cond.contains) {
    // ["field","needle"] ; currently supports field "text" only, but allows "state.X" and "geom.X"
    var field = String(cond.contains[0] || "");
    var needle = String(cond.contains[1] || "").toLowerCase();
    var hay = String(SCXPI_RESOLVE_VALUE(field, ctx) || "").toLowerCase();
    return needle.length ? (hay.indexOf(needle) !== -1) : false;
  }

  if (cond.eq) {
    var a = SCXPI_RESOLVE_ANY(cond.eq[0], ctx);
    var b = SCXPI_RESOLVE_ANY(cond.eq[1], ctx);
    return SCXPI_PRIM_EQ(a, b);
  }

  if (cond.gt) {
    var ga = Number(SCXPI_RESOLVE_ANY(cond.gt[0], ctx));
    var gb = Number(SCXPI_RESOLVE_ANY(cond.gt[1], ctx));
    return isFinite(ga) && isFinite(gb) ? (ga > gb) : false;
  }

  if (cond.lt) {
    var la = Number(SCXPI_RESOLVE_ANY(cond.lt[0], ctx));
    var lb = Number(SCXPI_RESOLVE_ANY(cond.lt[1], ctx));
    return isFinite(la) && isFinite(lb) ? (la < lb) : false;
  }

  if (cond.and) {
    for (var i = 0; i < cond.and.length; i++) {
      if (!SCXPI_RULE_MATCH(cond.and[i], ctx)) return false;
    }
    return true;
  }

  if (cond.or) {
    for (var j = 0; j < cond.or.length; j++) {
      if (SCXPI_RULE_MATCH(cond.or[j], ctx)) return true;
    }
    return false;
  }

  if (cond.not) {
    return !SCXPI_RULE_MATCH(cond.not, ctx);
  }

  return false;
}

/**
 * Resolve a string path like:
 * - "text" -> ctx.text
 * - "state.user.role" -> ctx.state.user.role
 * - "geom.symmetry_score" -> ctx.geom.symmetry_score
 */
function SCXPI_RESOLVE_VALUE(path, ctx) {
  if (path === "text") return ctx.text;
  if (path === "text_lc") return ctx.text_lc;

  var parts = String(path || "").split(".");
  var root = parts.shift();
  var cur;
  if (root === "state") cur = ctx.state;
  else if (root === "geom") cur = ctx.geom || {};
  else return undefined;

  for (var i = 0; i < parts.length; i++) {
    if (!cur || typeof cur !== "object") return undefined;
    cur = cur[parts[i]];
  }
  return cur;
}

/**
 * Resolve either a literal (number/string/bool/null) or a string path.
 * If value is a string beginning with "$", treat as path reference: "$geom.x"
 */
function SCXPI_RESOLVE_ANY(v, ctx) {
  if (typeof v === "string" && v.length && v.charAt(0) === "$") {
    return SCXPI_RESOLVE_VALUE(v.slice(1), ctx);
  }
  return v;
}

function SCXPI_PRIM_EQ(a, b) {
  // Only primitive eq semantics (no deep compare)
  return (a === b);
}

// ============================================================
// π scoring + collapse
// ============================================================

function SCXPI_SCORE(piSpec, feats, ctx) {
  var w = piSpec.weights || {};
  var threshold = Number(piSpec.threshold);
  if (!isFinite(threshold)) threshold = 0;

  // Minimal, extensible scoring:
  // score = w.risk*risk + w.allow*allow - w.deny*deny + w.tag*(tag_count>0)
  var score = 0;
  score += (Number(w.risk) || 0) * (Number(feats.risk) || 0);
  score += (Number(w.allow) || 0) * (Number(feats.allow) || 0);
  score -= (Number(w.deny) || 0) * (Number(feats.deny) || 0);
  score += (Number(w.tag) || 0) * (feats.tags.length > 0 ? 1 : 0);

  // Optional geometry contribution if present:
  // If w.geom is present, add w.geom * (normalized geom.signal if provided)
  if (ctx.geom && typeof w.geom === "number") {
    var gs = Number(ctx.geom.signal);
    if (isFinite(gs)) score += w.geom * gs;
  }

  var intent = (score >= threshold) ? "ALLOW" : "DENY";
  var confidence = SCXPI_CONFIDENCE(score, threshold);

  return { score: score, intent: intent, confidence: confidence };
}

function SCXPI_CONFIDENCE(score, threshold) {
  // Deterministic bounded confidence in [0,1]
  // Use a soft ratio around threshold.
  var denom = Math.abs(threshold) + 1e-9;
  var x = (score - threshold) / denom; // centered
  // squashing: 0.5 + x/(2*(1+|x|))
  var conf = 0.5 + (x / (2 * (1 + Math.abs(x))));
  return Math.max(0, Math.min(1, conf));
}

// ============================================================
// Proof + canonicalization + hashing
// ============================================================

function SCXPI_PROOF(pack, ctx, feats, pi, nowUtc) {
  // Proof input is canonical JSON of:
  // {model_id, model_v, text_lc, geom_hash, feats, score, threshold}
  var geomHash = ctx.geom ? SCXPI_SHA256_HEX(SCXPI_CANONICAL_STRING(ctx.geom)) : null;

  var proofInput = {
    model_id: pack["@id"],
    model_v: pack["@v"],
    text_lc: ctx.text_lc,
    geom_hash: geomHash,
    feats: {
      risk: feats.risk,
      allow: feats.allow,
      deny: feats.deny,
      route: feats.route,
      tags: feats.tags.slice(0).sort() // canonical
    },
    score: pi.score,
    threshold: pack["@pi"].threshold
  };

  var canonical = SCXPI_CANONICAL_STRING(proofInput);
  var hash = SCXPI_SHA256_HEX(canonical);

  return {
    alg: "sha256",
    at_utc: nowUtc,
    input_hash: hash,
    canonical_len: canonical.length
  };
}

/**
 * Produce canonical JSON string:
 * - stable key ordering (lexicographic)
 * - stable array order preserved except where caller sorts
 * - no whitespace
 */
function SCXPI_CANONICAL_STRING(value) {
  return JSON.stringify(SCXPI_CANONICALIZE(value));
}

function SCXPI_CANONICALIZE(v) {
  if (v === null) return null;
  var t = typeof v;
  if (t === "number") {
    // Normalize -0 to 0
    return Object.is(v, -0) ? 0 : v;
  }
  if (t === "string" || t === "boolean") return v;

  if (Array.isArray(v)) {
    return v.map(SCXPI_CANONICALIZE);
  }

  if (t === "object") {
    var keys = Object.keys(v).sort();
    var out = {};
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      // Skip undefined
      if (typeof v[k] === "undefined") continue;
      out[k] = SCXPI_CANONICALIZE(v[k]);
    }
    return out;
  }

  // undefined / function / symbol => null (but generally should not exist)
  return null;
}

function SCXPI_SHA256_HEX(str) {
  var bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, str, Utilities.Charset.UTF_8);
  return SCXPI_BYTES_TO_HEX(bytes);
}

function SCXPI_BYTES_TO_HEX(bytes) {
  var hex = [];
  for (var i = 0; i < bytes.length; i++) {
    var b = bytes[i];
    if (b < 0) b += 256;
    var h = b.toString(16);
    if (h.length === 1) h = "0" + h;
    hex.push(h);
  }
  return hex.join("");
}

function SCXPI_STRIP_SEAL(pack) {
  // Clone and remove @seal
  var clone = JSON.parse(JSON.stringify(pack));
  if (clone["@seal"]) delete clone["@seal"];
  return clone;
}

function SCXPI_NOW_UTC() {
  // Deterministic format (ISO without millis)
  var d = new Date();
  return Utilities.formatDate(d, "UTC", "yyyy-MM-dd'T'HH:mm:ss'Z'");
}

// ============================================================
// Helper: build a seal for a pack (run once, then paste into @seal)
// ============================================================

/**
 * Compute and return seal for a pack (sha256 canonical JSON of pack without @seal).
 * @param {Object} pack
 * @returns {Object} {alg:"sha256", hash:"..."}
 */
function SCXPI_MAKE_SEAL(pack) {
  SCXPI_VERIFY_PACK(pack, { enforceSeal: false });
  var canonical = SCXPI_CANONICAL_STRING(SCXPI_STRIP_SEAL(pack));
  return { alg: "sha256", hash: SCXPI_SHA256_HEX(canonical) };
}
```

## Demo pack

```json
{
  "@kind": "scxpi.model.v1",
  "@id": "asx://model/scxpi/demo-guard/v1",
  "@v": 1,
  "@meta": {
    "name": "Demo Guard — API Router + Safety Gate",
    "author": "ASX",
    "created_utc": "2026-01-15T00:00:00Z",
    "notes": "Tiny symbolic model: tags + risk gate + routing."
  },
  "@codex": {
    "dict": {
      "D0": ["ALLOW", "DENY", "ROUTE", "TAG", "RISK"],
      "D1": ["api", "mesh", "local"],
      "D2": ["greeting", "secret", "credential", "system"]
    },
    "fields": { "intent": 0, "risk": 1, "route": 2, "tags": 3 }
  },
  "@pi": {
    "weights": {
      "risk": -3.0,
      "allow": 1.0,
      "deny": 2.0,
      "tag": 0.5,
      "geom": 0.25
    },
    "threshold": 0.0,
    "clamp": true
  },
  "@geom": {
    "features": [
      { "name": "signal", "min": -10, "max": 10 },
      { "name": "symmetry_score", "min": 0, "max": 1 }
    ]
  },
  "@rules": [
    { "id": "tag_greeting", "priority": 10, "if": { "contains": ["text", "hello"] }, "emit": ["TAG", "greeting"] },
    { "id": "tag_greeting2", "priority": 10, "if": { "contains": ["text", "hi "] }, "emit": ["TAG", "greeting"] },

    { "id": "deny_password", "priority": 100, "if": { "contains": ["text", "password"] }, "emit": ["DENY", 1] },
    { "id": "deny_apikey", "priority": 100, "if": { "contains": ["text", "apikey"] }, "emit": ["DENY", 1] },
    { "id": "deny_private_key", "priority": 100, "if": { "contains": ["text", "private key"] }, "emit": ["DENY", 1] },

    { "id": "route_mesh", "priority": 5, "if": { "contains": ["text", "mesh"] }, "emit": ["ROUTE", "mesh"] },
    { "id": "route_local", "priority": 5, "if": { "contains": ["text", "offline"] }, "emit": ["ROUTE", "local"] },

    { "id": "default_allow", "priority": 0, "if": { "always": true }, "emit": ["ALLOW", 1] }
  ],
  "@seal": {
    "alg": "sha256",
    "hash": "REPLACE_WITH_SCXPI_MAKE_SEAL_OUTPUT"
  }
}
```

## Demo harness (GAS)

```javascript
/**
 * SCXPI_MODEL_v1 — Demo Harness (GAS)
 * ------------------------------------------------------------
 * Paste alongside runtime, then run SCXPI_DEMO().
 */
function SCXPI_DEMO() {
  // Paste your pack JSON here (as an object) or load it from Properties/Drive/etc.
  var PACK = SCXPI_DEMO_PACK_OBJECT_();

  // Build seal (one-time) and patch it in
  var seal = SCXPI_MAKE_SEAL(PACK);
  PACK["@seal"] = seal;

  // Now enforce the seal
  var opts = { enforceSeal: true };

  var cases = [
    { text: "hello there", geom: { signal: 0.2, symmetry_score: 0.8 } },
    { text: "mesh connect please", geom: { signal: 1.0 } },
    { text: "my apikey is ABC123", geom: { signal: 0.0 } },
    { text: "offline mode", geom: { signal: -0.5 } }
  ];

  var out = [];
  for (var i = 0; i < cases.length; i++) {
    out.push(SCXPI_EVAL(PACK, cases[i], opts));
  }

  Logger.log(JSON.stringify(out, null, 2));
  return out;
}

function SCXPI_DEMO_PACK_OBJECT_() {
  return {
    "@kind": "scxpi.model.v1",
    "@id": "asx://model/scxpi/demo-guard/v1",
    "@v": 1,
    "@meta": {
      "name": "Demo Guard — API Router + Safety Gate",
      "author": "ASX",
      "created_utc": "2026-01-15T00:00:00Z",
      "notes": "Tiny symbolic model: tags + risk gate + routing."
    },
    "@codex": {
      "dict": {
        "D0": ["ALLOW", "DENY", "ROUTE", "TAG", "RISK"],
        "D1": ["api", "mesh", "local"],
        "D2": ["greeting", "secret", "credential", "system"]
      },
      "fields": { "intent": 0, "risk": 1, "route": 2, "tags": 3 }
    },
    "@pi": {
      "weights": {
        "risk": -3.0,
        "allow": 1.0,
        "deny": 2.0,
        "tag": 0.5,
        "geom": 0.25
      },
      "threshold": 0.0,
      "clamp": true
    },
    "@geom": {
      "features": [
        { "name": "signal", "min": -10, "max": 10 },
        { "name": "symmetry_score", "min": 0, "max": 1 }
      ]
    },
    "@rules": [
      { "id": "tag_greeting", "priority": 10, "if": { "contains": ["text", "hello"] }, "emit": ["TAG", "greeting"] },
      { "id": "tag_greeting2", "priority": 10, "if": { "contains": ["text", "hi "] }, "emit": ["TAG", "greeting"] },

      { "id": "deny_password", "priority": 100, "if": { "contains": ["text", "password"] }, "emit": ["DENY", 1] },
      { "id": "deny_apikey", "priority": 100, "if": { "contains": ["text", "apikey"] }, "emit": ["DENY", 1] },
      { "id": "deny_private_key", "priority": 100, "if": { "contains": ["text", "private key"] }, "emit": ["DENY", 1] },

      { "id": "route_mesh", "priority": 5, "if": { "contains": ["text", "mesh"] }, "emit": ["ROUTE", "mesh"] },
      { "id": "route_local", "priority": 5, "if": { "contains": ["text", "offline"] }, "emit": ["ROUTE", "local"] },

      { "id": "default_allow", "priority": 0, "if": { "always": true }, "emit": ["ALLOW", 1] }
    ]
  };
}
```

## Proof object (example)

```json
{
  "@kind": "scxpi.proof.v1",
  "alg": "sha256",
  "at_utc": "2026-01-15T11:22:33Z",
  "input_hash": "…",
  "canonical_len": 412,
  "notes": {
    "what_this_proves": [
      "model id/version used",
      "input text (lowercased) and optional geometry hash",
      "features after rule application",
      "π score + threshold used to collapse intent"
    ],
    "replay_rule": "Recompute canonical JSON and sha256; must match input_hash."
  }
}
```

## Minimal doPost example (GAS Web App)

```javascript
/**
 * SCXPI_MODEL_v1 — Minimal doPost example (GAS Web App)
 * ------------------------------------------------------------
 * Deploy as Web App, then POST JSON:
 * { "pack": {...}, "input": {...}, "opts": {"enforceSeal": true} }
 *
 * Or omit pack and load it from Script Properties / Drive in your own wrapper.
 */
function doPost(e) {
  var body = (e && e.postData && e.postData.contents) ? e.postData.contents : "{}";
  var req = JSON.parse(body);

  var pack = req.pack;
  var input = req.input || {};
  var opts = req.opts || {};

  var result = SCXPI_EVAL(pack, input, { enforceSeal: !!opts.enforceSeal });
  return ContentService
    .createTextOutput(JSON.stringify(result))
    .setMimeType(ContentService.MimeType.JSON);
}
```

## SCXPI_GEOM_FEATURES_v1 (SVG)

This section defines a **GAS-safe SVG geometry feature contract** and a lightweight extractor that produces deterministic metrics for symbolic inference.

### Geometry feature contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-geom-features/v1",
  "@v": 1,
  "title": "SCXPI_GEOM_FEATURES_v1 — SVG Geometry Feature Contract (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "features"],
  "properties": {
    "@kind": { "const": "scxpi.geom.features.v1" },
    "@id": { "type": "string", "minLength": 8 },
    "@v": { "type": "integer", "minimum": 1 },
    "features": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
          "name": { "type": "string" },
          "type": { "type": "string", "enum": ["number", "integer", "boolean", "string"] },
          "min": { "type": "number" },
          "max": { "type": "number" },
          "units": { "type": "string" },
          "notes": { "type": "string" }
        },
        "additionalProperties": false
      }
    },
    "normalization": {
      "type": "object",
      "description": "Optional normalization guidance for feature consumers.",
      "properties": {
        "signal_formula": { "type": "string" },
        "signal_range": { "type": "array", "minItems": 2, "maxItems": 2, "items": { "type": "number" } }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

### SVG geometry extractor (GAS-safe)

```javascript
/**
 * SCXPI_GEOM_FEATURES_v1 (SVG) — GAS-safe extractor
 * ------------------------------------------------------------
 * Goal: extract deterministic numeric features from SVG text
 * WITHOUT full SVG parsing, DOM, or heavy libs.
 *
 * Strategy:
 *  - Treat SVG as text
 *  - Use lightweight regex + numeric parsing
 *  - Compute conservative metrics that are stable and cheap
 *
 * Inputs:
 *  - svgText: string (raw SVG)
 *
 * Outputs:
 *  - geom object with features + derived "signal" in roughly [-10, 10]
 *
 * NOTE:
 *  - This is intentionally not a renderer.
 *  - It is a structural feature extractor for symbolic inference.
 */

// ============================================================
// Public API
// ============================================================

/**
 * Extract geometry features from SVG text.
 * @param {string} svgText
 * @returns {Object} geom features
 */
function SCXPI_GEOM_FROM_SVG(svgText) {
  svgText = String(svgText || "");
  var svg = svgText;

  // normalize newlines (canonical-ish)
  svg = svg.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

  // --- basic counts (cheap + stable)
  var counts = {
    len: svg.length,
    tag_count: SCXPI__COUNT_TAGS(svg),
    path_count: SCXPI__COUNT_OPEN_TAG(svg, "path"),
    rect_count: SCXPI__COUNT_OPEN_TAG(svg, "rect"),
    circle_count: SCXPI__COUNT_OPEN_TAG(svg, "circle"),
    ellipse_count: SCXPI__COUNT_OPEN_TAG(svg, "ellipse"),
    line_count: SCXPI__COUNT_OPEN_TAG(svg, "line"),
    polyline_count: SCXPI__COUNT_OPEN_TAG(svg, "polyline"),
    polygon_count: SCXPI__COUNT_OPEN_TAG(svg, "polygon"),
    text_count: SCXPI__COUNT_OPEN_TAG(svg, "text"),
    group_count: SCXPI__COUNT_OPEN_TAG(svg, "g"),
    use_count: SCXPI__COUNT_OPEN_TAG(svg, "use"),
    defs_count: SCXPI__COUNT_OPEN_TAG(svg, "defs")
  };

  // --- viewBox / width / height
  var dims = SCXPI__EXTRACT_SVG_DIMS(svg);

  // --- path d stats
  var pathStats = SCXPI__PATH_D_STATS(svg);

  // --- style stats (fill/stroke/opacities)
  var styleStats = SCXPI__STYLE_STATS(svg);

  // --- transform stats
  var transformStats = SCXPI__TRANSFORM_STATS(svg);

  // --- symmetry-ish heuristic (cheap)
  // We can’t render, but we can detect "balance" signals:
  // - presence of mirrored scales: scale(-1,1) or scale(1,-1)
  // - repeated use via <use> suggests symmetry / repetition
  // - repeated path d hashes suggests duplication
  var symmetry = SCXPI__SYMMETRY_HEURISTIC(svg, counts, transformStats, pathStats);

  // --- density proxies
  var area = (dims.width > 0 && dims.height > 0) ? (dims.width * dims.height) : 0;
  var element_total = (
    counts.path_count + counts.rect_count + counts.circle_count + counts.ellipse_count +
    counts.line_count + counts.polyline_count + counts.polygon_count + counts.text_count
  );

  var density = (area > 0) ? (element_total / area) : 0; // elements per px^2 (tiny)
  var path_cmd_density = (counts.path_count > 0) ? (pathStats.cmd_count_total / counts.path_count) : 0;

  // --- derived signal in [-10,10] (for π)
  // signal rewards structured richness (elements, commands, gradients) and penalizes chaos (too many transforms, excessive length vs structure)
  var signal = SCXPI__SIGNAL({
    counts: counts,
    dims: dims,
    pathStats: pathStats,
    styleStats: styleStats,
    transformStats: transformStats,
    symmetry: symmetry,
    density: density,
    path_cmd_density: path_cmd_density
  });

  return {
    "@kind": "scxpi.geom.svg.v1",
    width: dims.width,
    height: dims.height,
    viewbox_present: dims.viewbox_present,
    area: area,
    len: counts.len,

    tag_count: counts.tag_count,
    element_total: element_total,

    path_count: counts.path_count,
    rect_count: counts.rect_count,
    circle_count: counts.circle_count,
    ellipse_count: counts.ellipse_count,
    line_count: counts.line_count,
    polyline_count: counts.polyline_count,
    polygon_count: counts.polygon_count,
    text_count: counts.text_count,
    group_count: counts.group_count,
    use_count: counts.use_count,
    defs_count: counts.defs_count,

    density: density,
    path_cmd_density: path_cmd_density,

    path_cmd_count_total: pathStats.cmd_count_total,
    path_coord_count_total: pathStats.coord_count_total,
    path_d_avg_len: pathStats.d_avg_len,
    path_d_max_len: pathStats.d_max_len,
    path_repeat_ratio: pathStats.repeat_ratio,

    fill_count: styleStats.fill_count,
    stroke_count: styleStats.stroke_count,
    stroke_width_avg: styleStats.stroke_width_avg,
    opacity_avg: styleStats.opacity_avg,
    gradient_count: styleStats.gradient_count,

    transform_count: transformStats.transform_count,
    has_negative_scale: transformStats.has_negative_scale,
    rotation_count: transformStats.rotation_count,

    symmetry_score: symmetry.symmetry_score,
    repetition_score: symmetry.repetition_score,

    signal: signal
  };
}

// ============================================================
// Contract: feature list (optional helper for pack authors)
// ============================================================

function SCXPI_GEOM_FEATURES_V1_SPEC() {
  return {
    "@kind": "scxpi.geom.features.v1",
    "@id": "asx://geom/svg/features/v1",
    "@v": 1,
    "features": [
      { "name": "width", "type": "number", "min": 0, "max": 100000, "units": "px" },
      { "name": "height", "type": "number", "min": 0, "max": 100000, "units": "px" },
      { "name": "area", "type": "number", "min": 0, "max": 1e12, "units": "px^2" },

      { "name": "len", "type": "integer", "min": 0, "max": 5000000, "units": "chars" },
      { "name": "tag_count", "type": "integer", "min": 0, "max": 500000 },

      { "name": "element_total", "type": "integer", "min": 0, "max": 500000 },
      { "name": "path_count", "type": "integer" },
      { "name": "rect_count", "type": "integer" },
      { "name": "circle_count", "type": "integer" },
      { "name": "ellipse_count", "type": "integer" },
      { "name": "line_count", "type": "integer" },
      { "name": "polyline_count", "type": "integer" },
      { "name": "polygon_count", "type": "integer" },
      { "name": "text_count", "type": "integer" },
      { "name": "group_count", "type": "integer" },
      { "name": "use_count", "type": "integer" },
      { "name": "defs_count", "type": "integer" },

      { "name": "density", "type": "number", "min": 0, "max": 10, "units": "elements/px^2" },
      { "name": "path_cmd_density", "type": "number", "min": 0, "max": 100000 },

      { "name": "path_cmd_count_total", "type": "integer" },
      { "name": "path_coord_count_total", "type": "integer" },
      { "name": "path_d_avg_len", "type": "number" },
      { "name": "path_d_max_len", "type": "number" },
      { "name": "path_repeat_ratio", "type": "number", "min": 0, "max": 1 },

      { "name": "fill_count", "type": "integer" },
      { "name": "stroke_count", "type": "integer" },
      { "name": "stroke_width_avg", "type": "number", "min": 0, "max": 1000 },
      { "name": "opacity_avg", "type": "number", "min": 0, "max": 1 },
      { "name": "gradient_count", "type": "integer" },

      { "name": "transform_count", "type": "integer" },
      { "name": "has_negative_scale", "type": "boolean" },
      { "name": "rotation_count", "type": "integer" },

      { "name": "symmetry_score", "type": "number", "min": 0, "max": 1 },
      { "name": "repetition_score", "type": "number", "min": 0, "max": 1 },

      { "name": "signal", "type": "number", "min": -10, "max": 10, "notes": "Derived compact signal for π scoring." }
    ],
    "normalization": {
      "signal_formula": "signal = clamp(-10..10, f(structure, density, styles, transforms, symmetry))",
      "signal_range": [-10, 10]
    }
  };
}

// ============================================================
// Internals
// ============================================================

function SCXPI__COUNT_TAGS(s) {
  // rough tag count: occurrences of "<" followed by a letter or "!" or "?"
  var m = s.match(/<([A-Za-z!?])/g);
  return m ? m.length : 0;
}

function SCXPI__COUNT_OPEN_TAG(s, tag) {
  // counts "<tag" not "</tag"
  // allow namespace prefix: <svg:path ...> is treated as path too
  var re = new RegExp("<\\s*(?:[A-Za-z0-9_-]+:)?" + tag + "\\b", "g");
  var m = s.match(re);
  return m ? m.length : 0;
}

function SCXPI__EXTRACT_SVG_DIMS(svg) {
  // Try viewBox first, then width/height attrs.
  var out = { width: 0, height: 0, viewbox_present: false };

  var vb = SCXPI__MATCH_ATTR(svg, "viewBox");
  if (vb) {
    // viewBox="minx miny w h" (commas allowed)
    var parts = vb.replace(/,/g, " ").trim().split(/\s+/);
    if (parts.length >= 4) {
      var w = Number(parts[2]);
      var h = Number(parts[3]);
      if (isFinite(w) && isFinite(h) && w > 0 && h > 0) {
        out.width = w;
        out.height = h;
        out.viewbox_present = true;
        return out;
      }
    }
  }

  // width/height
  var wAttr = SCXPI__MATCH_ATTR(svg, "width");
  var hAttr = SCXPI__MATCH_ATTR(svg, "height");
  var wNum = SCXPI__PARSE_SVG_NUMBER(wAttr);
  var hNum = SCXPI__PARSE_SVG_NUMBER(hAttr);
  if (wNum > 0) out.width = wNum;
  if (hNum > 0) out.height = hNum;

  return out;
}

function SCXPI__MATCH_ATTR(svg, attrName) {
  // finds first attrName="..."
  var re = new RegExp("\\b" + attrName + "\\s*=\\s*\"([^\"]+)\"", "i");
  var m = svg.match(re);
  return m ? m[1] : null;
}

function SCXPI__PARSE_SVG_NUMBER(v) {
  if (!v) return 0;
  // strip units: px, pt, em, %, etc. We only take the numeric part.
  var m = String(v).trim().match(/-?\d+(\.\d+)?/);
  if (!m) return 0;
  var n = Number(m[0]);
  return (isFinite(n) && n >= 0) ? n : 0;
}

function SCXPI__PATH_D_STATS(svg) {
  // Extract all d="..." occurrences for <path ...>
  // Compute:
  // - cmd_count_total: total command letters across all paths
  // - coord_count_total: total numeric tokens across all paths
  // - d_avg_len / d_max_len
  // - repeat_ratio: fraction of paths sharing the most common d hash (duplication)
  var re = /<\s*(?:[A-Za-z0-9_-]+:)?path\b[^>]*\bd\s*=\s*"([^"]*)"/gi;

  var ds = [];
  var m;
  while ((m = re.exec(svg)) !== null) {
    ds.push(m[1] || "");
    if (ds.length > 5000) break; // safety cap
  }

  var cmdTotal = 0;
  var numTotal = 0;
  var lenTotal = 0;
  var lenMax = 0;

  var freq = {};
  for (var i = 0; i < ds.length; i++) {
    var d = ds[i];
    var L = d.length;
    lenTotal += L;
    if (L > lenMax) lenMax = L;

    // count command letters
    var cm = d.match(/[MmZzLlHhVvCcSsQqTtAa]/g);
    cmdTotal += cm ? cm.length : 0;

    // count numeric tokens
    var nm = d.match(/-?\d+(\.\d+)?/g);
    numTotal += nm ? nm.length : 0;

    // hash for repetition measure (sha256 in GAS)
    var h = SCXPI__FAST_HASH32(d);
    freq[h] = (freq[h] || 0) + 1;
  }

  var maxFreq = 0;
  for (var k in freq) if (freq[k] > maxFreq) maxFreq = freq[k];

  var repeatRatio = (ds.length > 0) ? (maxFreq / ds.length) : 0;

  return {
    path_count: ds.length,
    cmd_count_total: cmdTotal,
    coord_count_total: numTotal,
    d_avg_len: (ds.length > 0) ? (lenTotal / ds.length) : 0,
    d_max_len: lenMax,
    repeat_ratio: repeatRatio
  };
}

function SCXPI__STYLE_STATS(svg) {
  // Approximate:
  // - fill_count: occurrences of fill="..." or "fill:"
  // - stroke_count: occurrences of stroke="..." or "stroke:"
  // - stroke_width_avg: avg of stroke-width occurrences
  // - opacity_avg: avg of opacity occurrences (opacity/fill-opacity/stroke-opacity)
  // - gradient_count: <linearGradient / <radialGradient
  var fillM = svg.match(/\bfill\s*=\s*"[^"]*"/gi);
  var fillCss = svg.match(/\bfill\s*:\s*[^;"]+/gi);
  var strokeM = svg.match(/\bstroke\s*=\s*"[^"]*"/gi);
  var strokeCss = svg.match(/\bstroke\s*:\s*[^;"]+/gi);

  var sw = [];
  var swM1 = svg.match(/\bstroke-width\s*=\s*"([^"]+)"/gi);
  if (swM1) {
    for (var i = 0; i < swM1.length; i++) {
      var mm = swM1[i].match(/"([^"]+)"/);
      var n = SCXPI__PARSE_SVG_NUMBER(mm ? mm[1] : null);
      if (n > 0) sw.push(n);
    }
  }
  var swM2 = svg.match(/\bstroke-width\s*:\s*([^;"]+)/gi);
  if (swM2) {
    for (var j = 0; j < swM2.length; j++) {
      var n2 = SCXPI__PARSE_SVG_NUMBER(swM2[j]);
      if (n2 > 0) sw.push(n2);
    }
  }

  var op = [];
  var opM = svg.match(/\b(?:opacity|fill-opacity|stroke-opacity)\s*=\s*"([^"]+)"/gi);
  if (opM) {
    for (var k = 0; k < opM.length; k++) {
      var mm2 = opM[k].match(/"([^"]+)"/);
      var n3 = Number(mm2 ? mm2[1] : NaN);
      if (isFinite(n3)) op.push(Math.max(0, Math.min(1, n3)));
    }
  }
  var opCss = svg.match(/\b(?:opacity|fill-opacity|stroke-opacity)\s*:\s*([^;"]+)/gi);
  if (opCss) {
    for (var z = 0; z < opCss.length; z++) {
      var n4 = Number(opCss[z].match(/-?\d+(\.\d+)?/) ? opCss[z].match(/-?\d+(\.\d+)?/)[0] : NaN);
      if (isFinite(n4)) op.push(Math.max(0, Math.min(1, n4)));
    }
  }

  var grad = 0;
  grad += SCXPI__COUNT_OPEN_TAG(svg, "linearGradient");
  grad += SCXPI__COUNT_OPEN_TAG(svg, "radialGradient");

  return {
    fill_count: (fillM ? fillM.length : 0) + (fillCss ? fillCss.length : 0),
    stroke_count: (strokeM ? strokeM.length : 0) + (strokeCss ? strokeCss.length : 0),
    stroke_width_avg: (sw.length ? SCXPI__AVG(sw) : 0),
    opacity_avg: (op.length ? SCXPI__AVG(op) : 1),
    gradient_count: grad
  };
}

function SCXPI__TRANSFORM_STATS(svg) {
  // transform="..." occurrences + detect negative scale, count rotate()
  var tM = svg.match(/\btransform\s*=\s*"[^"]*"/gi);
  var count = tM ? tM.length : 0;

  var hasNegScale = /scale\s*\(\s*-\s*\d/.test(svg) || /scale\s*\(\s*\d+(\.\d+)?\s*,\s*-\s*\d/.test(svg);
  var rotM = svg.match(/rotate\s*\(/gi);
  var rotCount = rotM ? rotM.length : 0;

  return {
    transform_count: count,
    has_negative_scale: !!hasNegScale,
    rotation_count: rotCount
  };
}

function SCXPI__SYMMETRY_HEURISTIC(svg, counts, transformStats, pathStats) {
  // Symmetry score: bounded [0,1]
  // Signals:
  // - negative scale suggests mirroring
  // - use_count suggests repetition
  // - repeat_ratio suggests duplication
  var s = 0;

  if (transformStats.has_negative_scale) s += 0.35;

  // repetition via <use>
  if (counts.use_count > 0) s += Math.min(0.35, counts.use_count / 50);

  // repeated paths
  s += Math.min(0.30, pathStats.repeat_ratio * 0.60);

  var symmetry_score = Math.max(0, Math.min(1, s));

  // repetition_score: use and repeat_ratio combined
  var rep = 0;
  rep += Math.min(0.6, counts.use_count / 20);
  rep += Math.min(0.4, pathStats.repeat_ratio);
  var repetition_score = Math.max(0, Math.min(1, rep));

  return { symmetry_score: symmetry_score, repetition_score: repetition_score };
}

function SCXPI__SIGNAL(o) {
  // compact scalar in [-10,10] for π scoring
  // This is intentionally simple + deterministic.
  //
  // + structure: element_total, path_cmd_density, gradient_count
  // - chaos: transform_count, huge len with low structure
  var counts = o.counts;
  var dims = o.dims;
  var pathStats = o.pathStats;
  var styleStats = o.styleStats;
  var transformStats = o.transformStats;

  var element_total = (
    counts.path_count + counts.rect_count + counts.circle_count + counts.ellipse_count +
    counts.line_count + counts.polyline_count + counts.polygon_count + counts.text_count
  );

  // normalize bits
  var structure = 0;
  structure += Math.log(1 + element_total);                 // 0.. ~
  structure += Math.log(1 + pathStats.cmd_count_total);     // 0.. ~
  structure += Math.min(2, styleStats.gradient_count * 0.5);

  var symmetryBoost = o.symmetry.symmetry_score * 1.5;

  var chaos = 0;
  chaos += Math.min(3, transformStats.transform_count / 10);
  chaos += Math.min(3, counts.len / 200000); // penalize huge SVGs

  // penalize "long but empty": len high, element_total low
  if (counts.len > 50000 && element_total < 5) chaos += 2;

  // density sanity (very tiny density is fine; extremely high density suggests spam/overdraw)
  var densityPenalty = 0;
  if (o.density > 0.002) densityPenalty += Math.min(2, o.density * 500); // scaled

  var raw = (structure + symmetryBoost) - (chaos + densityPenalty);

  // map to [-10,10] with soft clamp
  var sig = SCXPI__SOFT_CLAMP(raw * 2, 10);
  return sig;
}

function SCXPI__SOFT_CLAMP(x, limit) {
  // x in R -> (-limit, limit)
  // limit * x/(limit + |x|)
  var L = Number(limit) || 10;
  return (L * x) / (L + Math.abs(x));
}

function SCXPI__AVG(arr) {
  var s = 0;
  for (var i = 0; i < arr.length; i++) s += arr[i];
  return arr.length ? (s / arr.length) : 0;
}

// ============================================================
// Fast 32-bit hash for repetition bins (NOT cryptographic)
// This is just to bucket identical strings cheaply.
// ============================================================

function SCXPI__FAST_HASH32(str) {
  // FNV-1a 32-bit
  var h = 0x811c9dc5;
  for (var i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return ("00000000" + h.toString(16)).slice(-8);
}
```

### Integration helper

```javascript
/**
 * Integration: plug SCXPI_GEOM_FROM_SVG into SCXPI_EVAL input.geom
 * ------------------------------------------------------------
 * Example:
 *   var geom = SCXPI_GEOM_FROM_SVG(svgText);
 *   var result = SCXPI_EVAL(pack, { text: "classify svg", geom: geom }, { enforceSeal: true });
 */
function SCXPI_SVG_EVAL(pack, text, svgText, opts) {
  var geom = SCXPI_GEOM_FROM_SVG(svgText);
  return SCXPI_EVAL(pack, { text: String(text || ""), geom: geom }, opts || {});
}
```

### Example model pack (geometry-driven)

```json
{
  "@kind": "scxpi.model.v1",
  "@id": "asx://model/scxpi/svg-classifier-lite/v1",
  "@v": 1,
  "@meta": {
    "name": "SVG Classifier Lite (Geometry-Driven)",
    "author": "ASX",
    "created_utc": "2026-01-15T00:00:00Z",
    "notes": "Uses SCXPI_GEOM_FEATURES_v1 signal + counts to route/tag."
  },
  "@codex": {
    "dict": {
      "D0": ["ALLOW", "DENY", "ROUTE", "TAG", "RISK"],
      "D1": ["api", "mesh", "local"],
      "D2": ["icon", "diagram", "dense", "sparse", "textual", "geometric"]
    },
    "fields": { "intent": 0, "risk": 1, "route": 2, "tags": 3 }
  },
  "@pi": {
    "weights": {
      "risk": -2.0,
      "allow": 1.0,
      "deny": 2.0,
      "tag": 0.5,
      "geom": 1.25
    },
    "threshold": 0.0,
    "clamp": true
  },
  "@rules": [
    {
      "id": "tag_icon_low_complexity",
      "priority": 20,
      "if": { "and": [
        { "lt": ["$geom.path_count", 6] },
        { "lt": ["$geom.element_total", 12] },
        { "gt": ["$geom.symmetry_score", 0.4] }
      ]},
      "emit": ["TAG", "icon"]
    },
    {
      "id": "tag_textual",
      "priority": 20,
      "if": { "gt": ["$geom.text_count", 0] },
      "emit": ["TAG", "textual"]
    },
    {
      "id": "tag_dense",
      "priority": 10,
      "if": { "gt": ["$geom.path_cmd_count_total", 300] },
      "emit": ["TAG", "dense"]
    },
    {
      "id": "tag_sparse",
      "priority": 10,
      "if": { "and": [
        { "lt": ["$geom.element_total", 5] },
        { "gt": ["$geom.len", 20000] }
      ]},
      "emit": ["TAG", "sparse"]
    },
    {
      "id": "route_local_if_geometric",
      "priority": 5,
      "if": { "and": [
        { "gt": ["$geom.circle_count", 0] },
        { "gt": ["$geom.rect_count", 0] }
      ]},
      "emit": ["ROUTE", "local"]
    },
    { "id": "default_allow", "priority": 0, "if": { "always": true }, "emit": ["ALLOW", 1] }
  ]
}
```

### Demo harness

```javascript
/**
 * Demo harness for SVG geometry features.
 * Paste into GAS and run SCXPI_GEOM_DEMO().
 */
function SCXPI_GEOM_DEMO() {
  var svg1 = '<svg viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80"/><circle cx="50" cy="50" r="20"/></svg>';
  var svg2 = '<svg width="512" height="512"><path d="M10 10 L500 10 L500 500 L10 500 Z"/><path d="M20 20 L490 20 L490 490 L20 490 Z"/></svg>';
  var g1 = SCXPI_GEOM_FROM_SVG(svg1);
  var g2 = SCXPI_GEOM_FROM_SVG(svg2);
  Logger.log(JSON.stringify({ g1: g1, g2: g2 }, null, 2));
  return { g1: g1, g2: g2 };
}
```

### What you just got (SCXPI_GEOM_FEATURES_v1)

- **A formal feature contract** (`scxpi.geom.features.v1`)
- A **GAS-safe SVG feature extractor** (regex + numeric parsing only)
- A derived **`geom.signal`** in `[-10,10]` designed for π weighting
- A **ready model pack** that uses `$geom.*` conditions
- A demo to validate output

If you want the next upgrade layer, say: **“emit SCXPI_GEOM_PATH_BBOX_v1”** and a *bounded* path “approx bbox” extractor (still GAS-safe) can be added to estimate min/max for M/L/H/V commands (no curves) for stronger spatial reasoning without rendering.

## SCXPI_GEOM_PATH_BBOX_v1 (SVG path bbox approximation)

This section adds a **GAS-safe** path bounding-box estimator that provides stronger spatial signals without rendering.

### Path bbox contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-geom-path-bbox/v1",
  "@v": 1,
  "title": "SCXPI_GEOM_PATH_BBOX_v1 — SVG Path BBox Approximation (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "notes", "outputs"],
  "properties": {
    "@kind": { "const": "scxpi.geom.path.bbox.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "notes": { "type": "string" },
    "outputs": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "additionalProperties": false
}
```

### Path bbox extractor (GAS-safe)

```javascript
/**
 * SCXPI_GEOM_PATH_BBOX_v1 — Approx path bounding boxes (GAS-safe)
 * ------------------------------------------------------------
 * Adds stronger spatial signals without rendering:
 * - Parses <path d="..."> command stream (subset) and estimates bbox.
 *
 * Supported commands (absolute + relative):
 *   M/m, L/l, H/h, V/v, Z/z
 *
 * Curve commands (C/S/Q/T/A) are NOT geometrically evaluated.
 * For curves, we conservatively include their control/endpoints (cheap heuristic),
 * which is still useful and deterministic.
 *
 * Output:
 *   {
 *     path_bbox_union: { minx, miny, maxx, maxy, w, h },
 *     path_bbox_count: N,
 *     path_bbox_area: number,
 *     path_bbox_cover_ratio: bbox_area / svg_area (if svg dims known)
 *   }
 *
 * Integrates cleanly into SCXPI_GEOM_FROM_SVG by calling SCXPI_PATH_BBOX_FROM_SVG(svgText, dims).
 */

// ============================================================
// Public API
// ============================================================

/**
 * Compute union bbox over all <path d="..."> in an SVG string.
 * @param {string} svgText
 * @param {Object=} dims Optional {width,height,area} from SCXPI__EXTRACT_SVG_DIMS
 * @returns {Object}
 */
function SCXPI_PATH_BBOX_FROM_SVG(svgText, dims) {
  var svg = String(svgText || "");

  var re = /<\s*(?:[A-Za-z0-9_-]+:)?path\b[^>]*\bd\s*=\s*"([^"]*)"/gi;
  var m;
  var union = SCXPI__BBOX_INIT();
  var count = 0;

  while ((m = re.exec(svg)) !== null) {
    var d = m[1] || "";
    var bb = SCXPI_PATH_BBOX_FROM_D(d);
    if (bb && bb.valid) {
      SCXPI__BBOX_EXPAND(union, bb);
      count++;
    }
    if (count > 5000) break; // safety cap
  }

  var out = {
    "@kind": "scxpi.geom.path.bbox.union.v1",
    path_bbox_count: count,
    path_bbox_union: SCXPI__BBOX_FINAL(union),
    path_bbox_area: 0,
    path_bbox_cover_ratio: 0
  };

  if (out.path_bbox_union.valid) {
    out.path_bbox_area = out.path_bbox_union.w * out.path_bbox_union.h;
  }

  if (dims && dims.width > 0 && dims.height > 0 && out.path_bbox_area > 0) {
    var svgArea = dims.width * dims.height;
    out.path_bbox_cover_ratio = (svgArea > 0) ? (out.path_bbox_area / svgArea) : 0;
  }

  return out;
}

/**
 * Approx bbox from a single SVG path "d" string.
 * @param {string} d
 * @returns {Object} bbox {minx,miny,maxx,maxy,w,h,valid}
 */
function SCXPI_PATH_BBOX_FROM_D(d) {
  d = String(d || "");
  if (!d.length) return { valid: false };

  // Tokenize into commands + numbers.
  // We keep command letters as separate tokens.
  var tokens = SCXPI__TOKENIZE_PATH(d);
  if (!tokens.length) return { valid: false };

  var bb = SCXPI__BBOX_INIT();

  var cx = 0, cy = 0;      // current point
  var sx = 0, sy = 0;      // subpath start
  var lastCmd = null;

  var i = 0;
  while (i < tokens.length) {
    var t = tokens[i];

    // Command?
    if (SCXPI__IS_CMD(t)) {
      lastCmd = t;
      i++;
      if (t === "Z" || t === "z") {
        // Closepath: return to subpath start
        cx = sx; cy = sy;
        SCXPI__BBOX_POINT(bb, cx, cy);
      }
      continue;
    }

    // If no explicit command, SVG allows implicit repeat of last command
    var cmd = lastCmd;
    if (!cmd) {
      // If we have numbers before any command, treat as invalid
      break;
    }

    // Dispatch based on cmd, consuming numeric args
    var isRel = (cmd === cmd.toLowerCase());
    var C = cmd.toUpperCase();

    if (C === "M") {
      // M: (x y)+ first pair is move; subsequent pairs are treated as L
      var p = SCXPI__READ_PAIR(tokens, i);
      if (!p) break;
      i = p.next;

      var nx = isRel ? (cx + p.x) : p.x;
      var ny = isRel ? (cy + p.y) : p.y;

      cx = nx; cy = ny;
      sx = nx; sy = ny;
      SCXPI__BBOX_POINT(bb, cx, cy);

      // Subsequent pairs become implicit L commands until next command
      lastCmd = isRel ? "l" : "L";
      continue;
    }

    if (C === "L") {
      var p2 = SCXPI__READ_PAIR(tokens, i);
      if (!p2) break;
      i = p2.next;

      var lx = isRel ? (cx + p2.x) : p2.x;
      var ly = isRel ? (cy + p2.y) : p2.y;

      cx = lx; cy = ly;
      SCXPI__BBOX_POINT(bb, cx, cy);
      continue;
    }

    if (C === "H") {
      var n = SCXPI__READ_NUM(tokens, i);
      if (!n) break;
      i = n.next;

      var hx = isRel ? (cx + n.v) : n.v;
      cx = hx;
      SCXPI__BBOX_POINT(bb, cx, cy);
      continue;
    }

    if (C === "V") {
      var n2 = SCXPI__READ_NUM(tokens, i);
      if (!n2) break;
      i = n2.next;

      var vy = isRel ? (cy + n2.v) : n2.v;
      cy = vy;
      SCXPI__BBOX_POINT(bb, cx, cy);
      continue;
    }

    // Curves/arc: C/S/Q/T/A — we do a cheap conservative include of all numeric coordinates.
    // This is not a true bbox, but a stable feature.
    if (C === "C") {
      // C: (x1 y1 x2 y2 x y)+
      var c = SCXPI__READ_N_NUMS(tokens, i, 6);
      if (!c) break;
      i = c.next;
      // control points + end point
      var pts = SCXPI__COORDS_TO_POINTS(c.vals, isRel, cx, cy);
      for (var k = 0; k < pts.length; k++) SCXPI__BBOX_POINT(bb, pts[k].x, pts[k].y);
      // end point is last pair
      var end = pts[pts.length - 1];
      cx = end.x; cy = end.y;
      continue;
    }

    if (C === "S") {
      // S: (x2 y2 x y)+
      var s = SCXPI__READ_N_NUMS(tokens, i, 4);
      if (!s) break;
      i = s.next;
      var ptsS = SCXPI__COORDS_TO_POINTS(s.vals, isRel, cx, cy);
      for (var k2 = 0; k2 < ptsS.length; k2++) SCXPI__BBOX_POINT(bb, ptsS[k2].x, ptsS[k2].y);
      var endS = ptsS[ptsS.length - 1];
      cx = endS.x; cy = endS.y;
      continue;
    }

    if (C === "Q") {
      // Q: (x1 y1 x y)+
      var q = SCXPI__READ_N_NUMS(tokens, i, 4);
      if (!q) break;
      i = q.next;
      var ptsQ = SCXPI__COORDS_TO_POINTS(q.vals, isRel, cx, cy);
      for (var k3 = 0; k3 < ptsQ.length; k3++) SCXPI__BBOX_POINT(bb, ptsQ[k3].x, ptsQ[k3].y);
      var endQ = ptsQ[ptsQ.length - 1];
      cx = endQ.x; cy = endQ.y;
      continue;
    }

    if (C === "T") {
      // T: (x y)+
      var t2 = SCXPI__READ_PAIR(tokens, i);
      if (!t2) break;
      i = t2.next;
      var tx = isRel ? (cx + t2.x) : t2.x;
      var ty = isRel ? (cy + t2.y) : t2.y;
      cx = tx; cy = ty;
      SCXPI__BBOX_POINT(bb, cx, cy);
      continue;
    }

    if (C === "A") {
      // A: (rx ry x-axis-rot large-arc-flag sweep-flag x y)+
      // We'll include endpoint (x,y) and radii as weak signals.
      var a = SCXPI__READ_N_NUMS(tokens, i, 7);
      if (!a) break;
      i = a.next;
      // endpoint is last two
      var ex = a.vals[5], ey = a.vals[6];
      var ax = isRel ? (cx + ex) : ex;
      var ay = isRel ? (cy + ey) : ey;
      SCXPI__BBOX_POINT(bb, ax, ay);
      // Include radii around endpoint as conservative bbox expansion
      var rx = Math.abs(Number(a.vals[0]) || 0);
      var ry = Math.abs(Number(a.vals[1]) || 0);
      if (rx > 0 || ry > 0) {
        SCXPI__BBOX_POINT(bb, ax - rx, ay - ry);
        SCXPI__BBOX_POINT(bb, ax + rx, ay + ry);
      }
      cx = ax; cy = ay;
      continue;
    }

    // Unknown: break
    break;
  }

  return SCXPI__BBOX_FINAL(bb);
}

// ============================================================
// Tokenization helpers
// ============================================================

function SCXPI__TOKENIZE_PATH(d) {
  // Split into command letters and numbers. Handles commas and tight packing.
  // Example: "M10-20L30,40" -> ["M","10","-20","L","30","40"]
  var out = [];
  var re = /[MmZzLlHhVvCcSsQqTtAa]|-?\d*\.?\d+(?:e[-+]?\d+)?/g;
  var m;
  while ((m = re.exec(d)) !== null) {
    out.push(m[0]);
    if (out.length > 200000) break; // safety cap
  }
  return out;
}

function SCXPI__IS_CMD(t) {
  return /^[MmZzLlHhVvCcSsQqTtAa]$/.test(t);
}

function SCXPI__READ_NUM(tokens, i) {
  if (i >= tokens.length) return null;
  var t = tokens[i];
  if (SCXPI__IS_CMD(t)) return null;
  var v = Number(t);
  if (!isFinite(v)) return null;
  return { v: v, next: i + 1 };
}

function SCXPI__READ_PAIR(tokens, i) {
  var a = SCXPI__READ_NUM(tokens, i);
  if (!a) return null;
  var b = SCXPI__READ_NUM(tokens, a.next);
  if (!b) return null;
  return { x: a.v, y: b.v, next: b.next };
}

function SCXPI__READ_N_NUMS(tokens, i, n) {
  var vals = [];
  var idx = i;
  for (var k = 0; k < n; k++) {
    var r = SCXPI__READ_NUM(tokens, idx);
    if (!r) return null;
    vals.push(r.v);
    idx = r.next;
  }
  return { vals: vals, next: idx };
}

function SCXPI__COORDS_TO_POINTS(vals, isRel, cx, cy) {
  // Convert coords list to points list depending on command kind.
  // We interpret pairs in order as points; for rel we offset by (cx,cy).
  var pts = [];
  for (var i = 0; i < vals.length; i += 2) {
    var x = vals[i], y = vals[i + 1];
    var px = isRel ? (cx + x) : x;
    var py = isRel ? (cy + y) : y;
    pts.push({ x: px, y: py });
  }
  return pts;
}

// ============================================================
// BBox helpers
// ============================================================

function SCXPI__BBOX_INIT() {
  return { minx: Infinity, miny: Infinity, maxx: -Infinity, maxy: -Infinity, valid: false };
}

function SCXPI__BBOX_POINT(bb, x, y) {
  x = Number(x); y = Number(y);
  if (!isFinite(x) || !isFinite(y)) return;
  if (x < bb.minx) bb.minx = x;
  if (y < bb.miny) bb.miny = y;
  if (x > bb.maxx) bb.maxx = x;
  if (y > bb.maxy) bb.maxy = y;
  bb.valid = true;
}

function SCXPI__BBOX_EXPAND(union, bb) {
  if (!bb || !bb.valid) return;
  SCXPI__BBOX_POINT(union, bb.minx, bb.miny);
  SCXPI__BBOX_POINT(union, bb.maxx, bb.maxy);
}

function SCXPI__BBOX_FINAL(bb) {
  if (!bb || !bb.valid || bb.minx === Infinity) return { valid: false };
  var w = bb.maxx - bb.minx;
  var h = bb.maxy - bb.miny;
  return {
    minx: bb.minx,
    miny: bb.miny,
    maxx: bb.maxx,
    maxy: bb.maxy,
    w: isFinite(w) ? w : 0,
    h: isFinite(h) ? h : 0,
    valid: true
  };
}
```

### Integrating with SCXPI_GEOM_FROM_SVG

```javascript
/**
 * Patch SCXPI_GEOM_FROM_SVG to include bbox union outputs.
 * ------------------------------------------------------------
 * Add this snippet near the end of SCXPI_GEOM_FROM_SVG after dims/pathStats exist:
 *
 *   var bbox = SCXPI_PATH_BBOX_FROM_SVG(svg, { width: dims.width, height: dims.height });
 *   // then merge bbox fields into the returned geom object
 *
 * Here’s a helper wrapper that does it without editing your original:
 */
function SCXPI_GEOM_FROM_SVG_WITH_BBOX(svgText) {
  var geom = SCXPI_GEOM_FROM_SVG(svgText);

  var dims = { width: geom.width, height: geom.height };
  var bbox = SCXPI_PATH_BBOX_FROM_SVG(svgText, dims);

  // Merge
  geom.path_bbox_count = bbox.path_bbox_count;
  geom.path_bbox_area = bbox.path_bbox_area;
  geom.path_bbox_cover_ratio = bbox.path_bbox_cover_ratio;

  if (bbox.path_bbox_union && bbox.path_bbox_union.valid) {
    geom.path_bbox_minx = bbox.path_bbox_union.minx;
    geom.path_bbox_miny = bbox.path_bbox_union.miny;
    geom.path_bbox_maxx = bbox.path_bbox_union.maxx;
    geom.path_bbox_maxy = bbox.path_bbox_union.maxy;
    geom.path_bbox_w = bbox.path_bbox_union.w;
    geom.path_bbox_h = bbox.path_bbox_union.h;
  } else {
    geom.path_bbox_minx = 0;
    geom.path_bbox_miny = 0;
    geom.path_bbox_maxx = 0;
    geom.path_bbox_maxy = 0;
    geom.path_bbox_w = 0;
    geom.path_bbox_h = 0;
  }

  // Optionally: strengthen signal using cover ratio (still deterministic)
  // e.g., add small bump if paths occupy significant area.
  if (isFinite(geom.path_bbox_cover_ratio) && geom.path_bbox_cover_ratio > 0) {
    var bump = Math.min(1.0, geom.path_bbox_cover_ratio * 2.0);
    geom.signal = SCXPI__SOFT_CLAMP(geom.signal + bump, 10);
  }

  return geom;
}
```

### Demo harness

```javascript
/**
 * Demo: validate bbox extraction.
 */
function SCXPI_BBOX_DEMO() {
  var svg = [
    '<svg viewBox="0 0 100 100">',
    '<path d="M10 10 L90 10 L90 90 L10 90 Z"/>',
    '<path d="M20 20 L80 20 L80 80 L20 80 Z"/>',
    '</svg>'
  ].join("");

  var geom = SCXPI_GEOM_FROM_SVG_WITH_BBOX(svg);
  Logger.log(JSON.stringify(geom, null, 2));
  return geom;
}
```

### Path bbox outputs

```json
{
  "@kind": "scxpi.geom.path.bbox.v1",
  "@id": "asx://geom/svg/path-bbox/v1",
  "@v": 1,
  "notes": "Approx bbox from SVG path d. Exact for M/L/H/V/Z. Conservative for curves/arc by including endpoints/control points and arc radii around endpoints.",
  "outputs": [
    "path_bbox_count",
    "path_bbox_minx",
    "path_bbox_miny",
    "path_bbox_maxx",
    "path_bbox_maxy",
    "path_bbox_w",
    "path_bbox_h",
    "path_bbox_area",
    "path_bbox_cover_ratio"
  ]
}
```

### One-line usage

```javascript
function SCXPI_SVG_EVAL_WITH_BBOX(pack, text, svgText, opts) {
  var geom = SCXPI_GEOM_FROM_SVG_WITH_BBOX(svgText);
  return SCXPI_EVAL(pack, { text: String(text || ""), geom: geom }, opts || {});
}
```

If you want the next upgrade layer, say:
**“emit SCXPI_GEOM_SHAPE_HINTS_v1”** (square/rect/circle dominance, stroke-only vs filled, icon-likeness, diagram-likeness) built entirely from these cheap features + bbox ratios.

## SCXPI_GEOM_SHAPE_HINTS_v1 (SVG shape/intent hints)

This section derives **high-level semantic hints** (icon/diagram/text/dense/sparse/UI) from the geometry features.

### Shape hints contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-geom-shape-hints/v1",
  "@v": 1,
  "title": "SCXPI_GEOM_SHAPE_HINTS_v1 — SVG Shape & Intent Hints (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "hints"],
  "properties": {
    "@kind": { "const": "scxpi.geom.shape.hints.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "hints": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
          "name": { "type": "string" },
          "type": { "type": "string", "enum": ["boolean", "number"] },
          "range": { "type": "array", "minItems": 2, "maxItems": 2, "items": { "type": "number" } },
          "notes": { "type": "string" }
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
```

### Shape hints extractor (GAS-safe)

```javascript
/**
 * SCXPI_GEOM_SHAPE_HINTS_v1 — Derive high-level shape/intent hints
 * ------------------------------------------------------------
 * Inputs:
 *   geom — output of SCXPI_GEOM_FROM_SVG_WITH_BBOX (or without bbox; bbox hints degrade gracefully)
 *
 * Outputs:
 *   hints object with bounded scores in [0,1] and booleans.
 *
 * Design goals:
 *   - Deterministic
 *   - Cheap (no rendering)
 *   - Composable with π scoring + rules
 */

// ============================================================
// Public API
// ============================================================

/**
 * Compute shape hints from geometry features.
 * @param {Object} geom
 * @returns {Object} hints
 */
function SCXPI_GEOM_SHAPE_HINTS(geom) {
  geom = geom || {};

  // Safe getters
  var n = function (v) { v = Number(v); return isFinite(v) ? v : 0; };
  var b = function (v) { return !!v; };

  // Base counts
  var rect = n(geom.rect_count);
  var circle = n(geom.circle_count);
  var ellipse = n(geom.ellipse_count);
  var line = n(geom.line_count);
  var poly = n(geom.polyline_count) + n(geom.polygon_count);
  var path = n(geom.path_count);
  var text = n(geom.text_count);
  var group = n(geom.group_count);
  var use = n(geom.use_count);

  var element_total = n(geom.element_total) || (rect + circle + ellipse + line + poly + path + text);
  var len = n(geom.len);
  var density = n(geom.density);
  var cmd_total = n(geom.path_cmd_count_total);
  var cmd_density = n(geom.path_cmd_density);
  var symmetry = n(geom.symmetry_score);
  var repetition = n(geom.repetition_score);

  // BBox (optional)
  var bw = n(geom.path_bbox_w);
  var bh = n(geom.path_bbox_h);
  var bArea = n(geom.path_bbox_area);
  var cover = n(geom.path_bbox_cover_ratio);

  // Aspect ratio proxy
  var aspect = (bw > 0 && bh > 0) ? (Math.max(bw, bh) / Math.max(1, Math.min(bw, bh))) : 1;

  // ==========================================================
  // Hint scores (bounded [0,1])
  // ==========================================================

  // Icon-likeness:
  // Few elements, symmetry/repetition, bounded bbox cover, simple paths
  var icon_score =
    SCXPI__CLAMP01(
      0.35 * SCXPI__INV_SCALE(element_total, 1, 20) +
      0.25 * symmetry +
      0.20 * repetition +
      0.10 * SCXPI__INV_SCALE(cmd_density, 5, 80) +
      0.10 * SCXPI__MID_BAND(cover, 0.15, 0.85)
    );

  // Diagram-likeness:
  // Lines/paths, groups, moderate density, some text
  var diagram_score =
    SCXPI__CLAMP01(
      0.30 * SCXPI__SCALE(line + poly + path, 3, 200) +
      0.25 * SCXPI__SCALE(group + use, 1, 50) +
      0.20 * SCXPI__MID_BAND(density, 0.00001, 0.001) +
      0.15 * SCXPI__SCALE(text, 1, 30) +
      0.10 * SCXPI__MID_BAND(cover, 0.25, 0.9)
    );

  // Text-heavy:
  var text_heavy_score =
    SCXPI__CLAMP01(
      0.6 * SCXPI__SCALE(text, 3, 100) +
      0.2 * SCXPI__SCALE(len, 5000, 80000) +
      0.2 * SCXPI__INV_SCALE(cmd_total, 10, 200)
    );

  // Geometric-primitives dominance:
  // Rects/circles/ellipses over paths
  var primitive_total = rect + circle + ellipse;
  var primitive_score =
    SCXPI__CLAMP01(
      0.5 * SCXPI__RATIO(primitive_total, element_total) +
      0.3 * SCXPI__INV_SCALE(path, 1, 50) +
      0.2 * symmetry
    );

  // Stroke-only vs filled proxy:
  var stroke_only_score =
    SCXPI__CLAMP01(
      0.6 * SCXPI__SCALE(n(geom.stroke_count), 1, 200) -
      0.6 * SCXPI__SCALE(n(geom.fill_count), 1, 200) +
      0.4 * SCXPI__SCALE(line, 1, 200)
    );

  // Filled-heavy proxy:
  var filled_heavy_score =
    SCXPI__CLAMP01(
      0.6 * SCXPI__SCALE(n(geom.fill_count), 1, 200) +
      0.2 * SCXPI__INV_SCALE(line, 1, 200) +
      0.2 * SCXPI__MID_BAND(cover, 0.2, 1.0)
    );

  // Dense vs sparse:
  var dense_score =
    SCXPI__CLAMP01(
      0.5 * SCXPI__SCALE(cmd_total, 100, 2000) +
      0.3 * SCXPI__SCALE(density, 0.0002, 0.01) +
      0.2 * SCXPI__SCALE(element_total, 50, 500)
    );

  var sparse_score =
    SCXPI__CLAMP01(
      0.6 * SCXPI__INV_SCALE(element_total, 3, 50) +
      0.2 * SCXPI__INV_SCALE(cmd_total, 10, 300) +
      0.2 * SCXPI__INV_SCALE(len, 1000, 20000)
    );

  // UI-control hint (buttons/icons):
  // Rectangles + text OR icon-likeness + mid cover
  var ui_control_score =
    SCXPI__CLAMP01(
      0.4 * SCXPI__SCALE(rect, 1, 10) +
      0.3 * SCXPI__SCALE(text, 1, 5) +
      0.2 * icon_score +
      0.1 * SCXPI__MID_BAND(aspect, 1, 3)
    );

  // ==========================================================
  // Booleans (thresholded)
  // ==========================================================

  var is_icon = icon_score >= 0.6;
  var is_diagram = diagram_score >= 0.6;
  var is_text_heavy = text_heavy_score >= 0.6;
  var is_geometric = primitive_score >= 0.6;
  var is_dense = dense_score >= 0.6;
  var is_sparse = sparse_score >= 0.6;
  var is_ui_control = ui_control_score >= 0.6;

  return {
    "@kind": "scxpi.geom.shape.hints.result.v1",

    // Scores
    icon_score: icon_score,
    diagram_score: diagram_score,
    text_heavy_score: text_heavy_score,
    primitive_score: primitive_score,
    stroke_only_score: stroke_only_score,
    filled_heavy_score: filled_heavy_score,
    dense_score: dense_score,
    sparse_score: sparse_score,
    ui_control_score: ui_control_score,

    // Booleans
    is_icon: is_icon,
    is_diagram: is_diagram,
    is_text_heavy: is_text_heavy,
    is_geometric: is_geometric,
    is_dense: is_dense,
    is_sparse: is_sparse,
    is_ui_control: is_ui_control
  };
}

// ============================================================
// Helpers (bounded math)
// ============================================================

function SCXPI__CLAMP01(x) {
  return Math.max(0, Math.min(1, x));
}

function SCXPI__SCALE(v, lo, hi) {
  if (hi <= lo) return 0;
  return SCXPI__CLAMP01((v - lo) / (hi - lo));
}

function SCXPI__INV_SCALE(v, lo, hi) {
  if (hi <= lo) return 1;
  return SCXPI__CLAMP01(1 - ((v - lo) / (hi - lo)));
}

function SCXPI__RATIO(a, b) {
  if (b <= 0) return 0;
  return SCXPI__CLAMP01(a / b);
}

function SCXPI__MID_BAND(v, lo, hi) {
  // Peaks near middle of [lo,hi]
  if (hi <= lo) return 0;
  var mid = (lo + hi) / 2;
  var half = (hi - lo) / 2;
  var d = Math.abs(v - mid);
  return SCXPI__CLAMP01(1 - (d / Math.max(half, 1e-9)));
}
```

### Integration helper

```javascript
/**
 * Integration helper:
 * ------------------------------------------------------------
 * Use after SCXPI_GEOM_FROM_SVG_WITH_BBOX(svgText)
 * to enrich geom before SCXPI_EVAL.
 */
function SCXPI_GEOM_ENRICH_WITH_SHAPE_HINTS(geom) {
  var hints = SCXPI_GEOM_SHAPE_HINTS(geom);
  // Merge selected hints directly for rule access ($geom.*)
  geom.icon_score = hints.icon_score;
  geom.diagram_score = hints.diagram_score;
  geom.text_heavy_score = hints.text_heavy_score;
  geom.primitive_score = hints.primitive_score;
  geom.stroke_only_score = hints.stroke_only_score;
  geom.filled_heavy_score = hints.filled_heavy_score;
  geom.dense_score = hints.dense_score;
  geom.sparse_score = hints.sparse_score;
  geom.ui_control_score = hints.ui_control_score;

  geom.is_icon = hints.is_icon;
  geom.is_diagram = hints.is_diagram;
  geom.is_text_heavy = hints.is_text_heavy;
  geom.is_geometric = hints.is_geometric;
  geom.is_dense = hints.is_dense;
  geom.is_sparse = hints.is_sparse;
  geom.is_ui_control = hints.is_ui_control;

  return geom;
}
```

### End-to-end usage

```javascript
/**
 * End-to-end SVG eval with shape hints.
 */
function SCXPI_SVG_EVAL_WITH_HINTS(pack, text, svgText, opts) {
  var geom = SCXPI_GEOM_FROM_SVG_WITH_BBOX(svgText);
  geom = SCXPI_GEOM_ENRICH_WITH_SHAPE_HINTS(geom);
  return SCXPI_EVAL(pack, { text: String(text || ""), geom: geom }, opts || {});
}
```

### Shape hints outputs

```json
{
  "@kind": "scxpi.geom.shape.hints.v1",
  "@id": "asx://geom/svg/shape-hints/v1",
  "@v": 1,
  "hints": [
    { "name": "icon_score", "type": "number", "range": [0, 1], "notes": "Few elements, symmetry, repetition." },
    { "name": "diagram_score", "type": "number", "range": [0, 1], "notes": "Lines/paths/groups/text." },
    { "name": "text_heavy_score", "type": "number", "range": [0, 1], "notes": "Dominant text content." },
    { "name": "primitive_score", "type": "number", "range": [0, 1], "notes": "Rects/circles/ellipses over paths." },
    { "name": "stroke_only_score", "type": "number", "range": [0, 1], "notes": "Stroke-dominant drawings." },
    { "name": "filled_heavy_score", "type": "number", "range": [0, 1], "notes": "Fill-dominant drawings." },
    { "name": "dense_score", "type": "number", "range": [0, 1], "notes": "High command count/density." },
    { "name": "sparse_score", "type": "number", "range": [0, 1], "notes": "Low element/command counts." },
    { "name": "ui_control_score", "type": "number", "range": [0, 1], "notes": "Buttons/icons likelihood." },

    { "name": "is_icon", "type": "boolean", "range": [0, 1], "notes": "icon_score >= 0.6" },
    { "name": "is_diagram", "type": "boolean", "range": [0, 1], "notes": "diagram_score >= 0.6" },
    { "name": "is_text_heavy", "type": "boolean", "range": [0, 1], "notes": "text_heavy_score >= 0.6" },
    { "name": "is_geometric", "type": "boolean", "range": [0, 1], "notes": "primitive_score >= 0.6" },
    { "name": "is_dense", "type": "boolean", "range": [0, 1], "notes": "dense_score >= 0.6" },
    { "name": "is_sparse", "type": "boolean", "range": [0, 1], "notes": "sparse_score >= 0.6" },
    { "name": "is_ui_control", "type": "boolean", "range": [0, 1], "notes": "ui_control_score >= 0.6" }
  ]
}
```

### Shape-hint rules (example)

```json
{
  "id": "route_icons_local",
  "priority": 20,
  "if": { "gt": ["$geom.icon_score", 0.6] },
  "emit": ["ROUTE", "local"]
}
```

You now have **semantic shape understanding** (icon/diagram/UI/text/dense/sparse) built entirely from **symbolic geometry**, fully **GAS-native**, and ready to collapse via **π**.

## SCXPI_GEOM_LAYOUT_HINTS_v1 (SVG layout hints)

This section adds **layout/grid/alignment hints** using bbox + repetition + simple shape counts.

### Layout hints contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-geom-layout-hints/v1",
  "@v": 1,
  "title": "SCXPI_GEOM_LAYOUT_HINTS_v1 — SVG Layout Hints (GAS-safe, no rendering)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "hints"],
  "properties": {
    "@kind": { "const": "scxpi.geom.layout.hints.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "hints": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
          "name": { "type": "string" },
          "type": { "type": "string", "enum": ["number", "boolean", "integer"] },
          "range": { "type": "array", "minItems": 2, "maxItems": 2, "items": { "type": "number" } },
          "notes": { "type": "string" }
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
```

### Layout hints extractor (GAS-safe)

```javascript
/**
 * SCXPI_GEOM_LAYOUT_HINTS_v1 — Derive layout/grid/alignment hints
 * ------------------------------------------------------------
 * GAS-safe and deterministic. No DOM, no SVG rendering.
 *
 * Uses:
 *  - SVG width/height/viewBox
 *  - path bbox union (from SCXPI_GEOM_PATH_BBOX_v1)
 *  - repetition/symmetry
 *  - simple shape counts (rect/text/use/g)
 *
 * Outputs:
 *  - normalized bbox margins (left/right/top/bottom)
 *  - center alignment score
 *  - grid-likeness score + estimated grid columns/rows
 *  - whitespace score
 *  - header/footer/sidebar likelihood
 *
 * IMPORTANT:
 *  This is a "hint" layer: it guides rules/π, not ground truth.
 */

// ============================================================
// Public API
// ============================================================

/**
 * Compute layout hints from geom (preferably enriched with bbox + shape hints).
 * @param {Object} geom
 * @returns {Object} layout hints
 */
function SCXPI_GEOM_LAYOUT_HINTS(geom) {
  geom = geom || {};
  var n = function (v) { v = Number(v); return isFinite(v) ? v : 0; };
  var clamp01 = function (x) { return Math.max(0, Math.min(1, x)); };

  var W = n(geom.width);
  var H = n(geom.height);
  var svgArea = (W > 0 && H > 0) ? (W * H) : 0;

  // Prefer path bbox union if available; otherwise degrade to cover_ratio only.
  var minx = n(geom.path_bbox_minx);
  var miny = n(geom.path_bbox_miny);
  var maxx = n(geom.path_bbox_maxx);
  var maxy = n(geom.path_bbox_maxy);
  var bw = n(geom.path_bbox_w);
  var bh = n(geom.path_bbox_h);
  var cover = n(geom.path_bbox_cover_ratio);

  var hasBBox = !!(n(geom.path_bbox_count) > 0 && bw > 0 && bh > 0);

  // Normalized margins in [0,1] (fraction of width/height)
  var marginL = hasBBox && W > 0 ? clamp01(minx / W) : 0;
  var marginR = hasBBox && W > 0 ? clamp01((W - maxx) / W) : 0;
  var marginT = hasBBox && H > 0 ? clamp01(miny / H) : 0;
  var marginB = hasBBox && H > 0 ? clamp01((H - maxy) / H) : 0;

  // Content box center offset
  var cx = hasBBox ? (minx + maxx) / 2 : (W / 2);
  var cy = hasBBox ? (miny + maxy) / 2 : (H / 2);
  var dx = (W > 0) ? Math.abs(cx - (W / 2)) / W : 0;
  var dy = (H > 0) ? Math.abs(cy - (H / 2)) / H : 0;

  // Alignment scores
  var center_x_score = clamp01(1 - (dx * 4)); // within 25% => still okay
  var center_y_score = clamp01(1 - (dy * 4));
  var centered_score = clamp01((center_x_score + center_y_score) / 2);

  var symmetry = n(geom.symmetry_score);
  var repetition = n(geom.repetition_score);
  var useCount = n(geom.use_count);
  var groupCount = n(geom.group_count);
  var rectCount = n(geom.rect_count);
  var textCount = n(geom.text_count);

  // Whitespace score: high when cover ratio small OR bbox area small relative to svg
  var bboxArea = n(geom.path_bbox_area);
  var bboxFrac = (svgArea > 0 && bboxArea > 0) ? (bboxArea / svgArea) : cover; // fallback
  var whitespace_score = clamp01(1 - SCXPI__MID_BAND(bboxFrac, 0.15, 0.95)); // prefers neither extreme
  // Actually, whitespace: very low bboxFrac => lots of whitespace
  whitespace_score = clamp01(1 - Math.min(1, bboxFrac * 2.0)); // simple: bboxFrac 0.0 => 1.0 whitespace; 0.5 => 0.0

  // Header/footer/sidebar likelihood heuristics
  // header: content near top and wide
  var header_likelihood = hasBBox && H > 0 ? clamp01((1 - marginT) * SCXPI__MID_BAND(bw / Math.max(W, 1), 0.6, 1.0)) : 0;
  // footer: content near bottom and wide
  var footer_likelihood = hasBBox && H > 0 ? clamp01((1 - marginB) * SCXPI__MID_BAND(bw / Math.max(W, 1), 0.6, 1.0)) : 0;
  // sidebar: content narrow-ish and left or right aligned
  var sidebar_ratio = hasBBox && W > 0 ? (bw / W) : 0;
  var sidebar_side_bias = hasBBox ? Math.max(clamp01(1 - (marginL * 4)), clamp01(1 - (marginR * 4))) : 0;
  var sidebar_likelihood = clamp01(SCXPI__MID_BAND(sidebar_ratio, 0.12, 0.35) * sidebar_side_bias);

  // Grid-likeness:
  // - repetition via <use> and repeated paths helps
  // - multiple rects suggests UI grid/cards
  // - moderate symmetry helps
  // - not too text-heavy
  var textHeavy = n(geom.text_heavy_score); // if shape hints were merged
  var grid_score = clamp01(
    0.30 * SCXPI__SCALE(useCount, 1, 40) +
    0.25 * SCXPI__SCALE(rectCount, 2, 60) +
    0.20 * repetition +
    0.15 * symmetry +
    0.10 * (1 - SCXPI__SCALE(textCount, 10, 200)) -
    0.10 * (textHeavy || 0)
  );

  // Estimate columns/rows (VERY heuristic):
  // If it smells like a grid, guess cols from rect count and aspect.
  var cols = 1;
  var rows = 1;
  if (grid_score >= 0.45) {
    var r = Math.max(1, Math.round(Math.sqrt(Math.max(1, rectCount || useCount || 1))));
    // widen => more cols; tall => more rows
    var aspect2 = (W > 0 && H > 0) ? (W / H) : 1;
    cols = Math.max(1, Math.round(r * Math.max(0.8, Math.min(1.8, aspect2))));
    rows = Math.max(1, Math.ceil((Math.max(rectCount, useCount, 1)) / cols));
    // clamp to sane
    cols = Math.min(cols, 24);
    rows = Math.min(rows, 24);
  }

  // Gutter score: inferred from whitespace + grid + rect dominance
  var gutter_score = clamp01(
    0.45 * whitespace_score +
    0.35 * grid_score +
    0.20 * SCXPI__SCALE(rectCount, 2, 40)
  );

  // Layout archetype scores
  var card_grid_score = clamp01(0.6 * grid_score + 0.2 * gutter_score + 0.2 * SCXPI__SCALE(rectCount, 3, 80));
  var freeform_score = clamp01(1 - (0.6 * grid_score + 0.2 * symmetry + 0.2 * repetition));

  // Boolean flags
  var is_centered = centered_score >= 0.65;
  var is_grid = grid_score >= 0.60;
  var is_sidebar_layout = sidebar_likelihood >= 0.55;
  var is_header_layout = header_likelihood >= 0.55;
  var is_footer_layout = footer_likelihood >= 0.55;

  return {
    "@kind": "scxpi.geom.layout.hints.result.v1",

    // margins
    margin_left: marginL,
    margin_right: marginR,
    margin_top: marginT,
    margin_bottom: marginB,

    // alignment
    center_x_score: center_x_score,
    center_y_score: center_y_score,
    centered_score: centered_score,

    // whitespace / gutters
    whitespace_score: whitespace_score,
    gutter_score: gutter_score,

    // grid
    grid_score: grid_score,
    grid_cols_est: cols,
    grid_rows_est: rows,

    // layout archetypes
    card_grid_score: card_grid_score,
    freeform_score: freeform_score,

    // region likelihoods
    header_likelihood: header_likelihood,
    footer_likelihood: footer_likelihood,
    sidebar_likelihood: sidebar_likelihood,

    // flags
    is_centered: is_centered,
    is_grid: is_grid,
    is_sidebar_layout: is_sidebar_layout,
    is_header_layout: is_header_layout,
    is_footer_layout: is_footer_layout
  };
}

// ============================================================
// Helpers (reuse from prior layer if already present)
// ============================================================

function SCXPI__SCALE(v, lo, hi) {
  if (hi <= lo) return 0;
  return Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
}

function SCXPI__MID_BAND(v, lo, hi) {
  if (hi <= lo) return 0;
  var mid = (lo + hi) / 2;
  var half = (hi - lo) / 2;
  var d = Math.abs(v - mid);
  return Math.max(0, Math.min(1, 1 - (d / Math.max(half, 1e-9))));
}
```

### Integration helper

```javascript
/**
 * Integration: merge layout hints into geom for rule access ($geom.*).
 */
function SCXPI_GEOM_ENRICH_WITH_LAYOUT_HINTS(geom) {
  var h = SCXPI_GEOM_LAYOUT_HINTS(geom);

  geom.margin_left = h.margin_left;
  geom.margin_right = h.margin_right;
  geom.margin_top = h.margin_top;
  geom.margin_bottom = h.margin_bottom;

  geom.centered_score = h.centered_score;
  geom.center_x_score = h.center_x_score;
  geom.center_y_score = h.center_y_score;

  geom.whitespace_score = h.whitespace_score;
  geom.gutter_score = h.gutter_score;

  geom.grid_score = h.grid_score;
  geom.grid_cols_est = h.grid_cols_est;
  geom.grid_rows_est = h.grid_rows_est;

  geom.card_grid_score = h.card_grid_score;
  geom.freeform_score = h.freeform_score;

  geom.header_likelihood = h.header_likelihood;
  geom.footer_likelihood = h.footer_likelihood;
  geom.sidebar_likelihood = h.sidebar_likelihood;

  geom.is_centered = h.is_centered;
  geom.is_grid = h.is_grid;
  geom.is_sidebar_layout = h.is_sidebar_layout;
  geom.is_header_layout = h.is_header_layout;
  geom.is_footer_layout = h.is_footer_layout;

  return geom;
}
```

### End-to-end usage

```javascript
/**
 * End-to-end: SVG -> geom(bbox) -> shape hints -> layout hints -> SCXPI eval
 */
function SCXPI_SVG_EVAL_FULL_HINTS(pack, text, svgText, opts) {
  var geom = SCXPI_GEOM_FROM_SVG_WITH_BBOX(svgText);
  geom = SCXPI_GEOM_ENRICH_WITH_SHAPE_HINTS(geom);
  geom = SCXPI_GEOM_ENRICH_WITH_LAYOUT_HINTS(geom);
  return SCXPI_EVAL(pack, { text: String(text || ""), geom: geom }, opts || {});
}
```

### Layout hints outputs

```json
{
  "@kind": "scxpi.geom.layout.hints.v1",
  "@id": "asx://geom/svg/layout-hints/v1",
  "@v": 1,
  "hints": [
    { "name": "margin_left", "type": "number", "range": [0, 1], "notes": "Content left margin fraction of width." },
    { "name": "margin_right", "type": "number", "range": [0, 1], "notes": "Content right margin fraction of width." },
    { "name": "margin_top", "type": "number", "range": [0, 1], "notes": "Content top margin fraction of height." },
    { "name": "margin_bottom", "type": "number", "range": [0, 1], "notes": "Content bottom margin fraction of height." },

    { "name": "center_x_score", "type": "number", "range": [0, 1], "notes": "How centered content is horizontally." },
    { "name": "center_y_score", "type": "number", "range": [0, 1], "notes": "How centered content is vertically." },
    { "name": "centered_score", "type": "number", "range": [0, 1], "notes": "Mean of center_x/center_y." },

    { "name": "whitespace_score", "type": "number", "range": [0, 1], "notes": "High means lots of empty space." },
    { "name": "gutter_score", "type": "number", "range": [0, 1], "notes": "High suggests gutters/margins typical of UI layouts." },

    { "name": "grid_score", "type": "number", "range": [0, 1], "notes": "Likelihood the SVG encodes a grid/card layout." },
    { "name": "grid_cols_est", "type": "integer", "range": [1, 24], "notes": "Heuristic estimated columns." },
    { "name": "grid_rows_est", "type": "integer", "range": [1, 24], "notes": "Heuristic estimated rows." },

    { "name": "card_grid_score", "type": "number", "range": [0, 1], "notes": "Grid with card-like rect dominance + gutters." },
    { "name": "freeform_score", "type": "number", "range": [0, 1], "notes": "Opposite of grid-like structure." },

    { "name": "header_likelihood", "type": "number", "range": [0, 1], "notes": "Content wide and near top." },
    { "name": "footer_likelihood", "type": "number", "range": [0, 1], "notes": "Content wide and near bottom." },
    { "name": "sidebar_likelihood", "type": "number", "range": [0, 1], "notes": "Narrow content aligned to left/right." },

    { "name": "is_centered", "type": "boolean", "range": [0, 1], "notes": "centered_score >= 0.65" },
    { "name": "is_grid", "type": "boolean", "range": [0, 1], "notes": "grid_score >= 0.60" },
    { "name": "is_sidebar_layout", "type": "boolean", "range": [0, 1], "notes": "sidebar_likelihood >= 0.55" },
    { "name": "is_header_layout", "type": "boolean", "range": [0, 1], "notes": "header_likelihood >= 0.55" },
    { "name": "is_footer_layout", "type": "boolean", "range": [0, 1], "notes": "footer_likelihood >= 0.55" }
  ]
}
```

### Demo harness

```javascript
/**
 * Demo: layout hint sanity check.
 */
function SCXPI_LAYOUT_HINTS_DEMO() {
  // A fake "app card grid" SVG
  var svg = [
    '<svg viewBox="0 0 1200 800">',
    '<rect x="40" y="40" width="1120" height="720" fill="none" stroke="white"/>',
    '<g>',
    '<rect x="80" y="120" width="260" height="180"/><rect x="380" y="120" width="260" height="180"/><rect x="680" y="120" width="260" height="180"/><rect x="980" y="120" width="140" height="180"/>',
    '<rect x="80" y="340" width="260" height="180"/><rect x="380" y="340" width="260" height="180"/><rect x="680" y="340" width="260" height="180"/><rect x="980" y="340" width="140" height="180"/>',
    '</g>',
    '</svg>'
  ].join("");

  var geom = SCXPI_GEOM_FROM_SVG_WITH_BBOX(svg);
  geom = SCXPI_GEOM_ENRICH_WITH_SHAPE_HINTS(geom);
  geom = SCXPI_GEOM_ENRICH_WITH_LAYOUT_HINTS(geom);

  Logger.log(JSON.stringify({
    centered_score: geom.centered_score,
    grid_score: geom.grid_score,
    cols: geom.grid_cols_est,
    rows: geom.grid_rows_est,
    gutter: geom.gutter_score,
    header: geom.header_likelihood,
    sidebar: geom.sidebar_likelihood
  }, null, 2));

  return geom;
}
```

### Example rules using layout hints

```json
[
  { "id": "tag_grid", "priority": 20, "if": { "gt": ["$geom.grid_score", 0.6] }, "emit": ["TAG", "grid"] },
  { "id": "tag_sidebar", "priority": 20, "if": { "gt": ["$geom.sidebar_likelihood", 0.55] }, "emit": ["TAG", "sidebar"] },
  { "id": "tag_header", "priority": 20, "if": { "gt": ["$geom.header_likelihood", 0.55] }, "emit": ["TAG", "header"] },
  { "id": "route_local_for_ui", "priority": 10, "if": { "and": [
      { "gt": ["$geom.ui_control_score", 0.6] },
      { "gt": ["$geom.gutter_score", 0.6] }
    ]}, "emit": ["ROUTE", "local"] }
]
```

If you want the next “layout becomes data” leap: **SCXPI_GEOM_OBJECT_BBOXES_v1** — extract per-rect/circle/text bboxes (cheap attribute parsing) to build an actual **UI node list** (cards/buttons/headers) and feed it straight into your **XJSON layout compiler**.

## SCXPI_GEOM_OBJECT_BBOXES_v1 (per-object SVG bounding boxes)

This section extracts **per-object bounding boxes** for SVG elements to enable UI node lists without rendering.

### Object bbox contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-geom-object-bboxes/v1",
  "@v": 1,
  "title": "SCXPI_GEOM_OBJECT_BBOXES_v1 — Per-Object SVG Bounding Boxes (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "objects"],
  "properties": {
    "@kind": { "const": "scxpi.geom.object.bboxes.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "objects": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "type", "bbox"],
        "properties": {
          "id": { "type": "string" },
          "type": { "type": "string", "enum": ["rect", "circle", "ellipse", "line", "text", "image", "use", "path"] },
          "bbox": {
            "type": "object",
            "required": ["minx", "miny", "maxx", "maxy", "w", "h"],
            "properties": {
              "minx": { "type": "number" },
              "miny": { "type": "number" },
              "maxx": { "type": "number" },
              "maxy": { "type": "number" },
              "w": { "type": "number" },
              "h": { "type": "number" }
            }
          },
          "cx": { "type": "number" },
          "cy": { "type": "number" },
          "area": { "type": "number" },
          "meta": { "type": "object" }
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
```

### Object bbox extractor (GAS-safe)

```javascript
/**
 * SCXPI_GEOM_OBJECT_BBOXES_v1
 * ------------------------------------------------------------
 * Extract per-object bounding boxes from SVG text without rendering.
 * GAS-safe: regex + math only. Deterministic.
 *
 * Supported elements:
 *  rect, circle, ellipse, line, text, image, use, path (approx via SCXPI_PATH_BBOX_FROM_D)
 *
 * Notes:
 *  - text bbox is heuristic: uses x/y + font-size * text length factor.
 *  - use/image bbox relies on x/y/width/height attributes when present.
 *  - path bbox delegates to SCXPI_PATH_BBOX_FROM_D (approx).
 */

// ============================================================
// Public API
// ============================================================

function SCXPI_GEOM_OBJECT_BBOXES(svgText) {
  var svg = String(svgText || "");
  var out = {
    "@kind": "scxpi.geom.object.bboxes.result.v1",
    objects: []
  };

  var idCounter = 0;

  // RECT
  SCXPI__EACH_TAG(svg, "rect", function (attrs) {
    var x = N(attrs.x), y = N(attrs.y);
    var w = N(attrs.width), h = N(attrs.height);
    if (w > 0 && h > 0) {
      out.objects.push(SCXPI__OBJ("rect", ++idCounter, BBOX(x, y, x + w, y + h), attrs));
    }
  });

  // CIRCLE
  SCXPI__EACH_TAG(svg, "circle", function (attrs) {
    var cx = N(attrs.cx), cy = N(attrs.cy), r = N(attrs.r);
    if (r > 0) {
      out.objects.push(SCXPI__OBJ("circle", ++idCounter, BBOX(cx - r, cy - r, cx + r, cy + r), attrs));
    }
  });

  // ELLIPSE
  SCXPI__EACH_TAG(svg, "ellipse", function (attrs) {
    var cx = N(attrs.cx), cy = N(attrs.cy), rx = N(attrs.rx), ry = N(attrs.ry);
    if (rx > 0 && ry > 0) {
      out.objects.push(SCXPI__OBJ("ellipse", ++idCounter, BBOX(cx - rx, cy - ry, cx + rx, cy + ry), attrs));
    }
  });

  // LINE
  SCXPI__EACH_TAG(svg, "line", function (attrs) {
    var x1 = N(attrs.x1), y1 = N(attrs.y1), x2 = N(attrs.x2), y2 = N(attrs.y2);
    out.objects.push(SCXPI__OBJ("line", ++idCounter, BBOX(Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)), attrs));
  });

  // IMAGE / USE
  ["image", "use"].forEach(function (tag) {
    SCXPI__EACH_TAG(svg, tag, function (attrs) {
      var x = N(attrs.x), y = N(attrs.y);
      var w = N(attrs.width), h = N(attrs.height);
      if (w > 0 && h > 0) {
        out.objects.push(SCXPI__OBJ(tag, ++idCounter, BBOX(x, y, x + w, y + h), attrs));
      }
    });
  });

  // TEXT (heuristic)
  SCXPI__EACH_TAG(svg, "text", function (attrs, inner) {
    var x = N(attrs.x), y = N(attrs.y);
    var fs = N(attrs["font-size"]) || 16;
    var len = (inner || "").replace(/<[^>]*>/g, "").length;
    var w = fs * Math.max(1, len * 0.55);
    var h = fs * 1.2;
    out.objects.push(SCXPI__OBJ("text", ++idCounter, BBOX(x, y - h, x + w, y), attrs));
  });

  // PATH (approx)
  SCXPI__EACH_TAG(svg, "path", function (attrs) {
    var d = attrs.d || "";
    var bb = SCXPI_PATH_BBOX_FROM_D(d);
    if (bb && bb.valid) {
      out.objects.push(SCXPI__OBJ("path", ++idCounter, bb, { d_len: d.length }));
    }
  });

  return out;
}

// ============================================================
// Helpers
// ============================================================

function SCXPI__EACH_TAG(svg, tag, fn) {
  var re = new RegExp("<\\s*(?:[A-Za-z0-9_-]+:)?" + tag + "\\b([^>]*)>([\\s\\S]*?)<\\/\\s*(?:[A-Za-z0-9_-]+:)?" + tag + "\\s*>|<\\s*(?:[A-Za-z0-9_-]+:)?" + tag + "\\b([^>]*)\\/?>", "gi");
  var m;
  while ((m = re.exec(svg)) !== null) {
    var attrText = m[1] || m[3] || "";
    var inner = m[2] || "";
    fn(SCXPI__PARSE_ATTRS(attrText), inner);
  }
}

function SCXPI__PARSE_ATTRS(s) {
  var o = {};
  var re = /([a-zA-Z_:][\w:.-]*)\s*=\s*"([^"]*)"/g, m;
  while ((m = re.exec(s)) !== null) o[m[1]] = m[2];
  return o;
}

function SCXPI__OBJ(type, idn, bb, meta) {
  return {
    id: type + "_" + idn,
    type: type,
    bbox: { minx: bb.minx, miny: bb.miny, maxx: bb.maxx, maxy: bb.maxy, w: bb.w, h: bb.h },
    cx: (bb.minx + bb.maxx) / 2,
    cy: (bb.miny + bb.maxy) / 2,
    area: Math.max(0, bb.w * bb.h),
    meta: meta || {}
  };
}

function BBOX(minx, miny, maxx, maxy) {
  var w = maxx - minx, h = maxy - miny;
  return { minx: minx, miny: miny, maxx: maxx, maxy: maxy, w: w, h: h, valid: true };
}

function N(v) {
  v = Number(v);
  return isFinite(v) ? v : 0;
}
```

### Integration helper

```javascript
/**
 * Integration: enrich geom with object bboxes list for layout/UI reconstruction.
 */
function SCXPI_GEOM_ENRICH_WITH_OBJECT_BBOXES(geom, svgText) {
  var objs = SCXPI_GEOM_OBJECT_BBOXES(svgText);
  geom.object_bboxes = objs.objects;
  geom.object_count = objs.objects.length;
  return geom;
}
```

### Object bbox outputs

```json
{
  "@kind": "scxpi.geom.object.bboxes.v1",
  "@id": "asx://geom/svg/object-bboxes/v1",
  "@v": 1,
  "notes": "Per-object bounding boxes for rect/circle/ellipse/line/text/image/use/path. No rendering; path/text are heuristic.",
  "outputs": [
    "object_bboxes[].id",
    "object_bboxes[].type",
    "object_bboxes[].bbox.{minx,miny,maxx,maxy,w,h}",
    "object_bboxes[].cx",
    "object_bboxes[].cy",
    "object_bboxes[].area"
  ]
}
```

### Why this matters

This turns SVGs into a **UI node list** (cards, buttons, headers) that SCXPI rules + π can **collapse directly into layout primitives**—no tensors, no rendering, fully GAS-native.

If you want next: **SCXPI_GEOM_OBJECT_GRAPH_v1** (build adjacency/overlap/containment edges between these objects for auto-layout & interaction inference).

## SCXPI_GEOM_OBJECT_GRAPH_v1 (object adjacency/containment graph)

This section builds a **deterministic object graph** from per-object bounding boxes to infer containment, adjacency, and alignment relationships.

### Object graph contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-geom-object-graph/v1",
  "@v": 1,
  "title": "SCXPI_GEOM_OBJECT_GRAPH_v1 — Object Graph (Adjacency / Containment / Overlap) (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "graph"],
  "properties": {
    "@kind": { "const": "scxpi.geom.object.graph.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "graph": {
      "type": "object",
      "required": ["nodes", "edges", "stats"],
      "properties": {
        "nodes": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id", "type", "bbox"],
            "properties": {
              "id": { "type": "string" },
              "type": { "type": "string" },
              "bbox": {
                "type": "object",
                "required": ["minx", "miny", "maxx", "maxy", "w", "h"],
                "properties": {
                  "minx": { "type": "number" },
                  "miny": { "type": "number" },
                  "maxx": { "type": "number" },
                  "maxy": { "type": "number" },
                  "w": { "type": "number" },
                  "h": { "type": "number" }
                }
              },
              "cx": { "type": "number" },
              "cy": { "type": "number" },
              "area": { "type": "number" },
              "meta": { "type": "object" }
            },
            "additionalProperties": false
          }
        },
        "edges": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["a", "b", "type", "w"],
            "properties": {
              "a": { "type": "string" },
              "b": { "type": "string" },
              "type": { "type": "string", "enum": ["overlap", "contain", "adjacent", "near", "align_x", "align_y"] },
              "w": { "type": "number" },
              "meta": { "type": "object" }
            },
            "additionalProperties": false
          }
        },
        "stats": {
          "type": "object",
          "required": ["node_count", "edge_count"],
          "properties": {
            "node_count": { "type": "integer" },
            "edge_count": { "type": "integer" },
            "overlap_edges": { "type": "integer" },
            "contain_edges": { "type": "integer" },
            "adjacent_edges": { "type": "integer" },
            "near_edges": { "type": "integer" },
            "align_x_edges": { "type": "integer" },
            "align_y_edges": { "type": "integer" }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

### Object graph builder (GAS-safe)

```javascript
/**
 * SCXPI_GEOM_OBJECT_GRAPH_v1
 * ------------------------------------------------------------
 * Build a small, deterministic graph from per-object bboxes:
 * - overlap edges (IoU / intersection area ratio)
 * - containment edges (A contains B)
 * - adjacency edges (touching / small gap)
 * - near edges (distance threshold)
 * - alignment edges (centers aligned within epsilon)
 *
 * GAS-safe: O(n^2) pair loop (cap N to keep safe).
 *
 * Input:
 *  objects: array from SCXPI_GEOM_OBJECT_BBOXES(svg).objects
 *  dims: optional {width,height} to normalize thresholds
 *
 * Output:
 *  graph { nodes, edges, stats }
 */

// ============================================================
// Public API
// ============================================================

function SCXPI_GEOM_OBJECT_GRAPH(objects, dims, opts) {
  objects = objects || [];
  dims = dims || {};
  opts = opts || {};

  var W = num(dims.width), H = num(dims.height);
  var diag = Math.sqrt(Math.max(1, W * W + H * H)); // for distance normalization

  var MAX_N = opts.max_nodes || 300;   // safety
  var MAX_E = opts.max_edges || 5000;  // safety

  // Optionally prune to largest areas (layout-relevant)
  var nodes = objects.slice(0);
  if (nodes.length > MAX_N) {
    nodes.sort(function (a, b) { return (b.area || 0) - (a.area || 0); });
    nodes = nodes.slice(0, MAX_N);
  }

  var epsAlign = (opts.align_epsilon_px != null) ? num(opts.align_epsilon_px) : Math.max(2, Math.min(12, (Math.min(W, H) || 400) * 0.01));
  var nearPx = (opts.near_px != null) ? num(opts.near_px) : Math.max(6, Math.min(40, (Math.min(W, H) || 400) * 0.03));
  var gapPx = (opts.gap_px != null) ? num(opts.gap_px) : Math.max(2, Math.min(20, (Math.min(W, H) || 400) * 0.015));

  var overlapMin = (opts.overlap_min != null) ? num(opts.overlap_min) : 0.08; // IoU-like weight threshold
  var containPad = (opts.contain_pad_px != null) ? num(opts.contain_pad_px) : 0.5;

  var edges = [];
  var stats = {
    node_count: nodes.length,
    edge_count: 0,
    overlap_edges: 0,
    contain_edges: 0,
    adjacent_edges: 0,
    near_edges: 0,
    align_x_edges: 0,
    align_y_edges: 0
  };

  // Pairwise edges
  for (var i = 0; i < nodes.length; i++) {
    var A = nodes[i];
    for (var j = i + 1; j < nodes.length; j++) {
      var B = nodes[j];

      var bbA = A.bbox, bbB = B.bbox;
      if (!bbA || !bbB) continue;

      // Intersection / overlap
      var inter = SCXPI__INTERSECT(bbA, bbB);
      if (inter.area > 0) {
        var iou = inter.area / Math.max(1e-9, (bbA.w * bbA.h + bbB.w * bbB.h - inter.area));
        var wOverlap = clamp01(iou * 2); // boost a bit
        if (wOverlap >= overlapMin) {
          pushEdge(edges, stats, MAX_E, A.id, B.id, "overlap", wOverlap, { inter_area: inter.area, iou: iou });
        }
      }

      // Containment (A contains B or B contains A)
      var cAB = SCXPI__CONTAINS(bbA, bbB, containPad);
      var cBA = SCXPI__CONTAINS(bbB, bbA, containPad);
      if (cAB) {
        var wC = clamp01((bbB.w * bbB.h) / Math.max(1e-9, bbA.w * bbA.h)); // smaller inside bigger => smaller weight
        pushEdge(edges, stats, MAX_E, A.id, B.id, "contain", wC, { parent: A.id, child: B.id });
      } else if (cBA) {
        var wC2 = clamp01((bbA.w * bbA.h) / Math.max(1e-9, bbB.w * bbB.h));
        pushEdge(edges, stats, MAX_E, B.id, A.id, "contain", wC2, { parent: B.id, child: A.id });
      }

      // Adjacency (touching or small gap)
      var adj = SCXPI__ADJACENT(bbA, bbB, gapPx);
      if (adj.adjacent) {
        // weight based on closeness and facing length fraction
        var wAdj = clamp01(1 - (adj.gap / Math.max(1e-9, gapPx)));
        wAdj = clamp01((wAdj * 0.7) + (clamp01(adj.face_overlap / Math.max(1e-9, Math.min(bbA.w, bbA.h, bbB.w, bbB.h))) * 0.3));
        pushEdge(edges, stats, MAX_E, A.id, B.id, "adjacent", wAdj, adj);
      }

      // Near (center distance)
      var dx = (A.cx - B.cx), dy = (A.cy - B.cy);
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 0 && dist <= nearPx) {
        var wNear = clamp01(1 - (dist / Math.max(1e-9, nearPx)));
        pushEdge(edges, stats, MAX_E, A.id, B.id, "near", wNear, { dist_px: dist, dist_norm: dist / Math.max(1e-9, diag) });
      }

      // Alignment edges (center x or y within epsilon)
      if (Math.abs(A.cx - B.cx) <= epsAlign) {
        var wAx = clamp01(1 - (Math.abs(A.cx - B.cx) / Math.max(1e-9, epsAlign)));
        pushEdge(edges, stats, MAX_E, A.id, B.id, "align_x", wAx, { dx: A.cx - B.cx, eps: epsAlign });
      }
      if (Math.abs(A.cy - B.cy) <= epsAlign) {
        var wAy = clamp01(1 - (Math.abs(A.cy - B.cy) / Math.max(1e-9, epsAlign)));
        pushEdge(edges, stats, MAX_E, A.id, B.id, "align_y", wAy, { dy: A.cy - B.cy, eps: epsAlign });
      }

      if (edges.length >= MAX_E) break;
    }
    if (edges.length >= MAX_E) break;
  }

  stats.edge_count = edges.length;

  return {
    "@kind": "scxpi.geom.object.graph.result.v1",
    graph: {
      nodes: nodes,
      edges: edges,
      stats: stats
    }
  };
}

// ============================================================
// Edge + geometry helpers
// ============================================================

function pushEdge(edges, stats, MAX_E, a, b, type, w, meta) {
  if (edges.length >= MAX_E) return;
  edges.push({ a: a, b: b, type: type, w: w, meta: meta || {} });
  if (type === "overlap") stats.overlap_edges++;
  else if (type === "contain") stats.contain_edges++;
  else if (type === "adjacent") stats.adjacent_edges++;
  else if (type === "near") stats.near_edges++;
  else if (type === "align_x") stats.align_x_edges++;
  else if (type === "align_y") stats.align_y_edges++;
}

function SCXPI__INTERSECT(A, B) {
  var minx = Math.max(A.minx, B.minx);
  var miny = Math.max(A.miny, B.miny);
  var maxx = Math.min(A.maxx, B.maxx);
  var maxy = Math.min(A.maxy, B.maxy);
  var w = maxx - minx;
  var h = maxy - miny;
  var area = (w > 0 && h > 0) ? (w * h) : 0;
  return { minx: minx, miny: miny, maxx: maxx, maxy: maxy, w: Math.max(0, w), h: Math.max(0, h), area: area };
}

function SCXPI__CONTAINS(A, B, pad) {
  pad = num(pad);
  return (A.minx - pad <= B.minx) && (A.miny - pad <= B.miny) &&
         (A.maxx + pad >= B.maxx) && (A.maxy + pad >= B.maxy);
}

function SCXPI__ADJACENT(A, B, gapPx) {
  // adjacency if boxes are separated by small gap on one axis and overlap on the other
  var gap = Infinity;
  var faceOverlap = 0;
  var mode = null;

  // horizontal adjacency (A right near B left OR vice versa)
  var gap1 = Math.abs(A.maxx - B.minx);
  var gap2 = Math.abs(B.maxx - A.minx);
  var vOverlap = Math.max(0, Math.min(A.maxy, B.maxy) - Math.max(A.miny, B.miny));

  if (vOverlap > 0 && gap1 <= gapPx) { gap = gap1; faceOverlap = vOverlap; mode = "h"; }
  if (vOverlap > 0 && gap2 <= gapPx && gap2 < gap) { gap = gap2; faceOverlap = vOverlap; mode = "h"; }

  // vertical adjacency (A bottom near B top OR vice versa)
  var gap3 = Math.abs(A.maxy - B.miny);
  var gap4 = Math.abs(B.maxy - A.miny);
  var hOverlap = Math.max(0, Math.min(A.maxx, B.maxx) - Math.max(A.minx, B.minx));

  if (hOverlap > 0 && gap3 <= gapPx && gap3 < gap) { gap = gap3; faceOverlap = hOverlap; mode = "v"; }
  if (hOverlap > 0 && gap4 <= gapPx && gap4 < gap) { gap = gap4; faceOverlap = hOverlap; mode = "v"; }

  return { adjacent: isFinite(gap) && gap <= gapPx, gap: isFinite(gap) ? gap : 0, face_overlap: faceOverlap, axis: mode };
}

function num(v) { v = Number(v); return isFinite(v) ? v : 0; }
function clamp01(x) { return Math.max(0, Math.min(1, x)); }
```

### Integration helpers

```javascript
/**
 * Integration: build graph directly from SVG text.
 * Requires SCXPI_GEOM_OBJECT_BBOXES + SCXPI_PATH_BBOX_FROM_D.
 */
function SCXPI_GEOM_OBJECT_GRAPH_FROM_SVG(svgText, dims, opts) {
  var objs = SCXPI_GEOM_OBJECT_BBOXES(svgText);
  return SCXPI_GEOM_OBJECT_GRAPH(objs.objects, dims || {}, opts || {});
}
```

```javascript
/**
 * Optional enrichment: attach graph stats into geom for rule access.
 */
function SCXPI_GEOM_ENRICH_WITH_OBJECT_GRAPH_STATS(geom, graphResult) {
  geom = geom || {};
  var g = (graphResult && graphResult.graph) ? graphResult.graph : null;
  if (!g) return geom;

  var s = g.stats || {};
  geom.graph_node_count = s.node_count || 0;
  geom.graph_edge_count = s.edge_count || 0;
  geom.graph_overlap_edges = s.overlap_edges || 0;
  geom.graph_contain_edges = s.contain_edges || 0;
  geom.graph_adjacent_edges = s.adjacent_edges || 0;
  geom.graph_near_edges = s.near_edges || 0;
  geom.graph_align_x_edges = s.align_x_edges || 0;
  geom.graph_align_y_edges = s.align_y_edges || 0;

  return geom;
}
```

### Demo harness

```javascript
/**
 * Demo: object graph on a simple UI-ish SVG.
 */
function SCXPI_OBJECT_GRAPH_DEMO() {
  var svg = [
    '<svg viewBox="0 0 600 300">',
    '<rect x="20" y="20" width="560" height="260"/>',
    '<rect x="40" y="60" width="160" height="80"/>',
    '<rect x="220" y="60" width="160" height="80"/>',
    '<rect x="400" y="60" width="160" height="80"/>',
    '<text x="60" y="120" font-size="18">Card A</text>',
    '<text x="240" y="120" font-size="18">Card B</text>',
    '<text x="420" y="120" font-size="18">Card C</text>',
    '</svg>'
  ].join("");

  var dims = { width: 600, height: 300 };
  var res = SCXPI_GEOM_OBJECT_GRAPH_FROM_SVG(svg, dims, { max_nodes: 200, max_edges: 3000 });
  Logger.log(JSON.stringify(res.graph.stats, null, 2));
  // You can also inspect edges for "contain" (container->cards) and "align_y" for row alignment.
  return res;
}
```

### Object graph outputs

```json
{
  "@kind": "scxpi.geom.object.graph.v1",
  "@id": "asx://geom/svg/object-graph/v1",
  "@v": 1,
  "notes": "Builds an object graph from per-object bboxes. Edges: overlap (IoU), contain (A contains B), adjacent (touch/small gap), near (center distance), align_x/align_y (center alignment). Deterministic + GAS-safe with caps.",
  "outputs": [
    "graph.nodes[]",
    "graph.edges[].{a,b,type,w}",
    "graph.stats.{node_count,edge_count,overlap_edges,contain_edges,adjacent_edges,near_edges,align_x_edges,align_y_edges}"
  ],
  "recommended_defaults": {
    "max_nodes": 300,
    "max_edges": 5000,
    "align_epsilon_px": "max(2, min(12, min(width,height)*0.01))",
    "near_px": "max(6, min(40, min(width,height)*0.03))",
    "gap_px": "max(2, min(20, min(width,height)*0.015))",
    "overlap_min": 0.08,
    "contain_pad_px": 0.5
  }
}
```

### What this enables immediately

- **Container detection** (`contain` edges) → panels/cards/modals
- **Row/column inference** (`align_y` / `align_x` + adjacency) → grid reconstruction
- **Layering** (`overlap`) → stacks, badges, icons-on-buttons
- **Proximity clustering** (`near`) → component grouping

If you want the next step, say: **SCXPI_LAYOUT_RECONSTRUCTOR_v1** — it will collapse this graph into a small XJSON layout tree (containers → rows/cols → items) with deterministic rules.

## SCXPI_LAYOUT_RECONSTRUCTOR_v1 (SVG → layout tree)

This section turns object graphs into a **compact layout tree** (XJSON-ish) without rendering.

### Layout reconstructor contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-layout-reconstructor/v1",
  "@v": 1,
  "title": "SCXPI_LAYOUT_RECONSTRUCTOR_v1 — SVG→Layout Tree (XJSON-ish) (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "layout"],
  "properties": {
    "@kind": { "const": "scxpi.layout.reconstructor.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "layout": {
      "type": "object",
      "required": ["root"],
      "properties": {
        "root": { "type": "object" },
        "nodes": { "type": "array" },
        "stats": { "type": "object" }
      }
    }
  },
  "additionalProperties": false
}
```

### Layout reconstructor (GAS-safe)

```javascript
/**
 * SCXPI_LAYOUT_RECONSTRUCTOR_v1
 * ------------------------------------------------------------
 * Deterministic, no-render SVG layout reconstruction using:
 *  - object bboxes (SCXPI_GEOM_OBJECT_BBOXES_v1)
 *  - object graph edges (SCXPI_GEOM_OBJECT_GRAPH_v1)
 *  - layout hints (optional)
 *
 * Output: a compact layout tree (XJSON-ish) you can route into your DOM compiler.
 *
 * Layout node types:
 *  - "canvas"   : svg root
 *  - "container": inferred panels/cards/frames (contain edges)
 *  - "row"      : horizontal group (align_y)
 *  - "col"      : vertical group (align_x)
 *  - "item"     : leaf object mapped from svg element
 *
 * IMPORTANT LIMITATIONS (by design):
 *  - No transforms parsing (translate/scale/rotate) in this v1
 *  - No path-to-shape conversion
 *  - Only bbox-based layout inference
 */

// ============================================================
// Public API
// ============================================================

/**
 * Reconstruct layout tree from SVG (end-to-end).
 * @param {string} svgText
 * @param {Object} dims {width,height}
 * @param {Object=} opts
 * @returns {Object} result {layout:{root,nodes,stats}}
 */
function SCXPI_LAYOUT_RECONSTRUCT_FROM_SVG(svgText, dims, opts) {
  dims = dims || {};
  opts = opts || {};

  var objsRes = SCXPI_GEOM_OBJECT_BBOXES(svgText);
  var objects = (objsRes && objsRes.objects) ? objsRes.objects : [];

  var graphRes = SCXPI_GEOM_OBJECT_GRAPH(objects, dims, opts.graph || {});
  var graph = graphRes.graph;

  return SCXPI_LAYOUT_RECONSTRUCT(objects, graph, dims, opts);
}

/**
 * Reconstruct layout tree from objects + graph.
 * @param {Array} objects
 * @param {Object} graph {nodes,edges,stats}
 * @param {Object} dims {width,height}
 * @param {Object=} opts
 * @returns {Object}
 */
function SCXPI_LAYOUT_RECONSTRUCT(objects, graph, dims, opts) {
  objects = objects || [];
  graph = graph || { nodes: objects, edges: [], stats: {} };
  dims = dims || {};
  opts = opts || {};

  var W = num(dims.width), H = num(dims.height);

  var MAX_DEPTH = opts.max_depth || 6;
  var MIN_CONTAINER_CHILDREN = opts.min_container_children || 2;
  var CONTAIN_MIN_W = (opts.contain_min_w != null) ? num(opts.contain_min_w) : 0.12; // container must cover >=12% of svg width
  var CONTAIN_MIN_H = (opts.contain_min_h != null) ? num(opts.contain_min_h) : 0.12;

  // Index nodes
  var byId = {};
  for (var i = 0; i < objects.length; i++) byId[objects[i].id] = objects[i];

  // Build containment parent->children map from edges
  var contains = {};        // parentId -> [childId...]
  var parentOf = {};        // childId -> bestParentId (smallest containing)
  var containEdges = graph.edges.filter(function (e) { return e.type === "contain"; });

  // Sort containment edges by parent area ascending (smallest container wins as parent)
  containEdges.sort(function (e1, e2) {
    var A1 = byId[e1.a], A2 = byId[e2.a];
    return (A1 ? A1.area : 0) - (A2 ? A2.area : 0);
  });

  for (var c = 0; c < containEdges.length; c++) {
    var e = containEdges[c];
    var p = byId[e.a], ch = byId[e.b];
    if (!p || !ch) continue;

    // Filter: ignore containers that are too tiny relative to canvas
    if (W > 0 && H > 0) {
      var pw = p.bbox.w / W;
      var ph = p.bbox.h / H;
      if (pw < CONTAIN_MIN_W || ph < CONTAIN_MIN_H) continue;
    }

    // Assign parent if none yet, or if this parent is smaller (tighter)
    if (!parentOf[ch.id]) {
      parentOf[ch.id] = p.id;
    } else {
      var curP = byId[parentOf[ch.id]];
      if (curP && p.area < curP.area) parentOf[ch.id] = p.id;
    }
  }

  // Build contains map from chosen parentOf
  Object.keys(parentOf).forEach(function (childId) {
    var pId = parentOf[childId];
    if (!contains[pId]) contains[pId] = [];
    contains[pId].push(childId);
  });

  // Identify container candidates: have enough children
  var containerIds = Object.keys(contains).filter(function (pid) {
    return (contains[pid] || []).length >= MIN_CONTAINER_CHILDREN;
  });

  // Root canvas node
  var layoutNodes = [];
  var rootId = "layout_canvas_1";
  var root = {
    id: rootId,
    type: "canvas",
    bbox: { minx: 0, miny: 0, maxx: W, maxy: H, w: W, h: H },
    children: []
  };
  layoutNodes.push(root);

  // Top-level items = nodes without parents OR whose parent is not a kept container
  var keptContainer = {};
  containerIds.forEach(function (id) { keptContainer[id] = true; });

  var topLevel = [];
  for (var k = 0; k < objects.length; k++) {
    var oid = objects[k].id;
    var p = parentOf[oid];
    if (!p || !keptContainer[p]) topLevel.push(oid);
  }

  // Build tree recursively
  for (var t = 0; t < topLevel.length; t++) {
    var child = SCXPI__BUILD_SUBTREE(topLevel[t], 1);
    if (child) root.children.push(child.id);
  }

  // Post-pass: for each container, cluster its direct children into rows/cols
  // using align edges + adjacency.
  var edgeIndex = SCXPI__EDGE_INDEX(graph.edges);
  for (var n = 0; n < layoutNodes.length; n++) {
    var node = layoutNodes[n];
    if (node.type !== "container") continue;
    var kids = node.children.slice(0);
    var groups = SCXPI__GROUP_IN_CONTAINER(kids, edgeIndex, byId, opts.grouping || {});
    // Replace children with grouping nodes + item nodes in order
    node.children = groups.children;
    // Append grouping nodes created
    for (var g = 0; g < groups.newNodes.length; g++) layoutNodes.push(groups.newNodes[g]);
  }

  // Stats
  var stats = {
    object_count: objects.length,
    container_count: containerIds.length,
    layout_node_count: layoutNodes.length,
    depth_cap: MAX_DEPTH
  };

  return {
    "@kind": "scxpi.layout.reconstructor.result.v1",
    layout: {
      root: root,
      nodes: layoutNodes,
      stats: stats
    }
  };

  // ----------------------------------------------------------
  // inner: build subtree for one object id
  function SCXPI__BUILD_SUBTREE(objId, depth) {
    if (depth > MAX_DEPTH) return SCXPI__LEAF(objId);

    var obj = byId[objId];
    if (!obj) return null;

    // If this object is a kept container, create a container layout node
    if (keptContainer[objId]) {
      var cid = "layout_container_" + objId;
      var node = {
        id: cid,
        type: "container",
        ref: objId,
        bbox: cloneB(obj.bbox),
        children: []
      };
      layoutNodes.push(node);

      var kids = (contains[objId] || []).slice(0);
      // sort by y then x for deterministic ordering
      kids.sort(function (a, b) {
        var A = byId[a], B = byId[b];
        if (!A || !B) return 0;
        if (A.cy !== B.cy) return A.cy - B.cy;
        return A.cx - B.cx;
      });

      for (var i2 = 0; i2 < kids.length; i2++) {
        var ch = SCXPI__BUILD_SUBTREE(kids[i2], depth + 1);
        if (ch) node.children.push(ch.id);
      }
      return node;
    }

    return SCXPI__LEAF(objId);
  }

  function SCXPI__LEAF(objId) {
    var obj = byId[objId];
    if (!obj) return null;
    var id = "layout_item_" + objId;
    var node = {
      id: id,
      type: "item",
      ref: objId,
      item_type: obj.type,
      bbox: cloneB(obj.bbox),
      meta: obj.meta || {}
    };
    layoutNodes.push(node);
    return node;
  }

  function cloneB(bb) {
    return { minx: bb.minx, miny: bb.miny, maxx: bb.maxx, maxy: bb.maxy, w: bb.w, h: bb.h };
  }
}

// ============================================================
// Grouping inside containers: rows/cols via align edges
// ============================================================

function SCXPI__GROUP_IN_CONTAINER(childLayoutIds, edgeIndex, byId, opts) {
  opts = opts || {};
  var alignMin = (opts.align_min != null) ? num(opts.align_min) : 0.7;
  var nearMin = (opts.near_min != null) ? num(opts.near_min) : 0.4;
  var adjMin = (opts.adj_min != null) ? num(opts.adj_min) : 0.4;

  // Map layout_item_* ids back to object ids (layout_item_<objId>)
  var childObjIds = childLayoutIds.map(function (lid) {
    return (String(lid).indexOf("layout_item_") === 0) ? String(lid).slice("layout_item_".length) : null;
  }).filter(Boolean);

  // Build union-find clusters for rows (align_y strong)
  var rowUF = SCXPI__UF(childObjIds);
  for (var i = 0; i < childObjIds.length; i++) {
    for (var j = i + 1; j < childObjIds.length; j++) {
      var a = childObjIds[i], b = childObjIds[j];
      var ay = edgeIndex.getWeight(a, b, "align_y");
      var adj = edgeIndex.getWeight(a, b, "adjacent");
      var near = edgeIndex.getWeight(a, b, "near");
      if (ay >= alignMin || (adj >= adjMin && near >= nearMin)) rowUF.union(a, b);
    }
  }
  var rowClusters = rowUF.clusters();

  // For each row cluster, sort members by cx and wrap in row node if size>1
  var newNodes = [];
  var newChildren = [];
  var rowIdx = 0;

  // Deterministic order: by cluster top-left
  rowClusters.sort(function (c1, c2) {
    return SCXPI__CLUSTER_KEY(c1, byId) - SCXPI__CLUSTER_KEY(c2, byId);
  });

  for (var r = 0; r < rowClusters.length; r++) {
    var cluster = rowClusters[r];
    if (cluster.length <= 1) {
      // single item
      var only = cluster[0];
      newChildren.push("layout_item_" + only);
      continue;
    }

    // create row node
    rowIdx++;
    var rowId = "layout_row_" + rowIdx + "_" + hashish(cluster.join("|"));
    var bb = SCXPI__UNION_BBOX(cluster, byId);

    // sort members left->right
    cluster.sort(function (a, b) {
      return (byId[a] ? byId[a].cx : 0) - (byId[b] ? byId[b].cx : 0);
    });

    var rowNode = {
      id: rowId,
      type: "row",
      bbox: bb,
      children: cluster.map(function (oid) { return "layout_item_" + oid; })
    };
    newNodes.push(rowNode);
    newChildren.push(rowId);
  }

  return { newNodes: newNodes, children: newChildren };
}

function SCXPI__UNION_BBOX(objIds, byId) {
  var minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (var i = 0; i < objIds.length; i++) {
    var o = byId[objIds[i]];
    if (!o) continue;
    minx = Math.min(minx, o.bbox.minx);
    miny = Math.min(miny, o.bbox.miny);
    maxx = Math.max(maxx, o.bbox.maxx);
    maxy = Math.max(maxy, o.bbox.maxy);
  }
  if (!isFinite(minx)) minx = miny = maxx = maxy = 0;
  return { minx: minx, miny: miny, maxx: maxx, maxy: maxy, w: Math.max(0, maxx - minx), h: Math.max(0, maxy - miny) };
}

// Cluster sort key: top-left weighted
function SCXPI__CLUSTER_KEY(ids, byId) {
  var bb = SCXPI__UNION_BBOX(ids, byId);
  return (bb.miny * 1000000) + bb.minx;
}

// ============================================================
// Edge index (weights lookup)
// ============================================================

function SCXPI__EDGE_INDEX(edges) {
  var map = {}; // key "a|b|type" -> w
  for (var i = 0; i < edges.length; i++) {
    var e = edges[i];
    var a = e.a, b = e.b, t = e.type, w = num(e.w);
    map[a + "|" + b + "|" + t] = Math.max(map[a + "|" + b + "|" + t] || 0, w);
    map[b + "|" + a + "|" + t] = Math.max(map[b + "|" + a + "|" + t] || 0, w);
  }
  return {
    getWeight: function (a, b, t) { return num(map[a + "|" + b + "|" + t]); }
  };
}

// ============================================================
// Union-Find
// ============================================================

function SCXPI__UF(items) {
  var parent = {};
  items.forEach(function (x) { parent[x] = x; });

  function find(x) {
    var p = parent[x];
    if (p === x) return x;
    parent[x] = find(p);
    return parent[x];
  }
  function union(a, b) {
    var ra = find(a), rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  }
  function clusters() {
    var groups = {};
    items.forEach(function (x) {
      var r = find(x);
      if (!groups[r]) groups[r] = [];
      groups[r].push(x);
    });
    return Object.keys(groups).map(function (k) { return groups[k]; });
  }
  return { find: find, union: union, clusters: clusters };
}

// ============================================================
// Utils
// ============================================================

function num(v) { v = Number(v); return isFinite(v) ? v : 0; }

// Tiny deterministic hash-ish for IDs (not crypto)
function hashish(s) {
  s = String(s || "");
  var h = 2166136261;
  for (var i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return ("00000000" + h.toString(16)).slice(-8);
}
```

### Layout reconstructor outputs

```json
{
  "@kind": "scxpi.layout.reconstructor.v1",
  "@id": "asx://layout/reconstructor/v1",
  "@v": 1,
  "notes": "Deterministic bbox-based SVG→layout reconstruction. Builds container nodes from contain edges, then groups child items into row nodes using align_y + adjacency/near edges. Emits XJSON-ish layout tree suitable for DOM/layout compiler.",
  "outputs": [
    "layout.root",
    "layout.nodes[]",
    "layout.stats"
  ],
  "recommended_defaults": {
    "max_depth": 6,
    "min_container_children": 2,
    "contain_min_w": 0.12,
    "contain_min_h": 0.12,
    "graph": {
      "max_nodes": 300,
      "max_edges": 5000,
      "align_epsilon_px": "max(2, min(12, min(width,height)*0.01))",
      "near_px": "max(6, min(40, min(width,height)*0.03))",
      "gap_px": "max(2, min(20, min(width,height)*0.015))"
    },
    "grouping": {
      "align_min": 0.7,
      "near_min": 0.4,
      "adj_min": 0.4
    }
  }
}
```

### What you get (example shape)

- `canvas`
  - `container` (panel/card)
    - `row`
      - `item` (rect/text/path…)
      - `item`
    - `row`
      - `item` …

That’s enough to:

- emit a **UI tree**
- classify regions (header/sidebar/cards)
- feed it into **SCXPI rules + π collapse**
- compile to your **Atomic layout / XJSON DOM**

If you want the next upgrade, say: **SCXPI_LAYOUT_TO_XJSON_UI_v1** — it will take this layout tree and emit your actual **UI schema** (cards/buttons/text blocks) with deterministic tags and style hints (grid cols, gutters, primary/secondary regions).

## SCXPI_LAYOUT_TO_XJSON_UI_v1 (layout tree → XJSON UI model)

This section converts the layout tree into a **portable XJSON UI model** with normalized bboxes and minimal UI atoms.

### Layout-to-UI contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-layout-to-xjson-ui/v1",
  "@v": 1,
  "title": "SCXPI_LAYOUT_TO_XJSON_UI_v1 — Layout Tree → XJSON UI Model (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "ui"],
  "properties": {
    "@kind": { "const": "scxpi.layout.to.xjson.ui.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "ui": { "type": "object" }
  },
  "additionalProperties": false
}
```

### Layout-to-UI converter (GAS-safe)

```javascript
/**
 * SCXPI_LAYOUT_TO_XJSON_UI_v1
 * ------------------------------------------------------------
 * Convert SCXPI layout tree + extracted SVG objects into an XJSON UI model.
 *
 * Goal:
 *  - Produce a deterministic, portable UI description:
 *      @ui.root -> containers/rows/items
 *  - Attach style/layout hints as state (not behavior)
 *  - Emit minimal "atoms" (card, row, text, icon, shape, image, button-ish)
 *
 * Input:
 *  layoutRes: output of SCXPI_LAYOUT_RECONSTRUCT_* (layout.root/layout.nodes)
 *  objectsRes: output of SCXPI_GEOM_OBJECT_BBOXES(svg) (objects[])
 *  geom: optional geom enriched with hints (shape/layout scores)
 *  dims: {width,height}
 *
 * Output:
 *  { "@kind": "...result.v1", ui: { ...XJSON... } }
 *
 * NOTE:
 *  This is a UI *model*, not a renderer.
 */

// ============================================================
// Public API
// ============================================================

function SCXPI_LAYOUT_TO_XJSON_UI(layoutRes, objectsRes, geom, dims, opts) {
  opts = opts || {};
  geom = geom || {};
  dims = dims || {};

  var W = num(dims.width), H = num(dims.height);

  var layout = (layoutRes && layoutRes.layout) ? layoutRes.layout : null;
  if (!layout || !layout.root || !layout.nodes) {
    return { "@kind": "scxpi.layout.to.xjson.ui.result.v1", ui: SCXPI__EMPTY_UI(W, H) };
  }

  var objects = (objectsRes && objectsRes.objects) ? objectsRes.objects : (objectsRes || []);
  var objById = {};
  for (var i = 0; i < objects.length; i++) objById[objects[i].id] = objects[i];

  var nodeById = {};
  for (var j = 0; j < layout.nodes.length; j++) nodeById[layout.nodes[j].id] = layout.nodes[j];

  // --- build UI nodes recursively ---
  var uiNodes = {};
  var uiRoot = SCXPI__LAYOUT_NODE_TO_UI(layout.root, nodeById, objById, uiNodes, geom, { W: W, H: H }, opts);

  // Attach global hints (as state)
  var global = {
    width: W,
    height: H,
    centered_score: num(geom.centered_score),
    grid_score: num(geom.grid_score),
    grid_cols_est: num(geom.grid_cols_est),
    grid_rows_est: num(geom.grid_rows_est),
    gutter_score: num(geom.gutter_score),
    whitespace_score: num(geom.whitespace_score),
    icon_score: num(geom.icon_score),
    diagram_score: num(geom.diagram_score),
    ui_control_score: num(geom.ui_control_score)
  };

  // XJSON-ish UI envelope
  var ui = {
    "@kind": "xjson.ui.v1",
    "@v": 1,
    "@meta": {
      generator: "SCXPI_LAYOUT_TO_XJSON_UI_v1",
      source: "svg",
      dims: { width: W, height: H }
    },
    "@state": {
      layout: global
    },
    "@ui": {
      root: uiRoot,
      nodes: uiNodes
    }
  };

  return { "@kind": "scxpi.layout.to.xjson.ui.result.v1", ui: ui };
}

// ============================================================
// Core conversion
// ============================================================

function SCXPI__LAYOUT_NODE_TO_UI(layoutNode, nodeById, objById, uiNodes, geom, canvas, opts) {
  if (!layoutNode) return null;

  // Determine UI node type
  var t = layoutNode.type;
  var bbox = layoutNode.bbox || { minx: 0, miny: 0, maxx: 0, maxy: 0, w: 0, h: 0 };

  if (t === "canvas") {
    var id = "ui_canvas_1";
    uiNodes[id] = {
      id: id,
      type: "canvas",
      bbox: SCXPI__NORM_BBOX(bbox, canvas),
      children: SCXPI__MAP_CHILDREN(layoutNode.children, nodeById, objById, uiNodes, geom, canvas, opts)
    };
    return id;
  }

  if (t === "container") {
    var cId = "ui_container_" + safe(layoutNode.ref || layoutNode.id);
    uiNodes[cId] = {
      id: cId,
      type: SCXPI__CONTAINER_CLASS(layoutNode, geom, canvas),
      bbox: SCXPI__NORM_BBOX(bbox, canvas),
      style: SCXPI__CONTAINER_STYLE_HINTS(layoutNode, geom, canvas),
      children: SCXPI__MAP_CHILDREN(layoutNode.children, nodeById, objById, uiNodes, geom, canvas, opts)
    };
    return cId;
  }

  if (t === "row") {
    var rId = "ui_row_" + safe(layoutNode.id);
    uiNodes[rId] = {
      id: rId,
      type: "row",
      bbox: SCXPI__NORM_BBOX(bbox, canvas),
      layout: { dir: "x", align: "center", gap_hint: SCXPI__GAP_HINT(geom, canvas) },
      children: SCXPI__MAP_CHILDREN(layoutNode.children, nodeById, objById, uiNodes, geom, canvas, opts)
    };
    return rId;
  }

  if (t === "col") {
    var colId = "ui_col_" + safe(layoutNode.id);
    uiNodes[colId] = {
      id: colId,
      type: "col",
      bbox: SCXPI__NORM_BBOX(bbox, canvas),
      layout: { dir: "y", align: "start", gap_hint: SCXPI__GAP_HINT(geom, canvas) },
      children: SCXPI__MAP_CHILDREN(layoutNode.children, nodeById, objById, uiNodes, geom, canvas, opts)
    };
    return colId;
  }

  // Leaf item maps from underlying SVG object
  if (t === "item") {
    var obj = objById[layoutNode.ref];
    var uiItem = SCXPI__OBJECT_TO_UI_ATOM(obj, geom, canvas, opts);
    uiNodes[uiItem.id] = uiItem;
    return uiItem.id;
  }

  // Fallback
  var fId = "ui_unknown_" + safe(layoutNode.id);
  uiNodes[fId] = { id: fId, type: "unknown", bbox: SCXPI__NORM_BBOX(bbox, canvas), children: [] };
  return fId;
}

function SCXPI__MAP_CHILDREN(childLayoutIds, nodeById, objById, uiNodes, geom, canvas, opts) {
  childLayoutIds = childLayoutIds || [];
  var out = [];
  for (var i = 0; i < childLayoutIds.length; i++) {
    var lid = childLayoutIds[i];
    var ln = nodeById[lid];
    if (!ln) continue;
    var uid = SCXPI__LAYOUT_NODE_TO_UI(ln, nodeById, objById, uiNodes, geom, canvas, opts);
    if (uid) out.push(uid);
  }
  return out;
}

// ============================================================
// Atom mapping (SVG object → UI primitive)
// ============================================================

function SCXPI__OBJECT_TO_UI_ATOM(obj, geom, canvas, opts) {
  obj = obj || {};
  var t = String(obj.type || "unknown");
  var id = "ui_item_" + safe(obj.id || ("obj_" + hashish(JSON.stringify(obj))));

  var bb = obj.bbox || { minx: 0, miny: 0, maxx: 0, maxy: 0, w: 0, h: 0 };
  var nbb = SCXPI__NORM_BBOX(bb, canvas);

  // classify into atom types
  var atomType = "shape";
  var role = "decor";

  if (t === "text") {
    atomType = "text";
    role = "label";
  } else if (t === "image" || t === "use") {
    atomType = "image";
    role = "media";
  } else if (t === "line") {
    atomType = "divider";
    role = "structure";
  } else if (t === "circle" || t === "ellipse") {
    atomType = "icon";
    role = "decor";
  } else if (t === "rect") {
    // a rect can be a card/button field; use size/aspect to hint
    var aspect = (nbb.h > 0) ? (nbb.w / nbb.h) : 1;
    var area = nbb.w * nbb.h;
    if (area < 0.02 && aspect > 2.0) { atomType = "button"; role = "action"; }
    else if (area > 0.08) { atomType = "card"; role = "container_surface"; }
    else { atomType = "shape"; role = "decor"; }
  } else if (t === "path") {
    // treat as icon if icon_score high globally or the path is small
    var small = (nbb.w * nbb.h) < 0.02;
    if (num(geom.icon_score) > 0.6 || small) { atomType = "icon"; role = "decor"; }
    else atomType = "shape";
  }

  return {
    id: id,
    type: atomType,
    role: role,
    source: { kind: "svg", ref: obj.id || null, svg_type: t },
    bbox: nbb,
    style: SCXPI__ATOM_STYLE_HINTS(atomType, obj, geom, canvas)
  };
}

// ============================================================
// Style hints (state only)
// ============================================================

function SCXPI__CONTAINER_CLASS(layoutNode, geom, canvas) {
  // Very light heuristic mapping
  if (num(geom.sidebar_likelihood) > 0.55) return "sidebar";
  if (num(geom.header_likelihood) > 0.55) return "header";
  if (num(geom.footer_likelihood) > 0.55) return "footer";
  if (num(geom.card_grid_score) > 0.6) return "panel";
  return "container";
}

function SCXPI__CONTAINER_STYLE_HINTS(layoutNode, geom, canvas) {
  return {
    padding_hint: SCXPI__PAD_HINT(geom, canvas),
    radius_hint: SCXPI__RADIUS_HINT(geom),
    elevation_hint: SCXPI__ELEVATION_HINT(geom),
    grid_hint: {
      grid_score: num(geom.grid_score),
      cols_est: num(geom.grid_cols_est),
      rows_est: num(geom.grid_rows_est),
      gutter_score: num(geom.gutter_score)
    }
  };
}

function SCXPI__ATOM_STYLE_HINTS(atomType, obj, geom, canvas) {
  var st = {
    emphasis: 0,
    fill_like: (num(geom.filled_heavy_score) > 0.6),
    stroke_like: (num(geom.stroke_only_score) > 0.6)
  };

  if (atomType === "button") {
    st.emphasis = 0.7;
    st.padding_hint = SCXPI__PAD_HINT(geom, canvas);
    st.radius_hint = 0.6;
  } else if (atomType === "card") {
    st.emphasis = 0.5;
    st.radius_hint = 0.7;
    st.elevation_hint = 0.6;
  } else if (atomType === "text") {
    st.emphasis = 0.4;
    st.font_size_hint = num((obj.meta || {})["font-size"]) || 0;
  } else if (atomType === "icon") {
    st.emphasis = 0.3;
    st.size_hint = SCXPI__SIZE_HINT(obj, canvas);
  }

  return st;
}

function SCXPI__PAD_HINT(geom, canvas) {
  // 0..1 scale, use gutter_score as proxy
  var g = num(geom.gutter_score);
  return clamp01(0.2 + 0.7 * g);
}

function SCXPI__GAP_HINT(geom, canvas) {
  var g = num(geom.gutter_score);
  return clamp01(0.15 + 0.75 * g);
}

function SCXPI__RADIUS_HINT(geom) {
  // geometric primitives often imply rounded corners in UI-ish icons
  return clamp01(0.2 + 0.6 * num(geom.ui_control_score));
}

function SCXPI__ELEVATION_HINT(geom) {
  return clamp01(0.1 + 0.7 * num(geom.card_grid_score));
}

function SCXPI__SIZE_HINT(obj, canvas) {
  var bb = obj.bbox || { w: 0, h: 0 };
  var W = Math.max(1, num(canvas.W));
  var H = Math.max(1, num(canvas.H));
  var a = (bb.w * bb.h) / (W * H);
  return clamp01(a * 10); // small icons map to small hint
}

// ============================================================
// Normalization + envelope
// ============================================================

function SCXPI__NORM_BBOX(bb, canvas) {
  var W = Math.max(1, num(canvas.W));
  var H = Math.max(1, num(canvas.H));
  return {
    x: clamp01(num(bb.minx) / W),
    y: clamp01(num(bb.miny) / H),
    w: clamp01(num(bb.w) / W),
    h: clamp01(num(bb.h) / H)
  };
}

function SCXPI__EMPTY_UI(W, H) {
  return {
    "@kind": "xjson.ui.v1",
    "@v": 1,
    "@meta": { generator: "SCXPI_LAYOUT_TO_XJSON_UI_v1", source: "svg", dims: { width: W || 0, height: H || 0 } },
    "@state": { layout: {} },
    "@ui": { root: "ui_canvas_1", nodes: { "ui_canvas_1": { id: "ui_canvas_1", type: "canvas", bbox: { x: 0, y: 0, w: 1, h: 1 }, children: [] } } }
  };
}

// ============================================================
// Utils
// ============================================================

function num(v) { v = Number(v); return isFinite(v) ? v : 0; }
function clamp01(x) { return Math.max(0, Math.min(1, x)); }
function safe(s) { return String(s || "").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 120); }

// Tiny deterministic hash-ish for IDs (not crypto)
function hashish(s) {
  s = String(s || "");
  var h = 2166136261;
  for (var i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return ("00000000" + h.toString(16)).slice(-8);
}
```

### End-to-end demo

```javascript
/**
 * End-to-end demo: SVG -> objects -> graph -> layout -> XJSON UI.
 * Requires:
 *  - SCXPI_GEOM_FROM_SVG_WITH_BBOX
 *  - SCXPI_GEOM_ENRICH_WITH_SHAPE_HINTS
 *  - SCXPI_GEOM_ENRICH_WITH_LAYOUT_HINTS
 *  - SCXPI_GEOM_OBJECT_BBOXES
 *  - SCXPI_GEOM_OBJECT_GRAPH
 *  - SCXPI_LAYOUT_RECONSTRUCT_FROM_SVG
 *  - SCXPI_LAYOUT_TO_XJSON_UI
 */
function SCXPI_LAYOUT_TO_UI_DEMO() {
  var svg = [
    '<svg viewBox="0 0 600 300">',
    '<rect x="20" y="20" width="560" height="260"/>',
    '<rect x="40" y="60" width="160" height="80"/>',
    '<rect x="220" y="60" width="160" height="80"/>',
    '<rect x="400" y="60" width="160" height="80"/>',
    '<text x="60" y="110" font-size="18">Card A</text>',
    '<text x="240" y="110" font-size="18">Card B</text>',
    '<text x="420" y="110" font-size="18">Card C</text>',
    '</svg>'
  ].join("");

  var dims = { width: 600, height: 300 };

  // geom + hints
  var geom = SCXPI_GEOM_FROM_SVG_WITH_BBOX(svg);
  geom = SCXPI_GEOM_ENRICH_WITH_SHAPE_HINTS(geom);
  geom = SCXPI_GEOM_ENRICH_WITH_LAYOUT_HINTS(geom);

  // layout
  var layoutRes = SCXPI_LAYOUT_RECONSTRUCT_FROM_SVG(svg, dims, {});
  var objectsRes = SCXPI_GEOM_OBJECT_BBOXES(svg);

  // ui
  var uiRes = SCXPI_LAYOUT_TO_XJSON_UI(layoutRes, objectsRes, geom, dims, {});
  Logger.log(JSON.stringify(uiRes.ui, null, 2));
  return uiRes;
}
```

### Layout-to-UI outputs

```json
{
  "@kind": "scxpi.layout.to.xjson.ui.v1",
  "@id": "asx://layout/to-xjson-ui/v1",
  "@v": 1,
  "notes": "Converts SCXPI layout tree + object bboxes into an XJSON UI model (canvas/container/row/col + atoms: card/button/text/icon/shape/image). Emits normalized bbox + style hints as state only. Designed for GAS execution.",
  "outputs": [
    "ui.@ui.root",
    "ui.@ui.nodes[*].{id,type,bbox,children,style,source}",
    "ui.@state.layout"
  ]
}
```

### What this gives you immediately

- A **portable UI graph** your DOM layer can render anywhere (browser, server, replay)
- Deterministic **UI atoms** (card/button/text/icon) derived from pure geometry
- A clean seam for SCXPI rules/π to **override** classifications (“this rect is a sidebar”, “this group is a grid”, etc.)

If you want the final glue, say: **SCXPI_UI_TO_ATOMIC_CSS_HINTS_v1** — it will emit Atomic class suggestions + :root variables (padding/gap/radius/elevation) from this UI model.

## SCXPI_UI_TO_ATOMIC_CSS_HINTS_v1 (XJSON UI → Atomic CSS hints)

This section emits **state-only Atomic CSS hints** from the XJSON UI model for downstream renderers.

### UI-to-Atomic contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-ui-to-atomic-css-hints/v1",
  "@v": 1,
  "title": "SCXPI_UI_TO_ATOMIC_CSS_HINTS_v1 — XJSON UI → Atomic CSS Hints (State-Only, GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "atomic"],
  "properties": {
    "@kind": { "const": "scxpi.ui.to.atomic.css.hints.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "atomic": { "type": "object" }
  },
  "additionalProperties": false
}
```

### Atomic CSS hint generator (GAS-safe)

```javascript
/**
 * SCXPI_UI_TO_ATOMIC_CSS_HINTS_v1
 * ------------------------------------------------------------
 * Converts an XJSON UI model into:
 *  1) Atomic CSS class suggestions (semantic, non-executable)
 *  2) :root variable hints (spacing, radius, elevation, grid)
 *
 * IMPORTANT:
 *  - Emits hints only (state). No CSS rules, no DOM ops.
 *  - Deterministic, replayable, GAS-safe.
 *  - Designed to plug into ATOMIC.CSS / atomic.xjson.
 *
 * Input:
 *  uiRes: result of SCXPI_LAYOUT_TO_XJSON_UI_v1 (ui)
 *
 * Output:
 *  {
 *    "@kind": "scxpi.ui.to.atomic.css.hints.result.v1",
 *    atomic: {
 *      root_vars: {...},
 *      nodes: { <uiNodeId>: { classes:[], vars:{} } }
 *    }
 *  }
 */

// ============================================================
// Public API
// ============================================================

function SCXPI_UI_TO_ATOMIC_CSS_HINTS(uiRes, opts) {
  opts = opts || {};

  var ui = (uiRes && uiRes.ui) ? uiRes.ui : uiRes;
  if (!ui || !ui["@ui"] || !ui["@ui"].nodes) {
    return {
      "@kind": "scxpi.ui.to.atomic.css.hints.result.v1",
      atomic: { root_vars: {}, nodes: {} }
    };
  }

  var nodes = ui["@ui"].nodes;
  var state = ui["@state"] || {};
  var layout = state.layout || {};

  // ----------------------------------------------------------
  // Root-level CSS variables (global layout feel)
  // ----------------------------------------------------------

  var rootVars = {
    "--asx-gap": SCXPI__GAP_VAR(layout),
    "--asx-pad": SCXPI__PAD_VAR(layout),
    "--asx-radius": SCXPI__RADIUS_VAR(layout),
    "--asx-elevation": SCXPI__ELEVATION_VAR(layout),
    "--asx-grid-cols": SCXPI__GRID_COLS(layout),
    "--asx-grid-rows": SCXPI__GRID_ROWS(layout),
    "--asx-grid-gutter": SCXPI__GRID_GUTTER(layout),
    "--asx-density": clamp01(1 - num(layout.whitespace_score))
  };

  // ----------------------------------------------------------
  // Per-node atomic hints
  // ----------------------------------------------------------

  var atomicNodes = {};

  Object.keys(nodes).forEach(function (id) {
    var n = nodes[id];
    atomicNodes[id] = SCXPI__NODE_TO_ATOMIC(n, layout);
  });

  return {
    "@kind": "scxpi.ui.to.atomic.css.hints.result.v1",
    atomic: {
      root_vars: rootVars,
      nodes: atomicNodes
    }
  };
}

// ============================================================
// Node → Atomic mapping
// ============================================================

function SCXPI__NODE_TO_ATOMIC(node, layout) {
  var classes = [];
  var vars = {};

  // ----------------------------------------------------------
  // Structural classes
  // ----------------------------------------------------------

  if (node.type === "canvas") {
    classes.push("asx-canvas");
  }

  if (node.type === "container" || node.type === "panel") {
    classes.push("asx-container");
    classes.push("asx-surface");
  }

  if (node.type === "sidebar") {
    classes.push("asx-sidebar");
    classes.push("asx-vertical");
  }

  if (node.type === "header") {
    classes.push("asx-header");
    classes.push("asx-horizontal");
  }

  if (node.type === "footer") {
    classes.push("asx-footer");
    classes.push("asx-horizontal");
  }

  if (node.type === "row") {
    classes.push("asx-row");
    classes.push("asx-flex");
    vars["--asx-flow"] = "row";
  }

  if (node.type === "col") {
    classes.push("asx-col");
    classes.push("asx-flex");
    vars["--asx-flow"] = "column";
  }

  // ----------------------------------------------------------
  // Atom classes
  // ----------------------------------------------------------

  switch (node.type) {
    case "card":
      classes.push("asx-card");
      vars["--asx-radius"] = clamp01(0.4 + 0.5 * num(layout.card_grid_score));
      vars["--asx-elevation"] = clamp01(0.4 + 0.4 * num(layout.card_grid_score));
      break;

    case "button":
      classes.push("asx-btn");
      classes.push("asx-action");
      vars["--asx-radius"] = clamp01(0.5 + 0.3 * num(layout.ui_control_score));
      vars["--asx-pad"] = clamp01(0.4 + 0.4 * num(layout.gutter_score));
      break;

    case "text":
      classes.push("asx-text");
      vars["--asx-font-scale"] = SCXPI__FONT_SCALE(node, layout);
      break;

    case "icon":
      classes.push("asx-icon");
      vars["--asx-icon-scale"] = clamp01(0.6 + 0.6 * num(layout.icon_score));
      break;

    case "divider":
      classes.push("asx-divider");
      vars["--asx-opacity"] = clamp01(0.3 + 0.3 * num(layout.whitespace_score));
      break;

    case "image":
      classes.push("asx-media");
      break;
  }

  // ----------------------------------------------------------
  // Emphasis / visual weight
  // ----------------------------------------------------------

  if (node.style && node.style.emphasis != null) {
    vars["--asx-emphasis"] = clamp01(node.style.emphasis);
    if (node.style.emphasis > 0.6) classes.push("asx-primary");
    else if (node.style.emphasis > 0.3) classes.push("asx-secondary");
    else classes.push("asx-muted");
  }

  // ----------------------------------------------------------
  // Layout hints → CSS vars
  // ----------------------------------------------------------

  if (node.style && node.style.padding_hint != null) {
    vars["--asx-pad"] = clamp01(node.style.padding_hint);
  }
  if (node.style && node.style.radius_hint != null) {
    vars["--asx-radius"] = clamp01(node.style.radius_hint);
  }
  if (node.style && node.style.elevation_hint != null) {
    vars["--asx-elevation"] = clamp01(node.style.elevation_hint);
  }

  // ----------------------------------------------------------
  // Grid inference
  // ----------------------------------------------------------

  if (num(layout.grid_score) > 0.55) {
    classes.push("asx-grid");
    vars["--asx-grid-cols"] = SCXPI__GRID_COLS(layout);
    vars["--asx-grid-rows"] = SCXPI__GRID_ROWS(layout);
    vars["--asx-grid-gap"] = SCXPI__GRID_GUTTER(layout);
  }

  return {
    classes: uniq(classes),
    vars: vars
  };
}

// ============================================================
// Root variable derivation
// ============================================================

function SCXPI__GAP_VAR(layout) {
  return clamp01(0.2 + 0.6 * num(layout.gutter_score));
}

function SCXPI__PAD_VAR(layout) {
  return clamp01(0.25 + 0.55 * num(layout.gutter_score));
}

function SCXPI__RADIUS_VAR(layout) {
  return clamp01(0.15 + 0.6 * num(layout.ui_control_score));
}

function SCXPI__ELEVATION_VAR(layout) {
  return clamp01(0.1 + 0.7 * num(layout.card_grid_score));
}

function SCXPI__GRID_COLS(layout) {
  var c = Math.round(num(layout.grid_cols_est));
  return (c > 0) ? c : 1;
}

function SCXPI__GRID_ROWS(layout) {
  var r = Math.round(num(layout.grid_rows_est));
  return (r > 0) ? r : "auto";
}

function SCXPI__GRID_GUTTER(layout) {
  return clamp01(0.2 + 0.6 * num(layout.gutter_score));
}

function SCXPI__FONT_SCALE(node, layout) {
  // Prefer explicit hint, else derive from bbox height
  if (node.style && node.style.font_size_hint) {
    return clamp01(num(node.style.font_size_hint) / 24);
  }
  return clamp01(0.8 + 0.4 * num(layout.ui_control_score));
}

// ============================================================
// Utils
// ============================================================

function num(v) { v = Number(v); return isFinite(v) ? v : 0; }
function clamp01(x) { return Math.max(0, Math.min(1, x)); }
function uniq(a) {
  var o = {}, out = [];
  for (var i = 0; i < a.length; i++) if (!o[a[i]]) { o[a[i]] = 1; out.push(a[i]); }
  return out;
}
```

### Atomic CSS hint outputs

```json
{
  "@kind": "scxpi.ui.to.atomic.css.hints.v1",
  "@id": "asx://ui/to-atomic-css-hints/v1",
  "@v": 1,
  "notes": "Maps XJSON UI nodes to Atomic CSS semantic classes and :root / per-node CSS variable hints. Emits state-only styling signals (gap, padding, radius, elevation, grid). No executable CSS. GAS-safe.",
  "outputs": [
    "atomic.root_vars",
    "atomic.nodes[*].classes",
    "atomic.nodes[*].vars"
  ],
  "atomic_alignment": [
    "Atomic CSS 4-Block Rule",
    "Variables-as-State",
    "No imperative behavior",
    "DOM renderer decides final CSS"
  ]
}
```

### What you now have (end-to-end)

**SVG → Geometry → Layout → UI → Atomic CSS hints**

- Geometry decides *what exists*
- Layout decides *structure*
- UI decides *semantics*
- Atomic hints decide *visual state*

No tensors. No safetensors. No JS frameworks required.

If you want the final lock, next logical piece is: **SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON_v1** — emitting a ready-to-merge `atomic.xjson` fragment that your Ghost / ASXR shells can consume directly.

## SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON_v1 (Atomic hints → atomic.xjson fragment)

This section converts Atomic CSS hints into a **deterministic atomic.xjson fragment** for merging.

### Atomic fragment contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-atomic-hints-to-atomic-xjson/v1",
  "@v": 1,
  "title": "SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON_v1 — Atomic CSS Hints → atomic.xjson Fragment (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "atomic_xjson"],
  "properties": {
    "@kind": { "const": "scxpi.atomic.hints.to.atomic.xjson.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "atomic_xjson": { "type": "object" }
  },
  "additionalProperties": false
}
```

### Atomic fragment emitter (GAS-safe)

```javascript
/**
 * SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON_v1
 * ------------------------------------------------------------
 * Takes:
 *  - SCXPI_UI_TO_ATOMIC_CSS_HINTS_v1 output (atomic.root_vars + atomic.nodes)
 * And emits:
 *  - an atomic.xjson fragment suitable for merge into your canonical atomic.xjson
 *
 * Constraints:
 *  - state-only (vars + classes)
 *  - deterministic ordering
 *  - GAS-safe, no external schemas
 *
 * Output Shape (fragment):
 *  {
 *    "@kind": "atomic.xjson.fragment.v1",
 *    "@v": 1,
 *    "@atomic": {
 *      ":root": { "vars": {...} },
 *      "nodes": {
 *        "<nodeId>": { "classes": [...], "vars": {...} }
 *      }
 *    }
 *  }
 */

// ============================================================
// Public API
// ============================================================

function SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON(atomicHintsRes, opts) {
  opts = opts || {};
  var atomic = (atomicHintsRes && atomicHintsRes.atomic) ? atomicHintsRes.atomic : atomicHintsRes;
  atomic = atomic || {};

  var rootVars = atomic.root_vars || {};
  var nodes = atomic.nodes || {};

  // Deterministic key ordering
  var rootVarsOrdered = SCXPI__ORDERED_OBJ(rootVars);
  var nodesOrdered = SCXPI__ORDERED_NODES(nodes);

  var fragmentId = opts.id || ("atomic_fragment_" + SCXPI__HASHISH(JSON.stringify({ rootVarsOrdered: rootVarsOrdered, nodeCount: Object.keys(nodesOrdered).length })));

  var frag = {
    "@kind": "atomic.xjson.fragment.v1",
    "@v": 1,
    "@meta": {
      generator: "SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON_v1",
      id: fragmentId,
      merge_mode: (opts.merge_mode || "overlay"),
      notes: "Fragment contains :root vars + per-node class/var hints only."
    },
    "@atomic": {
      ":root": {
        vars: rootVarsOrdered
      },
      nodes: nodesOrdered
    }
  };

  return {
    "@kind": "scxpi.atomic.hints.to.atomic.xjson.result.v1",
    atomic_xjson: frag
  };
}

// ============================================================
// Deterministic ordering helpers
// ============================================================

function SCXPI__ORDERED_OBJ(o) {
  o = o || {};
  var keys = Object.keys(o).sort();
  var out = {};
  for (var i = 0; i < keys.length; i++) out[keys[i]] = o[keys[i]];
  return out;
}

function SCXPI__ORDERED_NODES(nodes) {
  nodes = nodes || {};
  var ids = Object.keys(nodes).sort();
  var out = {};
  for (var i = 0; i < ids.length; i++) {
    var id = ids[i];
    var n = nodes[id] || {};
    out[id] = {
      classes: SCXPI__ORDERED_ARRAY(n.classes || []),
      vars: SCXPI__ORDERED_OBJ(n.vars || {})
    };
  }
  return out;
}

function SCXPI__ORDERED_ARRAY(a) {
  a = a || [];
  var out = a.slice(0).sort();
  // de-dupe deterministically
  var seen = {};
  var uniq = [];
  for (var i = 0; i < out.length; i++) {
    var v = String(out[i]);
    if (!seen[v]) { seen[v] = 1; uniq.push(v); }
  }
  return uniq;
}

// ============================================================
// Tiny deterministic hash-ish (not crypto)
// ============================================================

function SCXPI__HASHISH(s) {
  s = String(s || "");
  var h = 2166136261;
  for (var i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return ("00000000" + h.toString(16)).slice(-8);
}
```

### End-to-end fragment emitter

```javascript
/**
 * End-to-end emitter: SVG -> UI -> Atomic hints -> atomic.xjson fragment
 * Requires previous layers present.
 */
function SCXPI_SVG_TO_ATOMIC_XJSON_FRAGMENT(svgText, dims, opts) {
  opts = opts || {};
  dims = dims || {};

  // geom + hints
  var geom = SCXPI_GEOM_FROM_SVG_WITH_BBOX(svgText);
  geom = SCXPI_GEOM_ENRICH_WITH_SHAPE_HINTS(geom);
  geom = SCXPI_GEOM_ENRICH_WITH_LAYOUT_HINTS(geom);

  // objects/layout/ui
  var objectsRes = SCXPI_GEOM_OBJECT_BBOXES(svgText);
  var layoutRes = SCXPI_LAYOUT_RECONSTRUCT_FROM_SVG(svgText, dims, {});
  var uiRes = SCXPI_LAYOUT_TO_XJSON_UI(layoutRes, objectsRes, geom, dims, {});

  // atomic hints
  var atomicHints = SCXPI_UI_TO_ATOMIC_CSS_HINTS(uiRes, {});
  // fragment
  return SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON(atomicHints, opts);
}
```

### Atomic fragment outputs

```json
{
  "@kind": "scxpi.atomic.hints.to.atomic.xjson.v1",
  "@id": "asx://atomic/hints-to-atomic-xjson/v1",
  "@v": 1,
  "notes": "Converts SCXPI atomic CSS hints (root_vars + per-node class/var hints) into a deterministic atomic.xjson fragment for merge into canonical atomic.xjson. State-only; no rendering; GAS-safe.",
  "outputs": [
    "atomic_xjson.@atomic.:root.vars",
    "atomic_xjson.@atomic.nodes[*].classes",
    "atomic_xjson.@atomic.nodes[*].vars",
    "atomic_xjson.@meta.merge_mode"
  ],
  "merge_guidance": {
    "merge_mode": "overlay",
    "overlay_rule": "Fragment vars override existing vars for same keys; classes unioned per node id; unknown node ids appended."
  }
}
```

### Merge target (canonical atomic.xjson)

This fragment is designed to drop into your canonical structure as:

- `@atomic.:root.vars` → Variables-as-State
- `@atomic.nodes[nodeId].classes/vars` → per-node style projections

Next (if you want the full pipeline lock): **SCXPI_ATOMIC_XJSON_MERGE_v1** — a deterministic merge function that takes canonical `atomic.xjson` + this fragment and outputs a new `atomic.xjson` with a proof hash.

## SCXPI_ATOMIC_XJSON_MERGE_v1 (deterministic atomic.xjson merger)

This section merges the fragment into canonical `atomic.xjson` with proof metadata.

### Atomic merge contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-atomic-xjson-merge/v1",
  "@v": 1,
  "title": "SCXPI_ATOMIC_XJSON_MERGE_v1 — Deterministic atomic.xjson Merger (Proofed, GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "merged"],
  "properties": {
    "@kind": { "const": "scxpi.atomic.xjson.merge.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "merged": { "type": "object" }
  },
  "additionalProperties": false
}
```

### Atomic merge implementation (GAS-safe)

```javascript
/**
 * SCXPI_ATOMIC_XJSON_MERGE_v1
 * ------------------------------------------------------------
 * Deterministically merges:
 *   - baseAtomicXjson (canonical atomic.xjson)
 *   - fragment (from SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON_v1)
 *
 * Laws:
 *  - State-only (vars + classes)
 *  - Deterministic ordering
 *  - No schema fetches
 *  - Proof hash emitted
 *
 * Merge Rules (LOCKED):
 *  1) :root.vars
 *     - overlay: fragment overrides base on same key
 *  2) nodes[nodeId].classes
 *     - union (set), sorted
 *  3) nodes[nodeId].vars
 *     - overlay: fragment overrides base on same key
 *  4) Unknown nodeIds are appended
 *
 * Output:
 *  {
 *    "@kind": "scxpi.atomic.xjson.merge.result.v1",
 *    merged: <atomic.xjson>,
 *    proof: { hash, inputs }
 *  }
 */

// ============================================================
// Public API
// ============================================================

function SCXPI_ATOMIC_XJSON_MERGE(baseAtomicXjson, fragment, opts) {
  opts = opts || {};

  var base = deepClone(baseAtomicXjson || {});
  var frag = (fragment && fragment.atomic_xjson) ? fragment.atomic_xjson : fragment;

  // Normalize envelopes
  base = SCXPI__ENSURE_ATOMIC_ENVELOPE(base);
  frag = SCXPI__ENSURE_FRAGMENT_ENVELOPE(frag);

  // ----------------------------------------------------------
  // Merge :root vars
  // ----------------------------------------------------------

  var baseRootVars = (((base["@atomic"] || {})[":root"] || {}).vars) || {};
  var fragRootVars = (((frag["@atomic"] || {})[":root"] || {}).vars) || {};

  var mergedRootVars = SCXPI__MERGE_VARS(baseRootVars, fragRootVars);

  // ----------------------------------------------------------
  // Merge nodes
  // ----------------------------------------------------------

  var baseNodes = (base["@atomic"] || {}).nodes || {};
  var fragNodes = (frag["@atomic"] || {}).nodes || {};

  var mergedNodes = {};
  var nodeIds = uniq(Object.keys(baseNodes).concat(Object.keys(fragNodes))).sort();

  for (var i = 0; i < nodeIds.length; i++) {
    var id = nodeIds[i];
    var b = baseNodes[id] || {};
    var f = fragNodes[id] || {};

    mergedNodes[id] = {
      classes: SCXPI__MERGE_CLASSES(b.classes || [], f.classes || []),
      vars: SCXPI__MERGE_VARS(b.vars || {}, f.vars || {})
    };
  }

  // ----------------------------------------------------------
  // Assemble merged atomic.xjson
  // ----------------------------------------------------------

  var merged = {
    "@kind": "atomic.xjson.v1",
    "@v": (base["@v"] || 1),
    "@meta": SCXPI__MERGE_META(base["@meta"], frag["@meta"], opts),
    "@atomic": {
      ":root": { vars: SCXPI__ORDERED_OBJ(mergedRootVars) },
      nodes: SCXPI__ORDERED_NODES(mergedNodes)
    }
  };

  // ----------------------------------------------------------
  // Proof
  // ----------------------------------------------------------

  var proof = {
    algo: "fnv1a-32",
    hash: SCXPI__HASHISH(JSON.stringify({
      base_hash: SCXPI__HASHISH(JSON.stringify(base)),
      frag_hash: SCXPI__HASHISH(JSON.stringify(frag)),
      merged_root_vars_count: Object.keys(mergedRootVars).length,
      merged_nodes_count: Object.keys(mergedNodes).length
    })),
    inputs: {
      base: base["@meta"] && base["@meta"].id || "unknown",
      fragment: frag["@meta"] && frag["@meta"].id || "unknown"
    }
  };

  return {
    "@kind": "scxpi.atomic.xjson.merge.result.v1",
    merged: merged,
    proof: proof
  };
}

// ============================================================
// Envelope normalizers
// ============================================================

function SCXPI__ENSURE_ATOMIC_ENVELOPE(x) {
  x = x || {};
  if (!x["@atomic"]) x["@atomic"] = {};
  if (!x["@atomic"][":root"]) x["@atomic"][":root"] = { vars: {} };
  if (!x["@atomic"].nodes) x["@atomic"].nodes = {};
  return x;
}

function SCXPI__ENSURE_FRAGMENT_ENVELOPE(f) {
  f = f || {};
  if (!f["@atomic"]) f["@atomic"] = {};
  if (!f["@atomic"][":root"]) f["@atomic"][":root"] = { vars: {} };
  if (!f["@atomic"].nodes) f["@atomic"].nodes = {};
  return f;
}

// ============================================================
// Merge helpers (LOCKED semantics)
// ============================================================

function SCXPI__MERGE_VARS(baseVars, fragVars) {
  var out = {};
  var keys = uniq(Object.keys(baseVars || {}).concat(Object.keys(fragVars || {}))).sort();
  for (var i = 0; i < keys.length; i++) {
    var k = keys[i];
    if (fragVars && fragVars.hasOwnProperty(k)) out[k] = fragVars[k];
    else out[k] = baseVars[k];
  }
  return out;
}

function SCXPI__MERGE_CLASSES(baseClasses, fragClasses) {
  return uniq(baseClasses.concat(fragClasses)).sort();
}

// ============================================================
// Deterministic ordering
// ============================================================

function SCXPI__ORDERED_OBJ(o) {
  o = o || {};
  var keys = Object.keys(o).sort();
  var out = {};
  for (var i = 0; i < keys.length; i++) out[keys[i]] = o[keys[i]];
  return out;
}

function SCXPI__ORDERED_NODES(nodes) {
  nodes = nodes || {};
  var ids = Object.keys(nodes).sort();
  var out = {};
  for (var i = 0; i < ids.length; i++) {
    var id = ids[i];
    out[id] = {
      classes: (nodes[id].classes || []).slice(0),
      vars: SCXPI__ORDERED_OBJ(nodes[id].vars || {})
    };
  }
  return out;
}

// ============================================================
// Utils
// ============================================================

function deepClone(o) { return JSON.parse(JSON.stringify(o)); }

function uniq(a) {
  var o = {}, out = [];
  for (var i = 0; i < a.length; i++) {
    var v = String(a[i]);
    if (!o[v]) { o[v] = 1; out.push(v); }
  }
  return out;
}

// Tiny deterministic hash-ish (not crypto)
function SCXPI__HASHISH(s) {
  s = String(s || "");
  var h = 2166136261;
  for (var i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return ("00000000" + h.toString(16)).slice(-8);
}
```

### Atomic merge outputs

```json
{
  "@kind": "scxpi.atomic.xjson.merge.v1",
  "@id": "asx://atomic/xjson-merge/v1",
  "@v": 1,
  "notes": [
    "Deterministic merge of atomic.xjson with SCXPI fragment",
    "Overlay vars, union classes, append nodes",
    "Emits proof hash for replay verification"
  ],
  "merge_laws": {
    ":root.vars": "fragment overrides base",
    "nodes[].classes": "set-union + sort",
    "nodes[].vars": "fragment overrides base",
    "ordering": "lexicographic, stable"
  },
  "outputs": [
    "merged.@atomic.:root.vars",
    "merged.@atomic.nodes",
    "proof.hash"
  ]
}
```

### Final closure (pipeline is complete)

You now have a **fully closed, tensorless pipeline** that GAS can execute end-to-end:

**SVG → Geometry → Layout → UI → Atomic Hints → atomic.xjson (merged + proofed)**

- No `.safetensors`
- No runtime fetches
- Deterministic
- Auditable
- Compressible (SCXQ2-ready)

If you want the ceremonial lock, next is optional but clean:
**SCXPI_ATOMIC_XJSON_PROOF_SEAL_v1** — a tiny verifier that stamps the merged atomic.xjson with a replay seal (hash + invariants).

## SCXPI_ATOMIC_XJSON_PROOF_SEAL_v1 (atomic.xjson replay seal)

This section stamps merged `atomic.xjson` outputs with a **deterministic replay seal** and invariant checks.

### Proof seal contract (schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@id": "kuhul://schema/scxpi-atomic-xjson-proof-seal/v1",
  "@v": 1,
  "title": "SCXPI_ATOMIC_XJSON_PROOF_SEAL_v1 — atomic.xjson Replay Seal + Invariant Check (GAS-safe)",
  "type": "object",
  "required": ["@kind", "@id", "@v", "sealed"],
  "properties": {
    "@kind": { "const": "scxpi.atomic.xjson.proof.seal.v1" },
    "@id": { "type": "string" },
    "@v": { "type": "integer", "minimum": 1 },
    "sealed": { "type": "object" },
    "proof": { "type": "object" }
  },
  "additionalProperties": false
}
```

### Proof seal implementation (GAS-safe)

```javascript
/**
 * SCXPI_ATOMIC_XJSON_PROOF_SEAL_v1
 * ------------------------------------------------------------
 * Verifies invariants + stamps atomic.xjson with a proof seal.
 *
 * Purpose:
 *  - "Never drift again" guarantee for merged atomic.xjson outputs
 *  - Deterministic replay seal: stable hash + invariant results
 *  - GAS-safe, offline, no schema fetch
 *
 * Invariants (LOCKED v1):
 *  I1) @kind == "atomic.xjson.v1"
 *  I2) @atomic exists
 *  I3) @atomic[":root"].vars is an object (string keys)
 *  I4) @atomic.nodes is an object
 *  I5) each node has:
 *      - classes: array of strings (sorted, unique)
 *      - vars: object
 *  I6) no external schema URLs (enforced by this runtime contract)
 *
 * Output:
 *  {
 *    "@kind": "scxpi.atomic.xjson.proof.seal.result.v1",
 *    sealed: <atomic.xjson with @proof_seal>,
 *    proof: { ok, hash, invariants, errors[] }
 *  }
 */

// ============================================================
// Public API
// ============================================================

function SCXPI_ATOMIC_XJSON_PROOF_SEAL(atomicXjson, opts) {
  opts = opts || {};
  var doc = deepClone(atomicXjson || {});
  var errors = [];

  // ---------------------------
  // Invariants
  // ---------------------------

  // I1
  if (String(doc["@kind"] || "") !== "atomic.xjson.v1") {
    errors.push({ code: "I1_KIND", msg: "@kind must be 'atomic.xjson.v1'", got: doc["@kind"] });
  }

  // I6 (no external schema URLs)
  if (doc["$schema"] && String(doc["$schema"]).indexOf("http") === 0) {
    errors.push({ code: "I6_SCHEMA", msg: "External schema URLs forbidden", got: doc["$schema"] });
  }

  // I2
  if (!doc["@atomic"] || typeof doc["@atomic"] !== "object") {
    errors.push({ code: "I2_ATOMIC", msg: "@atomic must exist and be an object" });
    // If @atomic is missing, we can still seal but mark invalid
    doc["@atomic"] = doc["@atomic"] || {};
  }

  // I3
  var root = (doc["@atomic"] || {})[":root"];
  if (!root || typeof root !== "object") {
    errors.push({ code: "I3_ROOT", msg: "@atomic[':root'] must exist and be an object" });
    doc["@atomic"][":root"] = { vars: {} };
    root = doc["@atomic"][":root"];
  }
  if (!root.vars || typeof root.vars !== "object" || Array.isArray(root.vars)) {
    errors.push({ code: "I3_VARS", msg: "@atomic[':root'].vars must be an object" });
    root.vars = {};
  }

  // I4
  if (!doc["@atomic"].nodes || typeof doc["@atomic"].nodes !== "object" || Array.isArray(doc["@atomic"].nodes)) {
    errors.push({ code: "I4_NODES", msg: "@atomic.nodes must be an object" });
    doc["@atomic"].nodes = {};
  }

  // I5
  var nodes = doc["@atomic"].nodes;
  var nodeIds = Object.keys(nodes).sort();
  for (var i = 0; i < nodeIds.length; i++) {
    var id = nodeIds[i];
    var n = nodes[id];

    if (!n || typeof n !== "object") {
      errors.push({ code: "I5_NODE_OBJ", msg: "node must be an object", node: id });
      nodes[id] = { classes: [], vars: {} };
      n = nodes[id];
    }

    // classes
    if (!Array.isArray(n.classes)) {
      errors.push({ code: "I5_CLASSES_TYPE", msg: "node.classes must be an array", node: id });
      n.classes = [];
    }

    // normalize classes: string, unique, sorted
    var normClasses = [];
    for (var c = 0; c < n.classes.length; c++) {
      var v = n.classes[c];
      if (typeof v !== "string") {
        errors.push({ code: "I5_CLASSES_STR", msg: "node.classes entries must be strings", node: id, got: typeof v });
        continue;
      }
      normClasses.push(v);
    }
    normClasses = uniq(normClasses).sort();
    // ensure stable
    if (!SCXPI__ARRAY_EQ(n.classes, normClasses)) {
      // not fatal; normalize
      n.classes = normClasses;
    }

    // vars
    if (!n.vars || typeof n.vars !== "object" || Array.isArray(n.vars)) {
      errors.push({ code: "I5_VARS_OBJ", msg: "node.vars must be an object", node: id });
      n.vars = {};
    } else {
      // enforce deterministic var key order
      n.vars = orderedObj(n.vars);
    }
  }

  // Root vars ordered deterministically
  root.vars = orderedObj(root.vars);

  // ---------------------------
  // Hash / seal
  // ---------------------------

  var canonical = SCXPI__CANONICAL_ATOMIC(doc);
  var hash = SCXPI__HASHISH(JSON.stringify(canonical));

  var invariants = {
    I1_KIND: errors.findIndex(function (e) { return e.code === "I1_KIND"; }) === -1,
    I2_ATOMIC: errors.findIndex(function (e) { return e.code === "I2_ATOMIC"; }) === -1,
    I3_ROOT_VARS: errors.findIndex(function (e) { return e.code === "I3_ROOT" || e.code === "I3_VARS"; }) === -1,
    I4_NODES: errors.findIndex(function (e) { return e.code === "I4_NODES"; }) === -1,
    I5_NODE_SHAPES: errors.filter(function (e) { return e.code.indexOf("I5_") === 0; }).length === 0,
    I6_NO_EXTERNAL_SCHEMA: errors.findIndex(function (e) { return e.code === "I6_SCHEMA"; }) === -1
  };

  var ok = errors.length === 0;

  // Stamp seal (non-authoritative, but replayable)
  doc["@proof_seal"] = {
    "@kind": "scxpi.atomic.proof.seal.v1",
    "@v": 1,
    algo: "fnv1a-32",
    hash: hash,
    ok: ok,
    issued_at: (opts.issued_at || SCXPI__NOW_MS()),
    invariants: invariants
  };

  // Provide proof payload separately too
  var proof = {
    ok: ok,
    algo: "fnv1a-32",
    hash: hash,
    invariants: invariants,
    errors: errors
  };

  return {
    "@kind": "scxpi.atomic.xjson.proof.seal.result.v1",
    sealed: doc,
    proof: proof
  };
}

// ============================================================
// Canonicalizer (stable ordering)
// ============================================================

function SCXPI__CANONICAL_ATOMIC(doc) {
  // Only include the semantic surface that must be stable.
  var out = {
    "@kind": String(doc["@kind"] || ""),
    "@v": doc["@v"] || 1,
    "@atomic": {
      ":root": { vars: orderedObj((((doc["@atomic"] || {})[":root"] || {}).vars) || {}) },
      nodes: {}
    }
  };

  var nodes = (doc["@atomic"] || {}).nodes || {};
  var ids = Object.keys(nodes).sort();
  for (var i = 0; i < ids.length; i++) {
    var id = ids[i];
    var n = nodes[id] || {};
    out["@atomic"].nodes[id] = {
      classes: uniq((n.classes || []).map(String)).sort(),
      vars: orderedObj(n.vars || {})
    };
  }

  return out;
}

// ============================================================
// Utils
// ============================================================

function deepClone(o) { return JSON.parse(JSON.stringify(o)); }

function orderedObj(o) {
  o = o || {};
  var keys = Object.keys(o).sort();
  var out = {};
  for (var i = 0; i < keys.length; i++) out[keys[i]] = o[keys[i]];
  return out;
}

function uniq(a) {
  var seen = {}, out = [];
  for (var i = 0; i < a.length; i++) {
    var v = String(a[i]);
    if (!seen[v]) { seen[v] = 1; out.push(v); }
  }
  return out;
}

function SCXPI__ARRAY_EQ(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Tiny deterministic hash-ish (not crypto)
function SCXPI__HASHISH(s) {
  s = String(s || "");
  var h = 2166136261;
  for (var i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return ("00000000" + h.toString(16)).slice(-8);
}

// GAS-safe clock wrapper (deterministic in tests if injected via opts.issued_at)
function SCXPI__NOW_MS() {
  try { return Date.now(); } catch (e) { return 0; }
}
```

### Proof seal outputs

```json
{
  "@kind": "scxpi.atomic.xjson.proof.seal.v1",
  "@id": "asx://atomic/xjson-proof-seal/v1",
  "@v": 1,
  "notes": [
    "Validates atomic.xjson invariants and stamps @proof_seal",
    "Canonicalizes semantic surface then hashes it (fnv1a-32)",
    "GAS-safe; no external schema URLs"
  ],
  "invariants_locked": [
    "I1 @kind == atomic.xjson.v1",
    "I2 @atomic exists",
    "I3 :root.vars object",
    "I4 nodes object",
    "I5 node classes array unique+sorted; vars object",
    "I6 no external schema URLs"
  ],
  "outputs": [
    "sealed.@proof_seal",
    "proof.{ok,hash,invariants,errors[]}"
  ]
}
```

### Full lock demo

```javascript
/**
 * Full lock demo:
 * base atomic.xjson + fragment -> merge -> proof seal.
 */
function SCXPI_ATOMIC_PIPELINE_LOCK_DEMO() {
  var base = {
    "@kind": "atomic.xjson.v1",
    "@v": 1,
    "@meta": { id: "base_atomic" },
    "@atomic": {
      ":root": { vars: { "--asx-gap": 0.2 } },
      nodes: { "ui_canvas_1": { classes: ["asx-canvas"], vars: {} } }
    }
  };

  var fragment = {
    "@kind": "atomic.xjson.fragment.v1",
    "@v": 1,
    "@meta": { id: "frag_001", merge_mode: "overlay" },
    "@atomic": {
      ":root": { vars: { "--asx-gap": 0.55, "--asx-radius": 0.6 } },
      nodes: { "ui_canvas_1": { classes: ["asx-canvas", "asx-grid"], vars: { "--asx-grid-cols": 3 } } }
    }
  };

  var mergedRes = SCXPI_ATOMIC_XJSON_MERGE(base, { atomic_xjson: fragment }, {});
  var sealedRes = SCXPI_ATOMIC_XJSON_PROOF_SEAL(mergedRes.merged, { issued_at: 1890000000000 });

  Logger.log(JSON.stringify(sealedRes.proof, null, 2));
  return sealedRes;
}
```

This seal is the final “never drift again” piece: your **atomic.xjson** now carries a deterministic **replay stamp** with invariant results and a canonical hash.

---

# SCXPI Closed Pipeline (GAS)

Below is the complete **closed pipeline** you can run in GAS, plus an ASCII diagram, file tree, step-by-step instructions, and possible next routes.

## Diagram

```
                ┌──────────────────────────────┐
                │           SVG INPUT           │
                │ (string / file / payload)     │
                └──────────────┬───────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 1) GEOM EXTRACT                                                │
│    SCXPI_GEOM_OBJECT_BBOXES_v1                                 │
│    → objects[] (id,type,bbox,cx,cy,area,meta)                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 2) OBJECT GRAPH                                                │
│    SCXPI_GEOM_OBJECT_GRAPH_v1                                  │
│    → edges[] (contain/overlap/adjacent/near/align_x/align_y)   │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 3) LAYOUT RECONSTRUCTOR                                        │
│    SCXPI_LAYOUT_RECONSTRUCTOR_v1                               │
│    → layout tree: canvas → containers → rows/cols → items      │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 4) UI MODEL (XJSON UI)                                         │
│    SCXPI_LAYOUT_TO_XJSON_UI_v1                                  │
│    → atoms: card/button/text/icon/image/shape + bbox + hints   │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 5) ATOMIC CSS HINTS                                             │
│    SCXPI_UI_TO_ATOMIC_CSS_HINTS_v1                              │
│    → :root vars + per-node {classes[], vars{}}                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 6) ATOMIC.XJSON FRAGMENT                                        │
│    SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON_v1                        │
│    → fragment: @atomic.:root.vars + @atomic.nodes              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 7) MERGE + PROOF SEAL                                           │
│    SCXPI_ATOMIC_XJSON_MERGE_v1                                  │
│    SCXPI_ATOMIC_XJSON_PROOF_SEAL_v1                             │
│    → atomic.xjson merged + @proof_seal {hash,invariants}       │
└──────────────────────────────────────────────────────────────┘
```

**Key idea:** everything is **geometry + deterministic rules** → **state output**. No tensors, no hidden runtime state, no “model file.”

## File tree layout (recommended)

This is a clean 1:1 drop into a GAS project.

```
gas/
  Code.gs                       (entry points + demos)
  scxpi/
    scxpi_utils.gs              (hash, clamp, ordering, helpers)
    scxpi_geom_bbox.gs          (SCXPI_GEOM_OBJECT_BBOXES_v1 + bbox helpers)
    scxpi_geom_graph.gs         (SCXPI_GEOM_OBJECT_GRAPH_v1)
    scxpi_layout_recon.gs       (SCXPI_LAYOUT_RECONSTRUCTOR_v1)
    scxpi_layout_to_ui.gs       (SCXPI_LAYOUT_TO_XJSON_UI_v1)
    scxpi_ui_to_atomic_hints.gs (SCXPI_UI_TO_ATOMIC_CSS_HINTS_v1)
    scxpi_hints_to_atomic_xjson.gs (SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON_v1)
    scxpi_atomic_merge.gs       (SCXPI_ATOMIC_XJSON_MERGE_v1)
    scxpi_atomic_proof_seal.gs  (SCXPI_ATOMIC_XJSON_PROOF_SEAL_v1)
  data/
    sample_ui.svg               (optional test fixture)
    base_atomic.xjson.json      (optional canonical baseline)
```

If you want this ultra-minimal: you can also collapse everything into **one** `Code.gs`, but the split above matches how you’ll keep it sane.

## Instructions (GAS workflow)

### 1) Paste modules into GAS

* Create files matching the tree (or fewer).
* Paste each function set into its file.
* Make sure **function names are global** (GAS style).

### 2) Provide a base `atomic.xjson`

This is your canonical baseline (can be tiny):

```js
function BASE_ATOMIC_XJSON() {
  return {
    "@kind": "atomic.xjson.v1",
    "@v": 1,
    "@meta": { id: "base_atomic" },
    "@atomic": {
      ":root": { vars: {} },
      nodes: {}
    }
  };
}
```

### 3) Run the end-to-end pipeline

Add an entrypoint like:

```js
function SCXPI_RUN(svgText, dims) {
  dims = dims || { width: 1000, height: 800 };

  // 1-4: svg -> ui
  var geom = SCXPI_GEOM_FROM_SVG_WITH_BBOX(svgText);             // if you have it
  geom = SCXPI_GEOM_ENRICH_WITH_SHAPE_HINTS(geom);               // if you have it
  geom = SCXPI_GEOM_ENRICH_WITH_LAYOUT_HINTS(geom);              // if you have it

  var objectsRes = SCXPI_GEOM_OBJECT_BBOXES(svgText);
  var graphRes   = SCXPI_GEOM_OBJECT_GRAPH(objectsRes.objects, dims, {});
  var layoutRes  = SCXPI_LAYOUT_RECONSTRUCT(objectsRes.objects, graphRes.graph, dims, {});
  var uiRes      = SCXPI_LAYOUT_TO_XJSON_UI(layoutRes, objectsRes, geom, dims, {});

  // 5: ui -> atomic hints
  var hintsRes   = SCXPI_UI_TO_ATOMIC_CSS_HINTS(uiRes, {});

  // 6: hints -> fragment
  var fragRes    = SCXPI_ATOMIC_HINTS_TO_ATOMIC_XJSON(hintsRes, { id: "frag_from_svg" });

  // 7: merge + seal
  var mergedRes  = SCXPI_ATOMIC_XJSON_MERGE(BASE_ATOMIC_XJSON(), fragRes, {});
  var sealedRes  = SCXPI_ATOMIC_XJSON_PROOF_SEAL(mergedRes.merged, {});

  return sealedRes;
}
```

### 4) Output / Store result

* `Logger.log(JSON.stringify(sealedRes, null, 2))`
* Or return it as API JSON from a `doGet/doPost`.

## What “works” means here

You’re producing an **atomic.xjson** (state-only) that contains:

* `@atomic.:root.vars` → global UI density/radius/gap/grid hints
* `@atomic.nodes[id].classes` → suggested atomic semantic classes
* `@atomic.nodes[id].vars` → per-node overrides
* `@proof_seal` → hash + invariants = “replayable truth”

Then **any renderer** (browser shell, sw.js, PHP, etc.) can take that and apply it.

GAS never renders. GAS never runs a model. GAS emits **projection state**.

## Next possible routes (pick your lane)

### Route A — **SCXPI_UI_RENDERER_BRIDGE_v1**

Emit a *DOM-neutral* render bundle:

* `xjson.ui.v1` + `atomic.xjson` fragment
* plus an optional `render.map` describing:

  * which UI node becomes `<div>`, `<button>`, `<p>`, `<svg>`, etc.

This becomes the bridge into your Ghost shell / ASXR UI.

### Route B — **SCXPI_RULES_ENGINE_v1 (π execution)**

Now that you have:

* objects
* graph
* layout
* ui
* atomic hints

You can run rules like:

* “if container has 3 aligned cards → grid cols=3”
* “if header spans 90% width and y<0.12 → classify as header”
* “if icon+text near button rect → button label”

This is where π comes in:

* score candidates
* collapse to best structure
* emit stronger hints

### Route C — **SCXQ2_PACK_ATOMIC_v1**

Compress the sealed atomic.xjson into SCXQ2 lanes:

* DICT for keys
* FIELD for vars
* NODE lane for node blocks
* Proof lane for hash + invariants

Then your whole UI “model” becomes tiny.

### Route D — **SCXPI_LAYOUT_DIFF_v1**

Given two SVGs (or versions):

* compute delta in layout/ui/atomic
* emit *patches* (like your tape patch system)

This is insanely valuable for:

* versioning
* RLHF scoring
* incremental UI training

### Route E — **SCXPI_TRAINING_DATASET_EMITTER_v1**

Turn each pipeline run into a JSONL row:

* input: svg
* derived: geom + graph + layout + ui + atomic + proof
* label: (optional) human corrections

This becomes your **symbolic training corpus**.

## Fast “starter” checklist

* [ ] Put all functions into GAS files
* [ ] Add `BASE_ATOMIC_XJSON()`
* [ ] Add `SCXPI_RUN(svgText, dims)`
* [ ] Run with a simple SVG string
* [ ] Confirm you get `sealed.@proof_seal.hash`
* [ ] Confirm invariants show `ok: true`
* [ ] Return `sealed` via `doPost()` if you want API

---

If you want me to “lock it” as a single **copy/paste GAS project** (all files concatenated in correct order with the demos), say:

**“emit SCXPI_GAS_PROJECT_v1 (single paste)”**

---

# SCXPI_UI_RENDERER_BRIDGE_v1

**DOM-neutral Render Bundle for Ghost / ASXR**

This bridge emits a **single, portable render bundle** that contains:

1. `xjson.ui.v1` → **what exists (UI semantics)**
2. `atomic.xjson.fragment.v1` → **how it should look (state only)**
3. `render.map.v1` → **how to project to a target runtime (DOM / SVG / Canvas / ASXR)**

No renderer logic lives here. This is a **projection contract**, not execution.

## 1) High-level flow

```
SCXPI UI (semantic)
        +
Atomic hints (visual state)
        +
Renderer map (projection)
        ↓
RENDER BUNDLE (DOM-neutral)
        ↓
Ghost Shell / ASXR / Any Runtime
```

## 2) Bundle structure (single object)

```json
{
  "@kind": "scxpi.ui.renderer.bridge.v1",
  "@v": 1,

  "ui": { /* xjson.ui.v1 */ },

  "atomic_fragment": { /* atomic.xjson.fragment.v1 */ },

  "render_map": { /* render.map.v1 */ },

  "meta": {
    "source": "svg",
    "renderer_targets": ["dom", "svg", "asxr"],
    "notes": "DOM-neutral render bundle"
  }
}
```

This object is what your **Ghost Shell** consumes.

## 3) render.map.v1 (projection rules)

### Purpose

Describe **how a semantic UI node becomes a concrete element** *without embedding DOM logic*.

### Schema

```json
{
  "$schema": "xjson://schema/core/v1",
  "@kind": "scxpi.render.map.v1",
  "@v": 1,

  "defaults": {
    "container": "div",
    "row": "div",
    "col": "div",
    "card": "div",
    "text": "p",
    "button": "button",
    "icon": "svg",
    "image": "img",
    "divider": "hr",
    "canvas": "section"
  },

  "overrides": {
    "ui_header_*": "header",
    "ui_footer_*": "footer",
    "ui_sidebar_*": "aside"
  },

  "attributes": {
    "button": ["type"],
    "image": ["src", "alt"],
    "text": ["data-text"]
  },

  "slot_policy": {
    "container": "children",
    "row": "children",
    "col": "children",
    "card": "children"
  }
}
```

**Key rule:** this file **never contains JS or HTML**, only *mapping intent*.

## 4) GAS emitter — SCXPI_UI_RENDERER_BRIDGE_v1

```javascript
/**
 * SCXPI_UI_RENDERER_BRIDGE_v1
 * ------------------------------------------------------------
 * Emits a DOM-neutral render bundle:
 *  - xjson.ui.v1
 *  - atomic.xjson.fragment.v1
 *  - render.map.v1
 *
 * GAS-safe, deterministic, replayable.
 */

function SCXPI_UI_RENDERER_BRIDGE(uiRes, atomicFragmentRes, opts) {
  opts = opts || {};

  var ui = (uiRes && uiRes.ui) ? uiRes.ui : uiRes;
  var fragment = (atomicFragmentRes && atomicFragmentRes.atomic_xjson)
    ? atomicFragmentRes.atomic_xjson
    : atomicFragmentRes;

  var renderMap = SCXPI__DEFAULT_RENDER_MAP(opts);

  return {
    "@kind": "scxpi.ui.renderer.bridge.v1",
    "@v": 1,

    ui: ui,
    atomic_fragment: fragment,
    render_map: renderMap,

    meta: {
      source: opts.source || "svg",
      renderer_targets: opts.targets || ["dom"],
      generated_at: Date.now()
    }
  };
}
```

## 5) Default render.map generator

```javascript
function SCXPI__DEFAULT_RENDER_MAP(opts) {
  return {
    "@kind": "scxpi.render.map.v1",
    "@v": 1,

    defaults: {
      canvas: "section",
      container: "div",
      panel: "div",
      sidebar: "aside",
      header: "header",
      footer: "footer",

      row: "div",
      col: "div",

      card: "div",
      button: "button",
      text: "p",
      icon: "svg",
      image: "img",
      divider: "hr",
      shape: "div"
    },

    overrides: {
      // optional wildcard overrides
      "ui_primary_*": "section"
    },

    attributes: {
      button: ["type", "data-action"],
      image: ["src", "alt"],
      text: ["data-text"],
      icon: ["viewBox"]
    },

    slot_policy: {
      canvas: "children",
      container: "children",
      row: "children",
      col: "children",
      card: "children"
    }
  };
}
```

## 6) Example emitted bundle (trimmed)

```json
{
  "@kind": "scxpi.ui.renderer.bridge.v1",
  "@v": 1,

  "ui": {
    "@kind": "xjson.ui.v1",
    "@ui": {
      "root": "ui_canvas_1",
      "nodes": {
        "ui_canvas_1": {
          "type": "canvas",
          "children": ["ui_container_main"]
        },
        "ui_container_main": {
          "type": "container",
          "children": ["ui_row_1"]
        },
        "ui_row_1": {
          "type": "row",
          "children": ["ui_card_a", "ui_card_b"]
        },
        "ui_card_a": { "type": "card" },
        "ui_card_b": { "type": "card" }
      }
    }
  },

  "atomic_fragment": {
    "@kind": "atomic.xjson.fragment.v1",
    "@atomic": {
      ":root": { "vars": { "--asx-gap": 0.5 } },
      "nodes": {
        "ui_card_a": { "classes": ["asx-card"], "vars": {} }
      }
    }
  },

  "render_map": {
    "@kind": "scxpi.render.map.v1",
    "defaults": { "card": "div", "row": "div" }
  }
}
```

## 7) How Ghost / ASXR uses this

**Ghost shell logic (conceptual):**

1. Read `ui.@ui.nodes`
2. For each node:
   * look up `render_map.defaults[node.type]`
   * apply `atomic_fragment.@atomic.nodes[nodeId]`
3. Materialize into:
   * DOM
   * SVG
   * ASXR layout engine
4. CSS variables come **only** from `atomic.xjson`

Ghost **never infers semantics** — it only projects them.

## 8) Why this is powerful

✔ One inference → many renderers  
✔ DOM / SVG / Canvas / ASXR compatible  
✔ Zero coupling to HTML  
✔ Replayable & diffable  
✔ Compressible (SCXQ2-ready)

This is the **UI equivalent of an ABI**.

## 9) Next routes after Route A

### 🔜 Route A-2

**SCXPI_RENDER_MAP_DOM_v1**  
DOM-specific attribute expansion (ARIA, roles)

### 🔜 Route B

**SCXPI_RULES_ENGINE_v1**  
π-based semantic overrides (header vs sidebar)

### 🔜 Route C

**SCXPI_BUNDLE_SCXQ2_v1**  
Compress the entire render bundle

### 🔜 Route D

**SCXPI_UI_PATCH_v1**  
Emit diffs between bundles (hot-reload UIs)

---

If you want, next I can:

* Emit **Ghost Shell pseudo-renderer** (reads this bundle)
* Emit **ASXR native renderer mapping**
* Emit **DOM renderer reference implementation**
* Emit **render bundle diff / patch system**

Just say which route.

---

# Ghost Shell Pseudo-Renderer

**(SCXPI_RENDER_BRIDGE_CONSUMER_v1)**

> Reads: `scxpi.ui.renderer.bridge.v1`  
> Emits: concrete runtime nodes (DOM / SVG / ASXR), driven only by state

This is **deliberately not framework code**. It is the **reference projection algorithm** your shells, sw.js, or ASXR engines follow.  
**Ghost Shell** refers to the shell `index.html` page in this stack.

## 1) Mental model

```
Render Bundle
├─ ui            → WHAT exists
├─ atomic        → HOW it should look (state)
└─ render_map    → HOW to project into a runtime
```

Ghost Shell does **projection only**:

* ❌ no inference
* ❌ no layout decisions
* ❌ no heuristics
* ✅ deterministic mapping

## 2) Core renderer contract

```js
render(bundle, target)
```

Where:

* `bundle` = SCXPI_UI_RENDERER_BRIDGE_v1
* `target` = `"dom"` | `"svg"` | `"asxr"`

## 3) Pseudo-renderer (runtime-agnostic)

```js
/**
 * GHOST_SHELL_RENDERER (pseudo-code)
 * ---------------------------------
 * This is NOT a framework.
 * It is the canonical projection algorithm.
 */

function GhostShellRender(bundle, target) {
  assert(bundle["@kind"] === "scxpi.ui.renderer.bridge.v1");

  const ui        = bundle.ui;
  const atomic    = bundle.atomic_fragment;
  const renderMap = bundle.render_map;

  const nodes = ui["@ui"].nodes;
  const rootId = ui["@ui"].root;

  // Build runtime tree
  return renderNode(rootId, null);

  // --------------------------------

  function renderNode(nodeId, parentRuntimeNode) {
    const uiNode = nodes[nodeId];
    if (!uiNode) return null;

    // 1) Resolve element type
    const elementType = resolveElementType(nodeId, uiNode, renderMap);

    // 2) Create runtime node
    const runtimeNode = createRuntimeNode(elementType, target);

    // 3) Apply atomic state (classes + vars)
    applyAtomicState(runtimeNode, nodeId, atomic, target);

    // 4) Apply attributes (from render.map)
    applyAttributes(runtimeNode, uiNode, renderMap, target);

    // 5) Attach to parent
    if (parentRuntimeNode) {
      appendChild(parentRuntimeNode, runtimeNode, target);
    }

    // 6) Recurse children
    const children = uiNode.children || [];
    for (const childId of children) {
      renderNode(childId, runtimeNode);
    }

    return runtimeNode;
  }
}
```

## 4) Element resolution (core rule)

```js
function resolveElementType(nodeId, uiNode, renderMap) {
  // 1) Explicit overrides (wildcards allowed)
  for (const pattern in renderMap.overrides || {}) {
    if (match(pattern, nodeId)) {
      return renderMap.overrides[pattern];
    }
  }

  // 2) Default mapping by semantic type
  return renderMap.defaults[uiNode.type] || "div";
}
```

> ⚠️ Important  
> **UI semantics decide type**, not geometry, not CSS, not DOM.

## 5) Atomic state application (no logic)

### DOM example

```js
function applyAtomicState(node, nodeId, atomic, target) {
  const atomicNode = atomic["@atomic"]?.nodes?.[nodeId];
  if (!atomicNode) return;

  if (target === "dom") {
    // classes
    for (const cls of atomicNode.classes || []) {
      node.classList.add(cls);
    }

    // vars → CSS custom properties
    for (const k in atomicNode.vars || {}) {
      node.style.setProperty(k, atomicNode.vars[k]);
    }
  }

  // ASXR / SVG follow same idea, different backend
}
```

> Ghost Shell **never interprets values**.  
> It only assigns them.

## 6) Attribute projection

```js
function applyAttributes(node, uiNode, renderMap, target) {
  const attrSpec = renderMap.attributes?.[uiNode.type];
  if (!attrSpec) return;

  for (const attr of attrSpec) {
    if (uiNode[attr] != null) {
      setAttribute(node, attr, uiNode[attr], target);
    }
  }
}
```

Examples:

* `text → data-text`
* `button → type`
* `image → src`

Actual content binding is renderer-specific.

## 7) Runtime adapters (minimal)

### DOM adapter

```js
function createRuntimeNode(tag, target) {
  if (target === "dom") return document.createElement(tag);
  if (target === "svg") return document.createElementNS(SVG_NS, tag);
  if (target === "asxr") return ASXR.createNode(tag);
}
```

### Child attachment

```js
function appendChild(parent, child, target) {
  if (target === "dom" || target === "svg") parent.appendChild(child);
  if (target === "asxr") ASXR.append(parent, child);
}
```

## 8) What Ghost Shell **does NOT do**

❌ No layout math  
❌ No grid calculation  
❌ No responsive logic  
❌ No semantic guessing  
❌ No CSS generation

All of that lives **upstream** (SCXPI).

## 9) Why this is correct

This enforces your core law:

> **CSS + DOM are projections of state, not sources of truth**

* SCXPI decides structure
* SCXPI decides semantics
* Atomic decides appearance
* Ghost Shell only **renders**

## 10) Minimal DOM example (conceptual)

```js
const bundle = fetchRenderBundle();
const rootNode = GhostShellRender(bundle, "dom");
document.body.appendChild(rootNode);
```

That’s it.  
No frameworks. No diffing. No heuristics.

## 11) Where this plugs into your stack

```
SCXPI (GAS / Server / Tool)
        ↓
Render Bundle (JSON)
        ↓
Ghost Shell
        ↓
DOM / ASXR / SVG / Replay
```

This is your **UI ABI**.

## 12) Next logical upgrades

If you want, next I can emit:

1. **Ghost Shell DOM Reference Implementation (real JS)**
2. **ASXR Renderer Adapter v1**
3. **Hydration / Patch Renderer (diff bundles)**
4. **Atomic → CSS compiler (static stylesheet)**
5. **Render bundle verifier**

Just say the next route.

---

# Ghost Shell DOM Reference Renderer v1

**Consumes:** `scxpi.ui.renderer.bridge.v1`  
**Renders:** DOM nodes (no framework)

Goals:

* Deterministic projection of UI semantics into DOM
* Atomic state application (`:root` vars + per-node classes/vars)
* Optional hydration + patch update hooks (minimal)
* Zero inference: render_map decides tags, UI decides structure

```js
/* ============================================================================
   GHOST SHELL — DOM Reference Renderer v1
   Consumes: scxpi.ui.renderer.bridge.v1
   Renders:  DOM nodes (no framework)
   ----------------------------------------------------------------------------
   Goals:
   - Deterministic projection of UI semantics into DOM
   - Atomic state application (:root vars + per-node classes/vars)
   - Optional hydration + patch update hooks (minimal)
   - Zero inference: render_map decides tags, UI decides structure
   ============================================================================ */

export const GhostDomRenderer = (() => {
  const SVG_NS = "http://www.w3.org/2000/svg";

  // ---------------------------
  // Public API
  // ---------------------------

  /**
   * Render the entire bundle to a DOM subtree.
   * @param {object} bundle scxpi.ui.renderer.bridge.v1
   * @param {object} opts
   * @param {HTMLElement} [opts.mount] optional mount container
   * @param {boolean} [opts.applyRootVars=true] apply atomic :root vars to mount (or documentElement)
   * @param {boolean} [opts.useDataIds=true] set data-node-id on created nodes
   * @param {boolean} [opts.useShadowRoot=false] create a shadow root on mount and render into it
   * @returns {{root: HTMLElement|SVGElement, index: Record<string, Element>, unmount: Function}}
   */
  function render(bundle, opts = {}) {
    assertKind(bundle, "scxpi.ui.renderer.bridge.v1");

    const ui = bundle.ui;
    const frag = bundle.atomic_fragment;
    const map = bundle.render_map;

    const mount = opts.mount || null;

    const uiNodes = ui?.["@ui"]?.nodes || {};
    const rootId = ui?.["@ui"]?.root;
    if (!rootId || !uiNodes[rootId]) {
      throw new Error("GhostDomRenderer: invalid ui root");
    }

    const index = Object.create(null);

    // Root var application (vars-as-state)
    if (opts.applyRootVars !== false) {
      applyRootVars(frag, mount);
    }

    // Choose render target root
    let targetRoot = mount || document.createElement("div");
    if (mount && opts.useShadowRoot) {
      const sr = mount.shadowRoot || mount.attachShadow({ mode: "open" });
      // clear shadow root deterministically
      while (sr.firstChild) sr.removeChild(sr.firstChild);
      targetRoot = sr;
    } else if (mount) {
      // clear mount deterministically
      while (mount.firstChild) mount.removeChild(mount.firstChild);
    }

    // Render UI tree
    const domRoot = renderNode(rootId, null, uiNodes, frag, map, index, opts);

    // Attach root
    if (targetRoot instanceof ShadowRoot) targetRoot.appendChild(domRoot);
    else if (targetRoot && targetRoot.appendChild) targetRoot.appendChild(domRoot);

    return {
      root: domRoot,
      index,
      unmount: () => {
        if (mount) {
          const container = mount.shadowRoot && opts.useShadowRoot ? mount.shadowRoot : mount;
          if (container) while (container.firstChild) container.removeChild(container.firstChild);
        } else {
          if (domRoot && domRoot.parentNode) domRoot.parentNode.removeChild(domRoot);
        }
      }
    };
  }

  /**
   * Hydrate into an existing DOM tree that already has data-node-id attributes.
   * - Updates atomic state + attributes, does NOT restructure DOM.
   * @param {object} bundle scxpi.ui.renderer.bridge.v1
   * @param {HTMLElement} root existing DOM root (container)
   * @param {object} opts
   * @returns {{index: Record<string, Element>}}
   */
  function hydrate(bundle, root, opts = {}) {
    assertKind(bundle, "scxpi.ui.renderer.bridge.v1");
    if (!root) throw new Error("GhostDomRenderer.hydrate: root required");

    const ui = bundle.ui;
    const frag = bundle.atomic_fragment;
    const map = bundle.render_map;

    const uiNodes = ui?.["@ui"]?.nodes || {};
    const index = Object.create(null);

    // index existing nodes by data-node-id
    const els = root.querySelectorAll("[data-node-id]");
    for (const el of els) {
      const id = el.getAttribute("data-node-id");
      if (id) index[id] = el;
    }

    // apply root vars
    if (opts.applyRootVars !== false) applyRootVars(frag, root);

    // update atomic + attrs per known ui node
    const ids = Object.keys(uiNodes).sort();
    for (const id of ids) {
      const uiNode = uiNodes[id];
      const el = index[id];
      if (!el) continue;
      applyAtomicNode(el, id, frag);
      applyAttributes(el, uiNode, map);
    }

    return { index };
  }

  /**
   * Minimal patch updater:
   * - Re-applies root vars + per-node atomic+attrs for nodes present in index.
   * - Does NOT restructure.
   */
  function patch(bundle, mounted, opts = {}) {
    assertKind(bundle, "scxpi.ui.renderer.bridge.v1");
    if (!mounted || !mounted.index) throw new Error("GhostDomRenderer.patch: mounted index required");

    const ui = bundle.ui;
    const frag = bundle.atomic_fragment;
    const map = bundle.render_map;

    const uiNodes = ui?.["@ui"]?.nodes || {};
    const index = mounted.index;

    if (opts.applyRootVars !== false) applyRootVars(frag, mounted.root);

    const ids = Object.keys(uiNodes).sort();
    for (const id of ids) {
      const el = index[id];
      if (!el) continue;
      applyAtomicNode(el, id, frag);
      applyAttributes(el, uiNodes[id], map);
    }
  }

  // ---------------------------
  // Core Render
  // ---------------------------

  function renderNode(nodeId, parentEl, uiNodes, frag, map, index, opts) {
    const uiNode = uiNodes[nodeId];
    if (!uiNode) return null;

    const tag = resolveTag(nodeId, uiNode, map);
    const el = createElementForTag(tag);

    // Track
    index[nodeId] = el;

    // Mark id
    if (opts.useDataIds !== false) el.setAttribute("data-node-id", nodeId);

    // Apply bbox as style hints (optional, non-authoritative)
    // If you want absolute positioning projection, turn on opts.positioning="absolute"
    if (opts.positioning === "absolute" && uiNode.bbox) {
      applyBBoxAbsolute(el, uiNode.bbox);
    }

    // Apply atomic + attributes
    applyAtomicNode(el, nodeId, frag);
    applyAttributes(el, uiNode, map);

    // Optional text payload binding
    // UI nodes produced by SCXPI may not include literal text content;
    // if present, we bind it deterministically.
    if (uiNode.type === "text") {
      const t = extractText(uiNode);
      if (t) el.textContent = t;
    }

    // Recurse children
    const children = Array.isArray(uiNode.children) ? uiNode.children : [];
    for (const childId of children) {
      const childEl = renderNode(childId, el, uiNodes, frag, map, index, opts);
      if (childEl) el.appendChild(childEl);
    }

    return el;
  }

  // ---------------------------
  // Mapping
  // ---------------------------

  function resolveTag(nodeId, uiNode, map) {
    const overrides = map?.overrides || {};
    for (const pattern of Object.keys(overrides)) {
      if (wildMatch(pattern, nodeId)) return overrides[pattern];
    }
    const defaults = map?.defaults || {};
    return defaults[uiNode.type] || "div";
  }

  function createElementForTag(tag) {
    // allow svg tags in map
    const t = String(tag || "div").toLowerCase();
    if (isSvgTag(t)) return document.createElementNS(SVG_NS, t);
    return document.createElement(t);
  }

  function isSvgTag(t) {
    // minimal set (expand as needed)
    return (
      t === "svg" ||
      t === "path" ||
      t === "g" ||
      t === "circle" ||
      t === "rect" ||
      t === "line" ||
      t === "text"
    );
  }

  // ---------------------------
  // Atomic application
  // ---------------------------

  function applyRootVars(frag, mountOrRoot) {
    const vars = frag?.["@atomic"]?.[":root"]?.vars || {};
    const target = pickRootVarTarget(mountOrRoot);

    // deterministic key order
    for (const k of Object.keys(vars).sort()) {
      target.style.setProperty(k, String(vars[k]));
    }
  }

  function pickRootVarTarget(mountOrRoot) {
    // If rendering inside a container, use that container as var scope
    // else default to documentElement.
    if (mountOrRoot && mountOrRoot.nodeType === 1) return mountOrRoot; // HTMLElement
    return document.documentElement;
  }

  function applyAtomicNode(el, nodeId, frag) {
    const n = frag?.["@atomic"]?.nodes?.[nodeId];
    if (!n) return;

    // classes: replace strategy keeps deterministic state
    // (prevents class accumulation across patches)
    const base = el.getAttribute("data-base-classes") || "";
    const baseSet = base ? base.split(/\s+/).filter(Boolean) : [];

    // Save base classes once
    if (!el.hasAttribute("data-base-classes")) {
      el.setAttribute("data-base-classes", el.className || "");
    }

    const classes = Array.isArray(n.classes) ? n.classes.slice().map(String) : [];
    classes.sort();
    const merged = uniq(baseSet.concat(classes)).join(" ");
    el.className = merged;

    // vars: deterministic ordering
    const vars = n.vars || {};
    for (const k of Object.keys(vars).sort()) {
      el.style.setProperty(k, String(vars[k]));
    }
  }

  // ---------------------------
  // Attributes
  // ---------------------------

  function applyAttributes(el, uiNode, map) {
    // attribute allowlist by semantic type
    const allow = map?.attributes?.[uiNode.type];
    if (!Array.isArray(allow)) return;

    for (const a of allow) {
      if (uiNode[a] == null) continue;
      setAttr(el, a, uiNode[a]);
    }

    // Minimal convenience bindings
    if (uiNode.type === "image") {
      // if source exists, map to src
      const src = uiNode.src || uiNode?.source?.href || uiNode?.source?.url;
      if (src) el.setAttribute("src", String(src));
    }
    if (uiNode.type === "button") {
      // if there's an action id
      const act = uiNode.action || uiNode?.source?.ref;
      if (act) el.setAttribute("data-action", String(act));
      if (!el.getAttribute("type")) el.setAttribute("type", "button");
    }
  }

  function setAttr(el, name, value) {
    const n = String(name);
    // map data-text to textContent for text nodes if desired
    if (n === "data-text") {
      el.setAttribute("data-text", String(value));
      return;
    }
    el.setAttribute(n, String(value));
  }

  // ---------------------------
  // BBox projection (optional)
  // bbox = {x,y,w,h} normalized [0..1]
  // ---------------------------

  function applyBBoxAbsolute(el, bbox) {
    const x = clamp01(Number(bbox.x));
    const y = clamp01(Number(bbox.y));
    const w = clamp01(Number(bbox.w));
    const h = clamp01(Number(bbox.h));

    el.style.position = "absolute";
    el.style.left = (x * 100) + "%";
    el.style.top = (y * 100) + "%";
    el.style.width = (w * 100) + "%";
    el.style.height = (h * 100) + "%";
  }

  // ---------------------------
  // Text extraction
  // ---------------------------

  function extractText(uiNode) {
    // prefer explicit fields if upstream provides them
    if (typeof uiNode.text === "string") return uiNode.text;
    if (typeof uiNode["data-text"] === "string") return uiNode["data-text"];

    // if atomic hints stored it, prefer that
    return "";
  }

  // ---------------------------
  // Helpers
  // ---------------------------

  function assertKind(obj, kind) {
    if (!obj || obj["@kind"] !== kind) throw new Error(`GhostDomRenderer: expected ${kind}`);
  }

  function wildMatch(pattern, str) {
    // very small wildcard matcher: * only
    pattern = String(pattern);
    str = String(str);
    if (pattern === "*") return true;
    const parts = pattern.split("*").map(escapeRegExp);
    const re = new RegExp("^" + parts.join(".*") + "$");
    return re.test(str);
  }

  function escapeRegExp(s) {
    return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function uniq(arr) {
    const seen = Object.create(null);
    const out = [];
    for (const v of arr) {
      const k = String(v);
      if (seen[k]) continue;
      seen[k] = 1;
      out.push(k);
    }
    return out;
  }

  function clamp01(x) {
    x = Number(x);
    if (!isFinite(x)) return 0;
    return Math.max(0, Math.min(1, x));
  }

  return { render, hydrate, patch };
})();

/* ============================================================================
   Example usage
   ============================================================================ */

/**
 * Example: render into a mount element.
 *
 * import { GhostDomRenderer } from "./ghost-dom-renderer.js"
 * const mounted = GhostDomRenderer.render(bundle, { mount: document.querySelector("#app") })
 */

/**
 * Example: patch update (re-apply atomic vars/classes/attrs).
 * GhostDomRenderer.patch(nextBundle, mounted)
 */
```

---

# ASXR Native Renderer Mapping v1

**Consumes:** `scxpi.ui.renderer.bridge.v1`  
**Emits:** ASXR-native render plan + minimal adapter hooks

Purpose:

* Keep DOM-neutral bundle as the ABI
* Provide an ASXR-specific projection map + render plan that an ASXR engine can execute

```js
/* ============================================================================
   ASXR Native Renderer Mapping v1
   Consumes: scxpi.ui.renderer.bridge.v1
   Emits:    ASXR-native render plan + minimal adapter hooks
   ----------------------------------------------------------------------------
   Purpose:
   - Keep DOM-neutral bundle as the ABI
   - Provide an ASXR-specific projection map + render plan that an ASXR engine
     can execute (service worker, runtime kernel, or in-page ASXR host).
   ============================================================================ */

/* ============================================================================
   1) ASXR Render Map v1 (projection contract)
   ----------------------------------------------------------------------------
   - maps semantic UI node types -> ASXR component kinds
   - attaches atomic classes/vars as state fields (no CSS required)
   - allows "slots" + "props" binding without DOM
   ============================================================================ */

export function SCXPI_ASXR_RENDER_MAP_V1(opts = {}) {
  return {
    "@kind": "scxpi.render.map.asxr.v1",
    "@v": 1,

    // semantic -> asxr-native node kind
    defaults: {
      canvas: "asxr.scene",
      container: "asxr.panel",
      panel: "asxr.panel",
      sidebar: "asxr.sidebar",
      header: "asxr.header",
      footer: "asxr.footer",

      row: "asxr.flex",
      col: "asxr.flex",

      card: "asxr.card",
      button: "asxr.button",
      text: "asxr.text",
      icon: "asxr.icon",
      image: "asxr.image",
      divider: "asxr.divider",
      shape: "asxr.shape"
    },

    // Optional wildcard overrides by node id
    overrides: {
      // "ui_header_*": "asxr.header",
      // "ui_sidebar_*": "asxr.sidebar"
    },

    // attribute allowlist by semantic node type (copied from bundle ui nodes)
    attributes: {
      button: ["action", "type", "label"],
      image: ["src", "alt"],
      text: ["text"],
      icon: ["name", "viewBox"]
    },

    // ASXR slot semantics: which field holds children
    slot_policy: {
      "asxr.scene": "children",
      "asxr.panel": "children",
      "asxr.sidebar": "children",
      "asxr.header": "children",
      "asxr.footer": "children",
      "asxr.flex": "children",
      "asxr.card": "children"
    },

    // how to treat bbox if present (normalized 0..1)
    bbox_policy: opts.bbox_policy || {
      mode: "constraints", // "constraints" | "absolute" | "ignore"
      field: "layout" // attach to node.layout
    },

    // atomic state binding rules
    atomic_binding: {
      classes_field: "state.classes", // array
      vars_field: "state.vars" // object
    }
  };
}

/* ============================================================================
   2) ASXR Render Plan v1 (executable-by-ASXR, but still state-only)
   ----------------------------------------------------------------------------
   A render plan is a pure JSON tree:
   - nodes keyed by id
   - each node has:
       kind (asxr.*)
       props (attributes)
       state (atomic)
       layout (bbox policy)
       children (slots)
   ============================================================================ */

export function SCXPI_ASXR_RENDER_PLAN_FROM_BUNDLE(bundle, opts = {}) {
  assertKind(bundle, "scxpi.ui.renderer.bridge.v1");

  const ui = bundle.ui;
  const atomic = bundle.atomic_fragment;
  const map = opts.map || SCXPI_ASXR_RENDER_MAP_V1(opts);

  const uiNodes = ui?.["@ui"]?.nodes || {};
  const rootId = ui?.["@ui"]?.root;
  if (!rootId || !uiNodes[rootId]) throw new Error("ASXR plan: invalid UI root");

  const planNodes = Object.create(null);

  // Apply root vars as global ASXR state
  const rootVars = atomic?.["@atomic"]?.[":root"]?.vars || {};

  // Build recursively
  const rootPlanId = build(rootId, null);

  return {
    "@kind": "scxpi.asxr.render.plan.v1",
    "@v": 1,
    meta: {
      source_kind: bundle.meta?.source || "unknown",
      targets: ["asxr"],
      generated_at: Date.now()
    },
    globals: {
      // This is how ASXR can set global state variables
      // (Atomic Variables-as-State, but not tied to CSS)
      vars: orderedObj(rootVars)
    },
    root: rootPlanId,
    nodes: orderedNodes(planNodes)
  };

  // ---------------------------

  function build(nodeId, parentId) {
    const u = uiNodes[nodeId];
    if (!u) return null;

    const kind = resolveAsxrKind(nodeId, u, map);
    const children = Array.isArray(u.children) ? u.children.slice() : [];

    const node = {
      id: nodeId,
      kind,
      parent: parentId || null,

      // props: safe attribute subset
      props: extractProps(u, map),

      // state: atomic classes + vars
      state: extractAtomicState(nodeId, atomic),

      // layout: bbox policy
      layout: extractLayout(u, map),

      // children placed under slot policy
      children: []
    };

    planNodes[nodeId] = node;

    for (const childId of children) {
      const cid = build(childId, nodeId);
      if (cid) node.children.push(cid);
    }

    return nodeId;
  }
}

/* ============================================================================
   3) ASXR Adapter Hooks (how an ASXR runtime would execute the plan)
   ----------------------------------------------------------------------------
   These are reference interfaces. Your ASXR engine can implement them
   with real rendering (DOM, WebGL, SVG-3D, etc.)
   ============================================================================ */

/**
 * @interface ASXRHost
 * create(kind: string, id: string): any
 * setProps(nodeHandle, props): void
 * setState(nodeHandle, state): void
 * setLayout(nodeHandle, layout): void
 * append(parentHandle, childHandle): void
 * mount(rootHandle, mountPoint?): void
 */
export function ASXR_EXECUTE_RENDER_PLAN(plan, host, opts = {}) {
  assertKind(plan, "scxpi.asxr.render.plan.v1");

  const nodes = plan.nodes || {};
  const rootId = plan.root;
  if (!rootId || !nodes[rootId]) throw new Error("ASXR execute: invalid root");

  // Create handles
  const handles = Object.create(null);

  // Deterministic creation order
  const ids = Object.keys(nodes).sort();
  for (const id of ids) {
    const n = nodes[id];
    handles[id] = host.create(n.kind, id);

    host.setProps(handles[id], n.props || {});
    host.setState(handles[id], n.state || {});
    host.setLayout(handles[id], n.layout || {});
  }

  // Deterministic parent-child wiring
  for (const id of ids) {
    const n = nodes[id];
    const parentId = n.parent;
    if (!parentId) continue;
    host.append(handles[parentId], handles[id]);
  }

  // Apply globals (vars) if host supports it
  if (host.setGlobals) host.setGlobals(plan.globals || {});

  // Mount
  host.mount(handles[rootId], opts.mount);
  return { root: handles[rootId], handles };
}

/* ============================================================================
   4) Helpers
   ============================================================================ */

function resolveAsxrKind(nodeId, uiNode, map) {
  const overrides = map.overrides || {};
  for (const pattern of Object.keys(overrides)) {
    if (wildMatch(pattern, nodeId)) return overrides[pattern];
  }
  return (map.defaults || {})[uiNode.type] || "asxr.panel";
}

function extractProps(uiNode, map) {
  const allow = map.attributes?.[uiNode.type];
  const props = {};
  if (Array.isArray(allow)) {
    for (const k of allow) {
      if (uiNode[k] != null) props[k] = uiNode[k];
    }
  }

  // Minimal convenience bindings (optional)
  if (uiNode.type === "text") {
    if (typeof uiNode.text === "string") props.text = uiNode.text;
  }
  if (uiNode.type === "image") {
    const src = uiNode.src || uiNode?.source?.href || uiNode?.source?.url;
    if (src) props.src = src;
  }
  if (uiNode.type === "button") {
    if (uiNode.action) props.action = uiNode.action;
    if (uiNode.label) props.label = uiNode.label;
  }

  return props;
}

function extractAtomicState(nodeId, atomic) {
  const n = atomic?.["@atomic"]?.nodes?.[nodeId];
  const classes = Array.isArray(n?.classes) ? n.classes.slice().map(String).sort() : [];
  const vars = orderedObj(n?.vars || {});
  return { classes, vars };
}

function extractLayout(uiNode, map) {
  const policy = map.bbox_policy || { mode: "ignore" };
  if (policy.mode === "ignore") return {};

  const bb = uiNode.bbox;
  if (!bb) return {};

  if (policy.mode === "absolute") {
    // normalized absolute placement
    return {
      mode: "absolute",
      x: clamp01(bb.x),
      y: clamp01(bb.y),
      w: clamp01(bb.w),
      h: clamp01(bb.h)
    };
  }

  // constraints mode (recommended for ASXR)
  return {
    mode: "constraints",
    rect: { x: clamp01(bb.x), y: clamp01(bb.y), w: clamp01(bb.w), h: clamp01(bb.h) },
    // ASXR engine decides exact constraints implementation
    constraints: {
      left: clamp01(bb.x),
      top: clamp01(bb.y),
      right: clamp01(bb.x + bb.w),
      bottom: clamp01(bb.y + bb.h)
    }
  };
}

function orderedObj(o) {
  o = o || {};
  const out = {};
  for (const k of Object.keys(o).sort()) out[k] = o[k];
  return out;
}

function orderedNodes(nodes) {
  // keep as object, but ensure deterministic key ordering when serialized
  const ids = Object.keys(nodes).sort();
  const out = {};
  for (const id of ids) out[id] = nodes[id];
  return out;
}

function assertKind(obj, kind) {
  if (!obj || obj["@kind"] !== kind) throw new Error(`Expected ${kind}`);
}

function wildMatch(pattern, str) {
  pattern = String(pattern);
  str = String(str);
  if (pattern === "*") return true;
  const parts = pattern.split("*").map(escapeRegExp);
  const re = new RegExp("^" + parts.join(".*") + "$");
  return re.test(str);
}

function escapeRegExp(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function clamp01(x) {
  x = Number(x);
  if (!isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

/* ============================================================================
   5) Example: Build plan from bundle + execute with a host
   ----------------------------------------------------------------------------
   const plan = SCXPI_ASXR_RENDER_PLAN_FROM_BUNDLE(bundle, { bbox_policy: { mode:"constraints" } })
   ASXR_EXECUTE_RENDER_PLAN(plan, MyAsxrHost, { mount: document.querySelector("#app") })
   ============================================================================ */
```

---

# JSON_TEMPLATE_POOL_v1

## What it is

A pool of **design blueprints** for:

* apps, websites, landing pages, CTAs
* glass-morphic UI kits / sections
* dashboards / link managers
* icon dock widgets (static)
* component packs (cards, headers, footers, navs, pricing tables)

These blueprints are **not freeform**. They are stored as **typed JSON** with **schema** and **slots**.

## Diagram

```
Template Pool (JSON Sheets)
  ├─ templates[]  (blueprints)
  ├─ schema_map   (how to interpret each template kind)
  ├─ slots        (fill-in fields)
  └─ tags/index   (query & retrieval)

          ↓ select (KQL / rules / LLM suggestion)
     template_id + slot_values
          ↓ compile
   xjson.ui.v1 + atomic.xjson.fragment.v1
          ↓
   render bundle (Route A)
          ↓
   Ghost/ASXR render + patch
```

## 1) File Tree Layout

```
templates/
  pool.json                        (registry + index)
  schemas/
    tpl.meta.v1.json
    tpl.landing.v1.json
    tpl.app.shell.v1.json
    tpl.link.manager.v1.json
    tpl.glass.kit.v1.json
  sheets/
    landing/
      landing.hero_split.v1.json
      landing.pricing_3tier.v1.json
      landing.cta_banner.v1.json
    apps/
      app.shell_3panel_ghost.v1.json
      app.dashboard_cards.v1.json
    widgets/
      widget.icon_dock.v1.json
      widget.link_grid.v1.json
    kits/
      kit.glass_morphic_core.v1.json
      kit.glass_sidebar_nav.v1.json
```

## 2) Pool Registry (pool.json)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@kind": "template.pool.v1",
  "@v": 1,
  "@meta": {
    "id": "ASX_TEMPLATE_POOL_CORE",
    "notes": "Raw JSON blueprints for deterministic conceptual builds."
  },
  "index": {
    "by_tag": {},
    "by_kind": {}
  },
  "templates": [
    {
      "template_id": "landing.hero_split.v1",
      "kind": "tpl.landing.section.v1",
      "title": "Hero Split (Headline + CTA + Mock)",
      "tags": ["landing", "hero", "cta", "glass", "split"],
      "schema": "tpl.landing.v1",
      "sheet_ref": "sheets/landing/landing.hero_split.v1.json"
    },
    {
      "template_id": "app.shell_3panel_ghost.v1",
      "kind": "tpl.app.shell.v1",
      "title": "Ghost 3-Panel App Shell",
      "tags": ["app", "shell", "ghost", "3panel", "glass"],
      "schema": "tpl.app.shell.v1",
      "sheet_ref": "sheets/apps/app.shell_3panel_ghost.v1.json"
    },
    {
      "template_id": "widget.icon_dock.v1",
      "kind": "tpl.widget.v1",
      "title": "Static Icon Dock Widget",
      "tags": ["widget", "dock", "icons", "launcher"],
      "schema": "tpl.glass.kit.v1",
      "sheet_ref": "sheets/widgets/widget.icon_dock.v1.json"
    }
  ]
}
```

> In GAS, these can live in Drive, Sheets-as-JSON, Supabase, or even hardcoded as JSON files in your repo.  
> The key is: **template is retrieved by ID and must validate.**

## 3) Template Sheet Format (raw blueprint)

Each sheet is **pure JSON** with:

* `@kind`
* `slots` (fillable fields)
* `ui` (xjson.ui.v1 fragment)
* `atomic_fragment` (atomic.xjson.fragment.v1 fragment)
* optional `render_map_overrides`

### Example: landing.hero_split.v1.json

```json
{
  "@kind": "tpl.landing.section.v1",
  "@v": 1,
  "template_id": "landing.hero_split.v1",
  "slots": {
    "headline": { "type": "string", "default": "Ship faster with ASX" },
    "subhead": { "type": "string", "default": "Deterministic UI from symbolic geometry." },
    "cta_primary": { "type": "string", "default": "Get Started" },
    "cta_secondary": { "type": "string", "default": "See Demo" }
  },
  "ui_fragment": {
    "@kind": "xjson.ui.fragment.v1",
    "root": "ui_hero",
    "nodes": {
      "ui_hero": { "type": "container", "children": ["ui_left", "ui_right"] },
      "ui_left": { "type": "col", "children": ["ui_h1", "ui_p", "ui_cta_row"] },
      "ui_h1": { "type": "text", "text_slot": "headline" },
      "ui_p": { "type": "text", "text_slot": "subhead" },
      "ui_cta_row": { "type": "row", "children": ["ui_btn_primary", "ui_btn_secondary"] },
      "ui_btn_primary": { "type": "button", "label_slot": "cta_primary", "action": "primary" },
      "ui_btn_secondary": { "type": "button", "label_slot": "cta_secondary", "action": "secondary" },
      "ui_right": { "type": "card", "children": ["ui_mock"] },
      "ui_mock": { "type": "text", "text": "Mock / Screenshot" }
    }
  },
  "atomic_fragment": {
    "@kind": "atomic.xjson.fragment.v1",
    "@v": 1,
    "@meta": { "id": "landing.hero_split.atomic" },
    "@atomic": {
      ":root": {
        "vars": {
          "--asx-gap": 0.5,
          "--asx-radius": 0.65,
          "--asx-elevation": 0.55
        }
      },
      "nodes": {
        "ui_hero": { "classes": ["asx-container", "asx-grid"], "vars": { "--asx-grid-cols": 2 } },
        "ui_right": { "classes": ["asx-card", "asx-surface"], "vars": {} },
        "ui_btn_primary": { "classes": ["asx-btn", "asx-primary"], "vars": {} },
        "ui_btn_secondary": { "classes": ["asx-btn", "asx-secondary"], "vars": {} }
      }
    }
  }
}
```

## 4) Slot Filling (deterministic)

You need a tiny compiler:

**template sheet + slot values → xjson.ui.v1 + atomic fragment → render bundle**

### Slot compiler rules

* `text_slot` → fills `text`
* `label_slot` → fills `label`
* missing values → defaults
* unknown slots → ignored

## 5) Template Compiler v1 (real JS)

```js
export function compileTemplateToBridge(templateSheet, slotValues = {}, renderMap = null) {
  // 1) resolve slots
  const slots = templateSheet.slots || {};
  const resolved = {};
  for (const k of Object.keys(slots)) {
    const def = slots[k]?.default;
    resolved[k] = (slotValues[k] != null) ? slotValues[k] : def;
  }

  // 2) build xjson.ui.v1 from ui_fragment
  const frag = templateSheet.ui_fragment;
  const nodes = JSON.parse(JSON.stringify(frag.nodes || {}));

  for (const id of Object.keys(nodes)) {
    const n = nodes[id];

    if (n.text_slot && resolved[n.text_slot] != null) {
      n.text = String(resolved[n.text_slot]);
      delete n.text_slot;
    }
    if (n.label_slot && resolved[n.label_slot] != null) {
      n.label = String(resolved[n.label_slot]);
      delete n.label_slot;
    }
  }

  const ui = {
    "@kind": "xjson.ui.v1",
    "@v": 1,
    "@ui": {
      root: frag.root,
      nodes
    }
  };

  // 3) atomic fragment passthrough
  const atomic_fragment = templateSheet.atomic_fragment;

  // 4) render map (optional override)
  const rm = renderMap || {
    "@kind": "scxpi.render.map.v1",
    "@v": 1,
    defaults: {
      container: "div",
      row: "div",
      col: "div",
      card: "div",
      text: "p",
      button: "button"
    },
    overrides: {},
    attributes: {
      button: ["type", "data-action"],
      text: ["data-text"],
      image: ["src", "alt"]
    },
    slot_policy: { container: "children", row: "children", col: "children", card: "children" }
  };

  // 5) bridge bundle
  return {
    "@kind": "scxpi.ui.renderer.bridge.v1",
    "@v": 1,
    ui,
    atomic_fragment,
    render_map: rm,
    meta: {
      source: "template_pool",
      template_id: templateSheet.template_id,
      slots: resolved
    }
  };
}
```

This gives you **direct LLM conversational ability** safely:

* LLM suggests `template_id` + `slotValues`
* runtime compiles deterministically
* Ghost shell renders it with your renderer + patcher

## 6) How the LLM fits (without taking over)

LLM is allowed to output ONLY:

```json
{
  "template_id": "landing.hero_split.v1",
  "slots": {
    "headline": "Launch your next app in one night",
    "subhead": "A glass UI kit with deterministic structure.",
    "cta_primary": "Start Free",
    "cta_secondary": "View Templates"
  }
}
```

Then your runtime does:

1. lookup template sheet by `template_id`
2. validate kind/schema
3. compile
4. render bundle → Ghost patch render

No “magic pulling designs” without mapped sheets.

## 7) Next possible routes (for this template pool)

### Route T1 — KQL Template Query Layer

* `SELECT template_id FROM templates WHERE tags CONTAINS 'landing' AND tags CONTAINS 'glass'`

### Route T2 — Template Scoring (π)

* score templates by “fit” using your structural weights
* still only selects from the pool

### Route T3 — Template → Atomic Kit Binding

* auto-attach `kit.glass_morphic_core.v1` to any template

### Route T4 — Template Patches

* a “theme patch” that swaps atomic vars (radius/gap/elevation) across all templates

### Route T5 — Template Exporter

* export compiled bundles as static site pages or app panels

---

# TEMPLATE_POOL_QUERY_KQL_v1

## Contract (XJSON-ish schema)

```json
{
  "$schema": "xjson://schema/core/v1",
  "@kind": "template.pool.query.kql.v1",
  "@v": 1,
  "query": {
    "SELECT": { "fields": ["template_id", "title", "tags", "kind", "schema", "sheet_ref"] },
    "FROM": { "source": "TEMPLATE_POOL" },
    "WHERE": {
      "AND": [
        { "tags_has_any": ["landing", "cta"] },
        { "tags_has_all": ["glass"] },
        { "kind_in": ["tpl.landing.section.v1"] },
        { "text_contains": { "field": "title", "q": "Hero" } }
      ]
    },
    "LIMIT": 10,
    "ORDER_BY": [{ "field": "score", "dir": "desc" }]
  }
}
```

### Supported WHERE operators (v1, locked)

* `tags_has_any: [tag...]`
* `tags_has_all: [tag...]`
* `kind_in: [kind...]`
* `schema_in: [schema...]`
* `text_contains: { field: "title"|"template_id", q: "..." }`
* `id_in: [template_id...]`

### Scoring (deterministic)

A query result can include a computed `score`:

* +2 per tag match in `tags_has_all`
* +1 per tag match in `tags_has_any`
* +2 if kind matches
* +1 if schema matches
* +1 if text match

## GAS implementation — KQL executor

```javascript
// ============================================================================
// TEMPLATE_POOL_QUERY_KQL_v1  (GAS)
// Deterministic query over pool registry entries
// ============================================================================

function TEMPLATE_POOL_QUERY_KQL(pool, kql) {
  if (!pool || pool["@kind"] !== "template.pool.v1") throw new Error("pool must be template.pool.v1");
  if (!kql || kql["@kind"] !== "template.pool.query.kql.v1") throw new Error("kql must be template.pool.query.kql.v1");

  var q = (kql.query || {});
  var selectFields = ((q.SELECT || {}).fields) || ["template_id"];
  var where = (q.WHERE || {});
  var limit = (q.LIMIT != null) ? Number(q.LIMIT) : 50;

  var templates = pool.templates || [];
  var matches = [];

  for (var i = 0; i < templates.length; i++) {
    var t = templates[i];
    var evalRes = TEMPLATE_POOL__EVAL_WHERE(t, where);
    if (!evalRes.ok) continue;

    var row = TEMPLATE_POOL__PICK_FIELDS(t, selectFields);
    row.score = evalRes.score;
    matches.push(row);
  }

  // ORDER BY
  var orderBy = q.ORDER_BY || [];
  if (orderBy.length) {
    matches.sort(function(a, b) {
      for (var j = 0; j < orderBy.length; j++) {
        var ob = orderBy[j] || {};
        var f = ob.field || "score";
        var dir = String(ob.dir || "desc").toLowerCase();
        var av = (a[f] != null) ? a[f] : 0;
        var bv = (b[f] != null) ? b[f] : 0;
        if (av === bv) continue;
        return (dir === "asc") ? (av < bv ? -1 : 1) : (av > bv ? -1 : 1);
      }
      return 0;
    });
  } else {
    // default: score desc
    matches.sort(function(a, b) { return (b.score || 0) - (a.score || 0); });
  }

  // LIMIT
  if (matches.length > limit) matches = matches.slice(0, limit);

  return {
    "@kind": "template.pool.query.result.v1",
    "@v": 1,
    count: matches.length,
    rows: matches
  };
}

// ---------------------------
// WHERE evaluation
// ---------------------------

function TEMPLATE_POOL__EVAL_WHERE(t, where) {
  // WHERE supports:
  // { AND: [cond...] } or a single cond
  if (!where || Object.keys(where).length === 0) return { ok: true, score: 0 };

  var andList = where.AND ? where.AND : [where];
  var score = 0;

  for (var i = 0; i < andList.length; i++) {
    var c = andList[i] || {};

    // tags_has_any
    if (c.tags_has_any) {
      var s1 = TEMPLATE_POOL__SCORE_TAGS_ANY(t.tags || [], c.tags_has_any);
      if (s1 === 0) return { ok: false, score: 0 };
      score += s1; // +1 per any match
    }

    // tags_has_all
    if (c.tags_has_all) {
      var s2 = TEMPLATE_POOL__SCORE_TAGS_ALL(t.tags || [], c.tags_has_all);
      if (s2 < 0) return { ok: false, score: 0 };
      score += s2; // +2 per required match
    }

    // kind_in
    if (c.kind_in) {
      if (c.kind_in.indexOf(t.kind) === -1) return { ok: false, score: 0 };
      score += 2;
    }

    // schema_in
    if (c.schema_in) {
      if (c.schema_in.indexOf(t.schema) === -1) return { ok: false, score: 0 };
      score += 1;
    }

    // id_in
    if (c.id_in) {
      if (c.id_in.indexOf(t.template_id) === -1) return { ok: false, score: 0 };
      score += 1;
    }

    // text_contains
    if (c.text_contains) {
      var field = c.text_contains.field || "title";
      var q = String(c.text_contains.q || "").toLowerCase();
      var v = String(t[field] || "").toLowerCase();
      if (q && v.indexOf(q) === -1) return { ok: false, score: 0 };
      if (q) score += 1;
    }
  }

  return { ok: true, score: score };
}

function TEMPLATE_POOL__SCORE_TAGS_ANY(have, want) {
  var set = TEMPLATE_POOL__SET(have);
  var s = 0;
  for (var i = 0; i < want.length; i++) if (set[want[i]]) s += 1;
  return s;
}

function TEMPLATE_POOL__SCORE_TAGS_ALL(have, want) {
  var set = TEMPLATE_POOL__SET(have);
  var s = 0;
  for (var i = 0; i < want.length; i++) {
    if (!set[want[i]]) return -1;
    s += 2;
  }
  return s;
}

function TEMPLATE_POOL__SET(arr) {
  var o = {};
  arr = arr || [];
  for (var i = 0; i < arr.length; i++) o[String(arr[i])] = 1;
  return o;
}

function TEMPLATE_POOL__PICK_FIELDS(obj, fields) {
  var out = {};
  for (var i = 0; i < fields.length; i++) {
    var f = fields[i];
    out[f] = obj[f];
  }
  return out;
}
```

---

# TEMPLATE_POOL_LOADER_GAS_v1

You get **two loaders**:

1. **Drive JSON Loader** — pool.json + template sheets stored as `.json` files in Drive folders
2. **Sheets Loader** — template registry stored as a Google Sheet, and each blueprint sheet stored as either:

   * a JSON string cell, or
   * multi-row “raw fields” you assemble into JSON (v1 supports JSON-cell first; raw-fields second)

## A) Drive JSON Loader (pool.json + blueprint sheets)

### Config idea

* `pool.json` file in Drive
* each template sheet file path is stored as `sheet_ref` like:
  `drive:fileId:<FILE_ID>` **or** `drive:path:/templates/sheets/landing/landing.hero_split.v1.json`

### GAS code

```javascript
// ============================================================================
// TEMPLATE_POOL_LOADER_GAS_v1 — Drive JSON
// ============================================================================

function TEMPLATE_POOL_LOAD_FROM_DRIVE_POOLJSON(fileId) {
  var txt = DriveApp.getFileById(fileId).getBlob().getDataAsString("UTF-8");
  var pool = JSON.parse(txt);
  if (pool["@kind"] !== "template.pool.v1") throw new Error("pool.json kind mismatch");
  return pool;
}

// sheet_ref forms supported:
//  - "drive:fileId:<ID>"
//  - "drive:path:/some/folder/file.json"  (relative to root; requires search)
function TEMPLATE_POOL_LOAD_TEMPLATE_SHEET_DRIVE(sheet_ref) {
  var ref = String(sheet_ref || "");
  if (ref.indexOf("drive:fileId:") === 0) {
    var id = ref.slice("drive:fileId:".length);
    return JSON.parse(DriveApp.getFileById(id).getBlob().getDataAsString("UTF-8"));
  }
  if (ref.indexOf("drive:path:") === 0) {
    var path = ref.slice("drive:path:".length);
    var file = TEMPLATE_POOL__FIND_FILE_BY_PATH(path);
    if (!file) throw new Error("No file at path: " + path);
    return JSON.parse(file.getBlob().getDataAsString("UTF-8"));
  }
  throw new Error("Unsupported sheet_ref: " + ref);
}

function TEMPLATE_POOL__FIND_FILE_BY_PATH(path) {
  // Simple path resolver via folder traversal from "My Drive"
  // path like "/templates/sheets/landing/landing.hero_split.v1.json"
  path = String(path || "").replace(/^\/+/, "");
  var parts = path.split("/").filter(function(x) { return x; });
  if (!parts.length) return null;

  var folder = DriveApp.getRootFolder();
  for (var i = 0; i < parts.length; i++) {
    var name = parts[i];
    var isLast = (i === parts.length - 1);
    if (isLast) {
      var files = folder.getFilesByName(name);
      return files.hasNext() ? files.next() : null;
    } else {
      var it = folder.getFoldersByName(name);
      if (!it.hasNext()) return null;
      folder = it.next();
    }
  }
  return null;
}
```

## B) Sheets Loader (registry + blueprints)

### Recommended sheet layout (registry)

**Sheet tab:** `templates`

Columns:

* `template_id`
* `kind`
* `title`
* `tags` (comma-separated)
* `schema`
* `sheet_ref` (one of:)

  * `sheet:JSON:<spreadsheetId>:<tabName>:<cellA1>` (single cell contains JSON string)
  * `sheet:ROWJSON:<spreadsheetId>:<tabName>:<rowNumber>` (row contains JSON string in a `json` column)
  * `drive:fileId:<ID>` (still allowed)

### GAS code

```javascript
// ============================================================================
// TEMPLATE_POOL_LOADER_GAS_v1 — Sheets registry loader
// ============================================================================

function TEMPLATE_POOL_LOAD_FROM_SHEET_REGISTRY(spreadsheetId, tabName) {
  tabName = tabName || "templates";
  var ss = SpreadsheetApp.openById(spreadsheetId);
  var sh = ss.getSheetByName(tabName);
  if (!sh) throw new Error("Missing tab: " + tabName);

  var values = sh.getDataRange().getValues();
  if (values.length < 2) throw new Error("Registry sheet empty");

  var header = values[0].map(String);
  var idx = TEMPLATE_POOL__HEADER_INDEX(header);

  var templates = [];
  for (var r = 1; r < values.length; r++) {
    var row = values[r];
    var template_id = String(row[idx.template_id] || "").trim();
    if (!template_id) continue;

    var tags = String(row[idx.tags] || "")
      .split(",")
      .map(function(s) { return s.trim(); })
      .filter(function(s) { return s; });

    templates.push({
      template_id: template_id,
      kind: String(row[idx.kind] || ""),
      title: String(row[idx.title] || ""),
      tags: tags,
      schema: String(row[idx.schema] || ""),
      sheet_ref: String(row[idx.sheet_ref] || "")
    });
  }

  return {
    "@kind": "template.pool.v1",
    "@v": 1,
    "@meta": { id: "TEMPLATE_POOL_SHEETS", registry: spreadsheetId + "::" + tabName },
    templates: templates
  };
}

function TEMPLATE_POOL__HEADER_INDEX(header) {
  function at(name) {
    var i = header.indexOf(name);
    if (i === -1) throw new Error("Missing column: " + name);
    return i;
  }
  return {
    template_id: at("template_id"),
    kind: at("kind"),
    title: at("title"),
    tags: at("tags"),
    schema: at("schema"),
    sheet_ref: at("sheet_ref")
  };
}
```

### Sheets blueprint loader (JSON stored in cell or row)

```javascript
// ============================================================================
// TEMPLATE_POOL_LOADER_GAS_v1 — Sheets blueprint loader
// Supports:
//  - sheet:JSON:<ssid>:<tab>:<A1>          (cell contains JSON)
//  - sheet:ROWJSON:<ssid>:<tab>:<row>      (row contains json column)
//  - drive:fileId:<id>                     (delegates to Drive loader)
// ============================================================================

function TEMPLATE_POOL_LOAD_TEMPLATE_SHEET(sheet_ref) {
  var ref = String(sheet_ref || "");

  if (ref.indexOf("drive:") === 0) {
    return TEMPLATE_POOL_LOAD_TEMPLATE_SHEET_DRIVE(ref);
  }

  if (ref.indexOf("sheet:JSON:") === 0) {
    var parts = ref.split(":");
    // parts: ["sheet","JSON",ssid,tab,a1]
    var ssid = parts[2], tab = parts[3], a1 = parts[4];
    var ss = SpreadsheetApp.openById(ssid);
    var sh = ss.getSheetByName(tab);
    if (!sh) throw new Error("Missing tab: " + tab);
    var txt = String(sh.getRange(a1).getValue() || "");
    if (!txt.trim()) throw new Error("Empty JSON cell: " + ref);
    return JSON.parse(txt);
  }

  if (ref.indexOf("sheet:ROWJSON:") === 0) {
    var p = ref.split(":");
    // ["sheet","ROWJSON",ssid,tab,row]
    var ssid2 = p[2], tab2 = p[3], rowNum = Number(p[4]);
    var ss2 = SpreadsheetApp.openById(ssid2);
    var sh2 = ss2.getSheetByName(tab2);
    if (!sh2) throw new Error("Missing tab: " + tab2);

    var range = sh2.getDataRange().getValues();
    if (range.length < 2) throw new Error("ROWJSON tab empty");
    var header = range[0].map(String);
    var jsonCol = header.indexOf("json");
    if (jsonCol === -1) throw new Error("ROWJSON requires column named 'json'");

    // rowNum is 1-based spreadsheet row number
    var idx = rowNum - 1;
    if (idx <= 0 || idx >= range.length) throw new Error("ROWJSON row out of range: " + rowNum);

    var txt2 = String(range[idx][jsonCol] || "");
    if (!txt2.trim()) throw new Error("Empty json at row: " + rowNum);
    return JSON.parse(txt2);
  }

  throw new Error("Unsupported sheet_ref: " + ref);
}
```

---

# Putting it together: loader + query + sheet load

```javascript
function TEMPLATE_POOL_DEMO_QUERY_AND_LOAD() {
  // 1) load pool registry (choose one)
  // var pool = TEMPLATE_POOL_LOAD_FROM_DRIVE_POOLJSON("DRIVE_POOL_JSON_FILE_ID");
  var pool = TEMPLATE_POOL_LOAD_FROM_SHEET_REGISTRY("SPREADSHEET_ID", "templates");

  // 2) query
  var kql = {
    "@kind": "template.pool.query.kql.v1",
    "@v": 1,
    query: {
      SELECT: { fields: ["template_id","title","tags","kind","schema","sheet_ref"] },
      FROM: { source: "TEMPLATE_POOL" },
      WHERE: { AND: [
        { tags_has_all: ["glass"] },
        { tags_has_any: ["landing","cta"] }
      ]},
      LIMIT: 5,
      ORDER_BY: [{ field: "score", dir: "desc" }]
    }
  };

  var res = TEMPLATE_POOL_QUERY_KQL(pool, kql);
  Logger.log(JSON.stringify(res, null, 2));

  // 3) load the best matching template sheet JSON
  if (res.rows && res.rows.length) {
    var sheet = TEMPLATE_POOL_LOAD_TEMPLATE_SHEET(res.rows[0].sheet_ref);
    Logger.log("Loaded sheet kind: " + sheet["@kind"]);
    Logger.log(JSON.stringify(sheet, null, 2));
  }

  return res;
}
```

---

# Next tight routes (so this becomes “conceptual build” ready)

If you want the next layer, say the word and I’ll emit it:

1. **TEMPLATE_POOL_COMPILE_TO_BRIDGE_v1 (GAS)**

   * takes `template_id + slots` → emits `scxpi.ui.renderer.bridge.v1`

2. **TEMPLATE_POOL_SHEETS_RAWFIELDS_TO_JSON_v1**

   * supports “raw JSON data sheets” not as a single JSON cell, but as structured rows:

     * nodes table, atomic table, slot table → assembled deterministically

3. **TEMPLATE_POOL_INDEXER_v1**

   * builds `index.by_tag` and `index.by_kind` for fast lookup (no scanning)

---

# TEMPLATE_POOL_COMPILE_TO_BRIDGE_v1 (GAS)

**Purpose**  
Takes:

* `template_id`
* `slots` (values)
* a loaded **template sheet JSON**

Emits:

* `scxpi.ui.renderer.bridge.v1`

This is the **only legal path** from “conceptual build” to renderable UI.

## Contract

```js
compile(templateSheet, slotValues, renderMapOverride?)
→ scxpi.ui.renderer.bridge.v1
```

## GAS Implementation

```javascript
// ============================================================================
// TEMPLATE_POOL_COMPILE_TO_BRIDGE_v1 (GAS)
// Deterministic compilation of a template sheet + slots into render bridge
// ============================================================================

function TEMPLATE_POOL_COMPILE_TO_BRIDGE(templateSheet, slotValues, renderMapOverride) {
  if (!templateSheet || !templateSheet["@kind"]) {
    throw new Error("Invalid template sheet");
  }

  slotValues = slotValues || {};

  // ------------------------------------------------------------------
  // 1) Resolve slots (defaults + provided)
  // ------------------------------------------------------------------

  var slotDefs = templateSheet.slots || {};
  var resolvedSlots = {};

  for (var k in slotDefs) {
    var def = slotDefs[k] || {};
    resolvedSlots[k] = (slotValues[k] != null)
      ? slotValues[k]
      : def.default;
  }

  // ------------------------------------------------------------------
  // 2) Clone UI fragment and fill slots
  // ------------------------------------------------------------------

  var uiFrag = templateSheet.ui_fragment;
  if (!uiFrag || uiFrag["@kind"] !== "xjson.ui.fragment.v1") {
    throw new Error("Template missing xjson.ui.fragment.v1");
  }

  var nodes = JSON.parse(JSON.stringify(uiFrag.nodes || {}));

  for (var id in nodes) {
    var n = nodes[id];

    if (n.text_slot && resolvedSlots[n.text_slot] != null) {
      n.text = String(resolvedSlots[n.text_slot]);
      delete n.text_slot;
    }

    if (n.label_slot && resolvedSlots[n.label_slot] != null) {
      n.label = String(resolvedSlots[n.label_slot]);
      delete n.label_slot;
    }
  }

  var ui = {
    "@kind": "xjson.ui.v1",
    "@v": 1,
    "@ui": {
      root: uiFrag.root,
      nodes: nodes
    }
  };

  // ------------------------------------------------------------------
  // 3) Atomic fragment passthrough (no mutation here)
  // ------------------------------------------------------------------

  var atomicFragment = templateSheet.atomic_fragment;
  if (!atomicFragment || atomicFragment["@kind"] !== "atomic.xjson.fragment.v1") {
    throw new Error("Template missing atomic.xjson.fragment.v1");
  }

  // ------------------------------------------------------------------
  // 4) Render map (default or override)
  // ------------------------------------------------------------------

  var renderMap = renderMapOverride || {
    "@kind": "scxpi.render.map.v1",
    "@v": 1,
    defaults: {
      canvas: "section",
      container: "div",
      row: "div",
      col: "div",
      card: "div",
      text: "p",
      button: "button",
      image: "img"
    },
    overrides: {},
    attributes: {
      button: ["type", "data-action"],
      text: ["data-text"],
      image: ["src", "alt"]
    },
    slot_policy: {
      container: "children",
      row: "children",
      col: "children",
      card: "children"
    }
  };

  // ------------------------------------------------------------------
  // 5) Emit bridge bundle
  // ------------------------------------------------------------------

  return {
    "@kind": "scxpi.ui.renderer.bridge.v1",
    "@v": 1,
    ui: ui,
    atomic_fragment: atomicFragment,
    render_map: renderMap,
    meta: {
      source: "template_pool",
      template_id: templateSheet.template_id,
      slots: resolvedSlots,
      compiled_at: Date.now()
    }
  };
}
```

---

# TEMPLATE_POOL_SHEETS_RAWFIELDS_TO_JSON_v1

This is the **critical missing piece** for large-scale authoring.

Instead of one JSON blob, templates are authored as **structured tables**:

* **slots**
* **ui_nodes**
* **ui_edges**
* **atomic_nodes**
* **atomic_root_vars**

All assembled deterministically.

## Expected Sheet Tabs

For a single template spreadsheet:

```
slots
ui_nodes
ui_edges
atomic_nodes
atomic_root
```

## A) `slots` tab

| slot     | type   | default     |
| -------- | ------ | ----------- |
| headline | string | Ship faster |
| cta      | string | Get Started |

## B) `ui_nodes` tab

| node_id | type      | text | text_slot | label_slot |
| ------- | --------- | ---- | --------- | ---------- |
| ui_root | container |      |           |            |
| ui_h1   | text      |      | headline  |            |
| ui_btn  | button    |      |           | cta        |

## C) `ui_edges` tab

| parent  | child  |
| ------- | ------ |
| ui_root | ui_h1  |
| ui_root | ui_btn |

## D) `atomic_nodes` tab

| node_id | classes                | vars_json         |
| ------- | ---------------------- | ----------------- |
| ui_root | asx-container,asx-grid | {"--asx-gap":0.5} |
| ui_btn  | asx-btn,asx-primary    | {}                |

## E) `atomic_root` tab

| var             | value |
| --------------- | ----- |
| --asx-radius    | 0.65  |
| --asx-elevation | 0.5   |

## GAS Assembler Implementation

```javascript
// ============================================================================
// TEMPLATE_POOL_SHEETS_RAWFIELDS_TO_JSON_v1 (GAS)
// Assemble structured Sheets tabs into template sheet JSON
// ============================================================================

function TEMPLATE_POOL_SHEETS_RAWFIELDS_TO_JSON(spreadsheetId, meta) {
  var ss = SpreadsheetApp.openById(spreadsheetId);

  // -----------------------------
  // slots
  // -----------------------------
  var slots = {};
  readTable(ss, "slots", function(row) {
    slots[row.slot] = {
      type: row.type,
      default: row.default
    };
  });

  // -----------------------------
  // ui_nodes
  // -----------------------------
  var uiNodes = {};
  readTable(ss, "ui_nodes", function(row) {
    uiNodes[row.node_id] = {
      type: row.type
    };
    if (row.text) uiNodes[row.node_id].text = row.text;
    if (row.text_slot) uiNodes[row.node_id].text_slot = row.text_slot;
    if (row.label_slot) uiNodes[row.node_id].label_slot = row.label_slot;
  });

  // -----------------------------
  // ui_edges
  // -----------------------------
  readTable(ss, "ui_edges", function(row) {
    var p = row.parent;
    var c = row.child;
    if (!uiNodes[p]) throw new Error("Missing parent node: " + p);
    if (!uiNodes[p].children) uiNodes[p].children = [];
    uiNodes[p].children.push(c);
  });

  // -----------------------------
  // atomic nodes
  // -----------------------------
  var atomicNodes = {};
  readTable(ss, "atomic_nodes", function(row) {
    atomicNodes[row.node_id] = {
      classes: splitCSV(row.classes),
      vars: row.vars_json ? JSON.parse(row.vars_json) : {}
    };
  });

  // -----------------------------
  // atomic root vars
  // -----------------------------
  var rootVars = {};
  readTable(ss, "atomic_root", function(row) {
    rootVars[row.var] = row.value;
  });

  // -----------------------------
  // Assemble template JSON
  // -----------------------------
  return {
    "@kind": meta.kind,
    "@v": 1,
    template_id: meta.template_id,
    slots: slots,
    ui_fragment: {
      "@kind": "xjson.ui.fragment.v1",
      root: meta.root,
      nodes: uiNodes
    },
    atomic_fragment: {
      "@kind": "atomic.xjson.fragment.v1",
      "@v": 1,
      "@atomic": {
        ":root": { vars: rootVars },
        nodes: atomicNodes
      }
    }
  };
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function readTable(ss, tabName, fn) {
  var sh = ss.getSheetByName(tabName);
  if (!sh) return;

  var rows = sh.getDataRange().getValues();
  if (rows.length < 2) return;

  var header = rows[0].map(String);
  for (var i = 1; i < rows.length; i++) {
    var row = {};
    for (var c = 0; c < header.length; c++) {
      row[header[c]] = rows[i][c];
    }
    fn(row);
  }
}

function splitCSV(s) {
  return String(s || "")
    .split(",")
    .map(function(x) { return x.trim(); })
    .filter(Boolean);
}
```

---

# End-to-End Flow (Now Complete)

```
Sheets (raw fields)
   ↓ RAWFIELDS_TO_JSON
Template Sheet JSON
   ↓ COMPILE_TO_BRIDGE (slots)
scxpi.ui.renderer.bridge.v1
   ↓ Ghost Renderer / Patch
DOM / ASXR / SVG UI
```

This satisfies your constraint:

> **“No way it can just pull designs without mapped schema sheets.”**

The LLM:

* can suggest `template_id`
* can suggest `slots`
* **cannot invent structure**

## Next optional emits (if you want to lock it further)

1. **TEMPLATE_POOL_SCHEMA_VALIDATOR_v1**
2. **TEMPLATE_POOL_VERSION_MIGRATOR_v1**
3. **TEMPLATE_POOL_PATCH_THEME_v1** (swap glass / neon / dark)
4. **TEMPLATE_POOL_EXPORT_STATIC_SITE_v1**

---

# TEMPLATE_POOL_SCHEMA_VALIDATOR_v1

**Purpose**  
Hard-fail any template that does not conform to required structure.  
This prevents “creative drift” and guarantees Ghost/ASXR stability.

### Validation scope (v1)

* Template sheet JSON
* UI fragment integrity
* Atomic fragment integrity
* Slot definitions
* Graph sanity (nodes + edges)
* No orphan references

### GAS Implementation

```javascript
// ============================================================================
// TEMPLATE_POOL_SCHEMA_VALIDATOR_v1 (GAS)
// Validates template sheet JSON against required invariants
// ============================================================================

function TEMPLATE_POOL_SCHEMA_VALIDATOR(template) {
  var errors = [];

  // -------- Core fields --------
  if (!template || typeof template !== "object") {
    return fail("Template is not an object");
  }
  req(template["@kind"], "@kind missing");
  req(template["@v"] === 1, "@v must be 1");
  req(template.template_id, "template_id missing");

  // -------- Slots --------
  var slots = template.slots || {};
  for (var k in slots) {
    var s = slots[k];
    if (!s || typeof s !== "object") errors.push("Slot invalid: " + k);
    if (!s.type) errors.push("Slot type missing: " + k);
  }

  // -------- UI Fragment --------
  var ui = template.ui_fragment;
  if (!ui || ui["@kind"] !== "xjson.ui.fragment.v1") {
    errors.push("ui_fragment missing or wrong kind");
  } else {
    validateUIFragment(ui, errors);
  }

  // -------- Atomic Fragment --------
  var atomic = template.atomic_fragment;
  if (!atomic || atomic["@kind"] !== "atomic.xjson.fragment.v1") {
    errors.push("atomic_fragment missing or wrong kind");
  } else {
    validateAtomicFragment(atomic, errors);
  }

  if (errors.length) {
    return { ok: false, errors: errors };
  }
  return { ok: true };

  // -------- Helpers --------

  function req(cond, msg) { if (!cond) errors.push(msg); }

  function fail(msg) { return { ok: false, errors: [msg] }; }
}

function validateUIFragment(ui, errors) {
  var nodes = ui.nodes || {};
  var root = ui.root;
  if (!nodes[root]) errors.push("UI root missing: " + root);

  // Check node integrity
  for (var id in nodes) {
    var n = nodes[id];
    if (!n.type) errors.push("UI node missing type: " + id);

    if (n.children) {
      for (var i = 0; i < n.children.length; i++) {
        var c = n.children[i];
        if (!nodes[c]) errors.push("UI child missing: " + c + " (parent " + id + ")");
      }
    }
  }
}

function validateAtomicFragment(atomic, errors) {
  var nodes = atomic["@atomic"] && atomic["@atomic"].nodes ? atomic["@atomic"].nodes : {};
  for (var id in nodes) {
    var n = nodes[id];
    if (n.classes && !Array.isArray(n.classes)) {
      errors.push("Atomic classes must be array: " + id);
    }
    if (n.vars && typeof n.vars !== "object") {
      errors.push("Atomic vars must be object: " + id);
    }
  }
}
```

---

# TEMPLATE_POOL_VERSION_MIGRATOR_v1

**Purpose**  
Upgrade older template versions → **current v1** without breaking builds.

This allows **long-lived template pools**.

### Supported migrations (v1)

* v0 → v1
* field renames
* slot normalization
* atomic fragment normalization

### GAS Implementation

```javascript
// ============================================================================
// TEMPLATE_POOL_VERSION_MIGRATOR_v1 (GAS)
// Migrates older template versions forward deterministically
// ============================================================================

function TEMPLATE_POOL_VERSION_MIGRATOR(template) {
  if (!template["@v"] || template["@v"] === 1) {
    return { migrated: false, template: template };
  }

  var v = Number(template["@v"]);
  var t = JSON.parse(JSON.stringify(template)); // clone

  // ---- v0 → v1 ----
  if (v === 0) {
    // Rename fields
    if (t.ui && !t.ui_fragment) {
      t.ui_fragment = t.ui;
      delete t.ui;
    }

    // Normalize slots
    if (t.slots) {
      for (var k in t.slots) {
        if (typeof t.slots[k] !== "object") {
          t.slots[k] = { type: "string", default: t.slots[k] };
        }
      }
    }

    // Normalize atomic
    if (t.atomic && !t.atomic_fragment) {
      t.atomic_fragment = {
        "@kind": "atomic.xjson.fragment.v1",
        "@v": 1,
        "@atomic": t.atomic
      };
      delete t.atomic;
    }

    t["@v"] = 1;
    return { migrated: true, from: 0, to: 1, template: t };
  }

  throw new Error("Unsupported migration path: v" + v);
}
```

---

# TEMPLATE_POOL_PATCH_THEME_v1

**(Glass / Neon / Dark)**

**Purpose**  
Apply **theme overlays** without touching structure or slots.

This works by **patching only atomic variables + classes**.

### Theme model

* themes are **atomic patches**
* composable
* reversible

## Theme Patch Definitions

```javascript
var TEMPLATE_POOL_THEME_PRESETS = {
  glass: {
    root_vars: {
      "--asx-surface-opacity": 0.18,
      "--asx-blur": "18px",
      "--asx-border-alpha": 0.22
    },
    class_map: {
      add: ["asx-glass"],
      remove: ["asx-neon", "asx-dark"]
    }
  },

  neon: {
    root_vars: {
      "--asx-glow": 0.8,
      "--asx-accent-intensity": 1
    },
    class_map: {
      add: ["asx-neon"],
      remove: ["asx-glass", "asx-dark"]
    }
  },

  dark: {
    root_vars: {
      "--asx-bg": "#020617",
      "--asx-fg": "#e6fffa"
    },
    class_map: {
      add: ["asx-dark"],
      remove: ["asx-glass", "asx-neon"]
    }
  }
};
```

## GAS Theme Patch Engine

```javascript
// ============================================================================
// TEMPLATE_POOL_PATCH_THEME_v1 (GAS)
// Applies theme overlays to atomic fragment only
// ============================================================================

function TEMPLATE_POOL_PATCH_THEME(template, themeName) {
  var theme = TEMPLATE_POOL_THEME_PRESETS[themeName];
  if (!theme) throw new Error("Unknown theme: " + themeName);

  var t = JSON.parse(JSON.stringify(template));
  var atomic = t.atomic_fragment["@atomic"];

  // ---- root vars ----
  atomic[":root"] = atomic[":root"] || { vars: {} };
  var rv = atomic[":root"].vars;
  for (var k in theme.root_vars) {
    rv[k] = theme.root_vars[k];
  }

  // ---- node classes ----
  var nodes = atomic.nodes || {};
  for (var id in nodes) {
    var cls = nodes[id].classes || [];
    cls = cls.filter(function(c) { return theme.class_map.remove.indexOf(c) === -1; });
    cls = cls.concat(theme.class_map.add);
    nodes[id].classes = uniq(cls);
  }

  return t;
}

function uniq(arr) {
  var o = {};
  var out = [];
  for (var i = 0; i < arr.length; i++) {
    var k = arr[i];
    if (!o[k]) {
      o[k] = 1;
      out.push(k);
    }
  }
  return out;
}
```

---

# End-to-End Lockdown Pipeline (Now Complete)

```
Sheets (raw fields)
   ↓ RAWFIELDS_TO_JSON
Template JSON
   ↓ SCHEMA_VALIDATOR
   ↓ VERSION_MIGRATOR
   ↓ PATCH_THEME (optional)
   ↓ COMPILE_TO_BRIDGE
Render Bundle
   ↓ Ghost Renderer / Patch
```

**Result:**

* Templates are **authoritative**
* Themes are **safe overlays**
* Versions evolve **without breakage**
* LLM is **advisory only**

## Natural next (optional, but powerful)

If you want, next I can emit:

* **TEMPLATE_POOL_DIFF_v1** (compare templates visually/structurally)
* **TEMPLATE_POOL_GALLERY_UI_v1** (self-rendering template browser)
* **TEMPLATE_POOL_PERMISSION_RULES_v1** (who can use which templates)
* **TEMPLATE_POOL_EXPORTER_STATIC_HTML_v1**

---

# JSON-Only ASX Tape Registry (Sheets-as-Truth)

## Core Principle (Locked)

> **Nothing is “stored in the editor.”**  
> Everything is **rendered from sheets**.  
> The editor is a *projection*, not a workspace.

This avoids:

* file drift
* stale builds
* editor state corruption
* LLM hallucinated structure

## Mental Model

```
┌─────────────────────────────┐
│   Google Sheets (Project)   │   ← SOURCE OF TRUTH
│─────────────────────────────│
│ nodes                        │
│ atomic                       │
│ slots                        │
│ assets                       │
│ routes                       │
│ metadata                     │
└──────────────┬──────────────┘
               │
        TEMPLATE_POOL_LOADER
               │
        SCXPI COMPILERS
               │
     scxpi.ui.renderer.bridge.v1
               │
      ┌────────┴────────┐
      │                 │
 Ghost Shell        ASXR Runtime
 (preview)          (live app)
```

## What Micronauts Actually Do (Important)

Micronauts **do not build apps**.  
They **link sheets → runtime → preview surfaces**.

They are **IO routers**, not generators.

### Micronaut Responsibilities

* Select project sheet
* Select view (app / page / component / diff / export)
* Load JSON rows
* Invoke SCXPI pipeline
* Render result
* Never persist UI state

This keeps them **stateless and safe**.

## The `.asx` Tape Registry (JSON-Only)

Each **project** gets **one sheet** that functions like a tape manifest.

### Project Sheet Tabs (Canonical)

```
PROJECT_<id>.sheet
├─ registry
├─ nodes
├─ atomic
├─ slots
├─ assets
├─ routes
├─ themes
├─ history (optional)
```

This is your **ASX Tape**, just spreadsheet-backed.

No JS files.  
No HTML files.  
No CSS files.

Only **structured rows**.

## Example: registry tab

| key              | value              |
| ---------------- | ------------------ |
| project_id       | landing_emerald_v1 |
| runtime          | ghost+asxr         |
| entry_node       | root               |
| default_theme    | glass              |
| version          | 1                  |
| last_compiled_at | 2026-01-15T04:02Z  |

## Example: nodes tab

| node_id | type      | parent | order | text_slot | children      |
| ------- | --------- | ------ | ----- | --------- | ------------- |
| root    | container |        | 0     |           | hero,features |
| hero    | card      | root   | 0     | title     | cta           |
| cta     | button    | hero   | 0     | cta_text  |               |

This becomes **xjson.ui.fragment.v1** deterministically.

## Runtime Behavior (Critical)

### 1) User opens project

Micronaut selects project sheet.

### 2) Loader pulls sheets

No editor state. No cache beyond runtime memory.

### 3) SCXPI pipeline runs

```
RAWFIELDS → JSON
→ VALIDATE
→ MIGRATE
→ PATCH_THEME
→ COMPILE_TO_BRIDGE
```

### 4) Runtime renders

* Ghost Shell → preview
* ASXR → live app

### 5) User “edits”

Edits **only modify sheet rows**.

There is **no save button** in the editor because nothing lives there.

## Why This Is Better Than Files

### Compared to file-based apps

| Files              | Sheet-based ASX |
| ------------------ | --------------- |
| Drift              | Deterministic   |
| Merge conflicts    | Row-level edits |
| Partial writes     | Atomic updates  |
| LLM hallucinations | Schema-locked   |
| Toolchain fragile  | Runtime-pure    |

## Where “App Files” Still Exist

You’re right: **not everything is GAS**.

### What *is* a file

* Micronaut UI JS
* Ghost Shell renderer
* ASXR runtime
* Preview widgets
* Exporters
* CLI tools

### What is *never* a file

* App UI
* Layout
* Theme
* Structure
* Content
* Navigation

Those live **only in sheets**.

## This Is a JSON-Only Tape System

You’ve effectively created:

> **A fileless app platform**  
> where apps are *rendered*, not *built*.

The “bot”:

* does not emit files
* does not store artifacts
* does not own state

It only **maps intent → sheet mutations**.

## What This Enables Next (Naturally)

If you want, the next logical extensions are:

1. **Per-project diff timelines** (sheet history → visual diff)
2. **Fork project sheet → instant app fork**
3. **Marketplace = read-only shared sheets**
4. **Runtime hot-swap between Ghost / ASXR / SVG**
5. **Offline export = snapshot of compiled bridge**

No new concepts required — just more projections.

If you want, next I can:

* formalize the **PROJECT_SHEET_SCHEMA_v1**
* emit a **Micronaut Project Switcher**
* define **sheet-level permissions**
* or map this directly to **ASX-R conformance**

Just point to the next fold.

---

# PROJECT_SHEET_DIFF_TIMELINE_v1

## What it is

A **deterministic timeline** of a project sheet’s evolution that can drive:

* **structural diffs** (nodes/edges/slots/atomic/routes)
* **render diffs** (bridge bundles + optional DOM patch bundles)
* **proof seals** (hash chain over snapshots)

It works even if you never store “files” — the **sheet rows are the tape**.

## 1) Timeline data model

### A) Snapshot

A snapshot is the **canonical JSON assembly** of a sheet at a point in time.

```json
{
  "@kind": "asx.project.sheet.snapshot.v1",
  "@v": 1,
  "sheet_id": "…",
  "project_id": "…",
  "rev": 12,
  "at": "2026-01-15T05:33:01Z",
  "actor": "email:someone@x.com",
  "reason": "edit nodes",
  "raw": { "tabs": { "registry": [], "nodes": [], "atomic_nodes": [], "atomic_root": [], "slots": [], "routes": [], "themes": [], "assets": [], "permissions": [] } },
  "assembled": { "@kind": "tpl.project.v1", "@v": 1, "template_id": "…", "slots": {}, "ui_fragment": {}, "atomic_fragment": {} },
  "conformance": { "@kind": "asx.project.sheet.conformance.v1", "@v": 1, "overall": "pass", "vectors": [] },
  "hash": "sha256:…"
}
```

### B) Diff event

Each diff event compares two snapshots:

```json
{
  "@kind": "asx.project.sheet.diff.event.v1",
  "@v": 1,
  "from_rev": 11,
  "to_rev": 12,
  "at": "2026-01-15T05:33:01Z",
  "diff": {
    "@kind": "template.pool.diff.v1",
    "@v": 1,
    "structural": { "...": "…" },
    "atomic": { "...": "…" },
    "summary": { "structural_changes": 2, "atomic_changes": 1 }
  },
  "bridge_patch": {
    "@kind": "scxpi.ui.patch.bundle.v1",
    "@v": 1
  },
  "seal": {
    "@kind": "asx.proof.seal.v1",
    "@v": 1,
    "prev": "sha256:…",
    "curr": "sha256:…",
    "chain": "sha256(prev||curr||diff)"
  }
}
```

### C) Timeline

```json
{
  "@kind": "asx.project.sheet.diff.timeline.v1",
  "@v": 1,
  "sheet_id": "…",
  "project_id": "…",
  "head_rev": 12,
  "events": [ /* diff.event.v1 */ ]
}
```

## 2) Where revisions come from (two valid modes)

### Mode 1 (recommended): **history tab** (append-only)

Add a `history` tab where every update appends a row with:

* `rev` (monotonic)
* `at`
* `actor`
* `reason`
* `tabs_changed` (csv)
* `snapshot_json` (optional) OR a lightweight delta pointer

This is deterministic and works in GAS.

### Mode 2: Sheets Version History (harder in GAS)

Google Sheets “version history” isn’t easily enumerable from plain Apps Script without Drive API + permissions and still won’t give you table-level diffs cleanly. For ASX law, **Mode 1 wins**.

## 3) GAS emit: PROJECT_SHEET_DIFF_TIMELINE_v1

This assumes:

* you already have `PROJECT_SHEET_CONFORMANCE_VECTORS(sheetId)`
* you already have “rawfields → template json” assembler (the one we emitted / used in the switcher)
* you maintain a `history` tab with `rev, at, actor, reason, snapshot_json` (snapshot_json can be optional; if absent, we reconstruct from current tabs only for head, and diffs are “best effort”)

### GAS code

```javascript
// ============================================================================
// PROJECT_SHEET_DIFF_TIMELINE_v1 (GAS)
// Builds a diff timeline from an append-only "history" tab.
// ============================================================================

function PROJECT_SHEET_DIFF_TIMELINE(sheetId, opts) {
  opts = opts || {};
  var maxEvents = opts.max_events || 50;

  var ss = SpreadsheetApp.openById(sheetId);

  // Load history rows (required for real timeline)
  var hist = readTab_(ss, "history");
  if (!hist) {
    return {
      "@kind": "asx.project.sheet.diff.timeline.v1",
      "@v": 1,
      "sheet_id": sheetId,
      "project_id": "",
      "head_rev": 0,
      "events": [],
      "note": "history tab missing; timeline unavailable"
    };
  }

  // Sort by rev asc (string-safe)
  hist.sort(function(a, b) { return Number(a.rev || 0) - Number(b.rev || 0); });

  // Build snapshots list (limited)
  var snaps = [];
  for (var i = Math.max(0, hist.length - (maxEvents + 1)); i < hist.length; i++) {
    var row = hist[i];
    var snap = buildSnapshotFromHistoryRow_(ss, sheetId, row);
    snaps.push(snap);
  }

  // Build diff events
  var events = [];
  for (var j = 1; j < snaps.length; j++) {
    var prev = snaps[j - 1];
    var curr = snaps[j];

    var diff = TEMPLATE_SHEET_DIFF_V1_(prev.assembled, curr.assembled);

    var chainSeal = {
      "@kind": "asx.proof.seal.v1",
      "@v": 1,
      "prev": prev.hash,
      "curr": curr.hash,
      "chain": "sha256:" + sha256Hex_(prev.hash + "||" + curr.hash + "||" + JSON.stringify(diff))
    };

    // optional: patch bundle placeholder
    var patch = {
      "@kind": "scxpi.ui.patch.bundle.v1",
      "@v": 1,
      "from_rev": prev.rev,
      "to_rev": curr.rev,
      "ops": [] // you can fill later with node-level ops
    };

    events.push({
      "@kind": "asx.project.sheet.diff.event.v1",
      "@v": 1,
      "from_rev": prev.rev,
      "to_rev": curr.rev,
      "at": curr.at,
      "diff": diff,
      "bridge_patch": patch,
      "seal": chainSeal
    });
  }

  var projectId = snaps.length ? snaps[snaps.length - 1].project_id : "";

  return {
    "@kind": "asx.project.sheet.diff.timeline.v1",
    "@v": 1,
    "sheet_id": sheetId,
    "project_id": projectId,
    "head_rev": snaps.length ? snaps[snaps.length - 1].rev : 0,
    "events": events
  };
}

// ----------------------------------------------------------------------------
// Snapshot builder
// ----------------------------------------------------------------------------

function buildSnapshotFromHistoryRow_(ss, sheetId, row) {
  var rev = Number(row.rev || 0);
  var at = row.at ? String(row.at) : new Date().toISOString();
  var actor = row.actor ? String(row.actor) : "";
  var reason = row.reason ? String(row.reason) : "";

  var rawTabs;
  if (row.snapshot_json && String(row.snapshot_json).trim()) {
    // Stored snapshot JSON (preferred)
    rawTabs = JSON.parse(String(row.snapshot_json));
  } else {
    // Fallback: reconstruct current tabs (head-only accuracy)
    rawTabs = {
      tabs: {
        registry: readTab_(ss, "registry") || [],
        slots: readTab_(ss, "slots") || [],
        nodes: readTab_(ss, "nodes") || [],
        routes: readTab_(ss, "routes") || [],
        atomic_nodes: readTab_(ss, "atomic_nodes") || [],
        atomic_root: readTab_(ss, "atomic_root") || [],
        themes: readTab_(ss, "themes") || [],
        assets: readTab_(ss, "assets") || [],
        permissions: readTab_(ss, "permissions") || []
      }
    };
  }

  var reg = toRegistry_(rawTabs.tabs.registry || []);
  var projectId = reg.project_id || "";

  // Assemble to template sheet JSON (same deterministic rules as switcher)
  var entry = reg.entry_node || "root";
  var assembled = RAWFIELDS_TABS_TO_TEMPLATE_SHEET_V1_(rawTabs.tabs, {
    template_id: projectId || ("project_" + sheetId),
    kind: "tpl.project.v1",
    root: entry
  });

  var conformance = PROJECT_SHEET_CONFORMANCE_VECTORS(sheetId);

  // Hash snapshot (stable-ish): use assembled+rev+entry
  var hashPayload = JSON.stringify({ rev: rev, at: at, assembled: assembled });
  var hash = "sha256:" + sha256Hex_(hashPayload);

  return {
    "@kind": "asx.project.sheet.snapshot.v1",
    "@v": 1,
    "sheet_id": sheetId,
    "project_id": projectId,
    "rev": rev,
    "at": at,
    "actor": actor,
    "reason": reason,
    "raw": rawTabs,
    "assembled": assembled,
    "conformance": conformance,
    "hash": hash
  };
}

// ----------------------------------------------------------------------------
// Minimal diff engine (template.pool.diff.v1 compatible)
// ----------------------------------------------------------------------------

function TEMPLATE_SHEET_DIFF_V1_(tA, tB) {
  // Very small subset; can be swapped with your richer diff module later.
  var uiA = tA.ui_fragment, uiB = tB.ui_fragment;
  var aNodes = (uiA && uiA.nodes) ? uiA.nodes : {};
  var bNodes = (uiB && uiB.nodes) ? uiB.nodes : {};

  var added = [], removed = [], changed = [];

  for (var id in bNodes) if (!aNodes[id]) added.push(id);
  for (var id2 in aNodes) if (!bNodes[id2]) removed.push(id2);

  for (var id3 in aNodes) {
    if (!bNodes[id3]) continue;
    var na = aNodes[id3], nb = bNodes[id3];
    var deltas = [];
    if (String(na.type) !== String(nb.type)) deltas.push({ field: "type", a: na.type, b: nb.type });

    var ca = (na.children || []).join(",");
    var cb = (nb.children || []).join(",");
    if (ca !== cb) deltas.push({ field: "children", a: na.children || [], b: nb.children || [] });

    if (deltas.length) changed.push({ node_id: id3, deltas: deltas });
  }

  return {
    "@kind": "template.pool.diff.v1",
    "@v": 1,
    "template_a": tA.template_id,
    "template_b": tB.template_id,
    "structural": { nodes: { added: added, removed: removed, changed: changed } },
    "atomic": { nodes: { added: [], removed: [], changed: [] }, root_vars: { added: [], removed: [], changed: [] } },
    "summary": { structural_changes: added.length + removed.length + changed.length, atomic_changes: 0 }
  };
}

// ----------------------------------------------------------------------------
// Deterministic assembler: tabs -> template sheet json (browser/GAS shared idea)
// ----------------------------------------------------------------------------

function RAWFIELDS_TABS_TO_TEMPLATE_SHEET_V1_(tabs, meta) {
  // slots
  var slots = {};
  (tabs.slots || []).forEach(function(r) {
    var s = String(r.slot || "").trim();
    if (!s) return;
    slots[s] = { type: String(r.type || "string"), "default": r["default"] };
  });

  // nodes
  var uiNodes = {};
  (tabs.nodes || []).forEach(function(r) {
    var id = String(r.node_id || "").trim();
    if (!id) return;
    uiNodes[id] = { type: String(r.type || "container") };
    if (r.text) uiNodes[id].text = String(r.text);
    if (r.text_slot) uiNodes[id].text_slot = String(r.text_slot);
    if (r.label) uiNodes[id].label = String(r.label);
    if (r.label_slot) uiNodes[id].label_slot = String(r.label_slot);
    if (r.action) uiNodes[id].action = String(r.action);
    uiNodes[id].parent = r.parent ? String(r.parent) : "";
    uiNodes[id].order = (r.order != null && r.order !== "") ? Number(r.order) : 0;
  });

  // parent/order -> children (derived deterministically)
  var byParent = {};
  for (var id in uiNodes) {
    var p = uiNodes[id].parent;
    if (!p) continue;
    if (!byParent[p]) byParent[p] = [];
    byParent[p].push({ id: id, order: uiNodes[id].order });
  }
  for (var pid in byParent) {
    byParent[pid].sort(function(a, b) { return a.order - b.order; });
    uiNodes[pid] = uiNodes[pid] || { type: "container" };
    uiNodes[pid].children = byParent[pid].map(function(x) { return x.id; });
  }

  // atomic nodes
  var atomicNodes = {};
  (tabs.atomic_nodes || []).forEach(function(r) {
    var idn = String(r.node_id || "").trim();
    if (!idn) return;
    var classes = String(r.classes || "").split(",").map(function(s) { return s.trim(); }).filter(Boolean);
    var vars = {};
    if (r.vars_json && String(r.vars_json).trim()) {
      try { vars = JSON.parse(String(r.vars_json)); } catch (e) { vars = {}; }
    }
    atomicNodes[idn] = { classes: classes, vars: vars };
  });

  // atomic root vars
  var rootVars = {};
  (tabs.atomic_root || []).forEach(function(r) {
    var k = String(r.var || "").trim();
    if (!k) return;
    rootVars[k] = r.value;
  });

  return {
    "@kind": meta.kind,
    "@v": 1,
    "template_id": meta.template_id,
    "slots": slots,
    "ui_fragment": { "@kind": "xjson.ui.fragment.v1", "root": meta.root, "nodes": stripInternal_(uiNodes) },
    "atomic_fragment": { "@kind": "atomic.xjson.fragment.v1", "@v": 1, "@atomic": { ":root": { vars: rootVars }, "nodes": atomicNodes } }
  };
}

function stripInternal_(uiNodes) {
  // remove parent/order from fragment (runtime derives children)
  var out = {};
  for (var id in uiNodes) {
    var n = {};
    for (var k in uiNodes[id]) {
      if (k === "parent" || k === "order") continue;
      n[k] = uiNodes[id][k];
    }
    out[id] = n;
  }
  return out;
}

function toRegistry_(rows) {
  var o = {};
  (rows || []).forEach(function(r) {
    var k = String(r.key || "").trim();
    if (!k) return;
    o[k] = String(r.value || "");
  });
  return o;
}

function readTab_(ss, name) {
  var sh = ss.getSheetByName(name);
  if (!sh) return null;
  var v = sh.getDataRange().getValues();
  if (v.length < 2) return [];
  var h = v[0].map(String);
  var out = [];
  for (var r = 1; r < v.length; r++) {
    var o = {};
    for (var c = 0; c < h.length; c++) o[h[c]] = v[r][c];
    // skip empty
    var any = false;
    for (var k in o) if (String(o[k] || "").trim()) { any = true; break; }
    if (any) out.push(o);
  }
  return out;
}

// SHA-256 helper using Apps Script Utilities
function sha256Hex_(s) {
  var bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, s, Utilities.Charset.UTF_8);
  return bytes.map(function(b) { var v = (b < 0 ? b + 256 : b); return (v < 16 ? "0" : "") + v.toString(16); }).join("");
}
```

### What to add to your project sheet

Add a `history` tab with columns:

* `rev` (number)
* `at` (ISO string)
* `actor` (principal string)
* `reason` (string)
* `snapshot_json` (optional JSON; recommended for true diffs)

If you *don’t* store `snapshot_json`, the timeline still emits, but historical diffs will be “best effort” (because it can’t reconstruct old states).

---

# LANGPACK_REGISTRY_SCHEMA_v1 (sheet-backed)

## What it is

A **single registry spreadsheet** that indexes many **language packs**.  
Each language pack can live in:

* its own spreadsheet (recommended), or
* a bundle of tabs inside the registry spreadsheet (small scale)

The registry is the “tape index”. The packs are the “tapes”.

## A) Registry Spreadsheet: required tabs

**Required**

* `langpacks`
* `artifacts`
* `versions`
* `capabilities`

**Optional**

* `aliases`
* `licenses`
* `tests`
* `permissions`

### 1) `langpacks` tab (the catalog)

Columns (required):

* `lang_id` (string, unique) — canonical id: `js`, `ts`, `py`, `go`, `rust`, `java`, `cpp`, `c`, `php`, `sql`, `html`, `css`, `json`, `yaml`, `toml`, `bash`, …
* `title` (string) — display name
* `family` (string) — e.g. `c-like`, `ml`, `shell`, `markup`, `query`
* `status` (enum) — `active|deprecated|draft`
* `pack_sheet_id` (string) — Google Sheet ID containing the pack tables
* `default_version` (string) — e.g. `1.0.0`
* `entry_artifact_id` (string) — e.g. `oracle.core.v1`
* `tags` (csv)
* `updated_at` (ISO)

Columns (optional):

* `notes`
* `homepage`

Example:

| lang_id | title      | family | status | pack_sheet_id | default_version | entry_artifact_id | tags     | updated_at        |
| ------- | ---------- | ------ | ------ | ------------- | --------------- | ----------------- | -------- | ----------------- |
| js      | JavaScript | c-like | active | 1Abc…         | 1.0.0           | oracle.core.v1    | web,node | 2026-01-15T05:44Z |

### 2) `artifacts` tab (what exists per language)

Each row declares an artifact and where it lives.

Columns (required):

* `lang_id` (FK → langpacks.lang_id)
* `artifact_id` (string) — e.g. `grammar.ebnf.v1`, `ast.schema.v1`, `oracle.core.v1`, `prettyprint.rules.v1`, `tokens.profile.v1`
* `kind` (enum) — `grammar|ast_schema|oracle|printer|tests|token_profile|lint_rules`
* `version` (string) — `1.0.0`
* `location_kind` (enum) — `sheet_tab|json_url|inline_json`
* `location_ref` (string) — if `sheet_tab`, tab name in pack sheet; if `json_url`, URL; if inline_json, a key
* `hash` (string) — optional proof hash
* `required` (bool) — if true, pack is invalid without it

Example:

| lang_id | artifact_id     | kind    | version | location_kind | location_ref | required |
| ------- | --------------- | ------- | ------- | ------------- | ------------ | -------- |
| js      | oracle.core.v1  | oracle  | 1.0.0   | sheet_tab     | oracle       | true     |
| js      | grammar.ebnf.v1 | grammar | 1.0.0   | sheet_tab     | grammar      | true     |

### 3) `versions` tab (version policy + migration)

Columns:

* `lang_id`
* `from_version`
* `to_version`
* `migrator_artifact_id` (e.g. `migrate.v1`)
* `notes`

### 4) `capabilities` tab (runtime expectations)

Columns:

* `lang_id`
* `cap` (enum-ish string) — `parse`, `ast`, `format`, `lint`, `typecheck`, `eval_safe`, `eval_unsafe`
* `level` (number 0–3) — 0 none, 1 partial, 2 good, 3 strict
* `notes`

Example:

| lang_id | cap       | level |
| ------- | --------- | ----- |
| js      | parse     | 3     |
| js      | typecheck | 1     |

### 5) Optional `aliases` tab

Columns:

* `alias` (string) — `javascript`, `node`, `ecmascript`
* `lang_id` (FK)

## B) Per-Language Pack Spreadsheet (recommended)

Each pack sheet is **typed tables**. Required tabs for v1:

* `meta`
* `grammar` (or `grammar_rules`)
* `ast_schema`
* `oracle`
* `tests`

Optional:

* `printer`
* `lint_rules`
* `token_profile`

### `meta` tab

Columns: `key`, `value`  
Required keys:

* `lang_id`
* `version`
* `oracle_entry` (usually `oracle.core.v1`)
* `grammar_kind` (`ebnf|peg|parser_tables`)
* `ast_kind` (`json_schema|custom`)
* `updated_at`

## Registry JSON view (canonical “assembled” form)

Your GAS loader will emit this:

```json
{
  "@kind": "asx.langpack.registry.v1",
  "@v": 1,
  "langpacks": {
    "js": {
      "title": "JavaScript",
      "pack_sheet_id": "1Abc…",
      "default_version": "1.0.0",
      "entry_artifact_id": "oracle.core.v1",
      "artifacts": {
        "oracle.core.v1": { "kind": "oracle", "version": "1.0.0", "location": { "kind": "sheet_tab", "ref": "oracle" } },
        "grammar.ebnf.v1": { "kind": "grammar", "version": "1.0.0", "location": { "kind": "sheet_tab", "ref": "grammar" } }
      },
      "capabilities": { "parse": 3, "ast": 2, "format": 1 }
    }
  },
  "aliases": { "javascript": "js", "ecmascript": "js" }
}
```

---

# CODE_ORACLE_ABI_v1 (JS / Python / GAS)

### Goal

A uniform interface:

**parse → ast → score → errors**

The oracle is not “the model”. It is the **legality governor**.

## 1) ABI Envelope

### Request

```json
{
  "@kind": "code.oracle.request.v1",
  "@v": 1,
  "lang_id": "js",
  "mode": "parse|lint|format|score",
  "input": {
    "source": "let x = 1;",
    "filename": "main.js",
    "context": {
      "dialect": "ecmascript",
      "strict": true
    }
  },
  "options": {
    "max_errors": 50,
    "timeout_ms": 1500,
    "return_ast": true,
    "return_tokens": false,
    "return_formatted": false
  }
}
```

### Response

```json
{
  "@kind": "code.oracle.response.v1",
  "@v": 1,
  "lang_id": "js",
  "mode": "parse",
  "ok": true,
  "score": {
    "legality": 1.0,
    "style": 0.75,
    "completeness": 0.9,
    "risk": 0.0
  },
  "ast": { "@kind": "ast.tree.v1", "root": "Program", "nodes": [] },
  "errors": [],
  "warnings": [],
  "formatted": null,
  "metrics": {
    "parse_ms": 7,
    "ast_nodes": 42,
    "tokens": 0
  }
}
```

## 2) Error object (stable across runtimes)

```json
{
  "code": "E_PARSE|E_LEX|E_AST|E_STYLE|E_TYPE",
  "message": "Unexpected token",
  "severity": "error|warning",
  "span": { "line": 1, "col": 5, "end_line": 1, "end_col": 6 },
  "hint": "Did you forget a semicolon?",
  "rule_id": "js/grammar/…"
}
```

## 3) Deterministic scoring rules (v1)

Scores are 0..1.

* `legality`:
  * 1.0 if parse succeeds with zero errors
  * else `max(0, 1 - errors*0.1 - fatal*0.3)`
* `style`:
  * based on lint warnings count
* `completeness`:
  * optional heuristic (e.g. unmatched braces, missing returns)
* `risk`:
  * optional (e.g. eval usage, shell exec)

These are **rules**, not model opinions.

## 4) JS reference shape (browser/Node)

```js
// CODE_ORACLE_ABI_v1 (JS interface)
export class CodeOracle {
  constructor(langpack) { this.langpack = langpack; }

  /**
   * @param {object} req code.oracle.request.v1
   * @returns {object} code.oracle.response.v1
   */
  run(req) {
    // 1) parse
    // 2) build AST
    // 3) lint/style
    // 4) score
    // This wrapper stays stable; per-language implementation plugs in.
    throw new Error("Not implemented");
  }
}
```

### JS adapter contract (per language)

```js
export const JsOracleAdapter = {
  lang_id: "js",
  parse(source, opts) { /* returns { ok, ast, errors, warnings, metrics } */ },
  format(sourceOrAst, opts) { /* returns { formatted, warnings } */ },
  lint(ast, opts) { /* returns warnings */ },
  score(result, opts) { /* returns score object */ }
};
```

## 5) Python reference shape

```python
# CODE_ORACLE_ABI_v1 (Python)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class OracleResponse:
  kind: str
  v: int
  lang_id: str
  mode: str
  ok: bool
  score: Dict[str, float]
  ast: Optional[Dict[str, Any]]
  errors: List[Dict[str, Any]]
  warnings: List[Dict[str, Any]]
  formatted: Optional[str]
  metrics: Dict[str, Any]

class CodeOracle:
  def __init__(self, adapter):
    self.adapter = adapter

  def run(self, req: Dict[str, Any]) -> OracleResponse:
    src = req["input"]["source"]
    mode = req["mode"]
    opts = req.get("options", {})
    parsed = self.adapter.parse(src, opts)

    out = {
      "@kind": "code.oracle.response.v1",
      "@v": 1,
      "lang_id": req["lang_id"],
      "mode": mode,
      "ok": bool(parsed.get("ok")),
      "score": self.adapter.score(parsed, opts),
      "ast": parsed.get("ast") if opts.get("return_ast") else None,
      "errors": parsed.get("errors", []),
      "warnings": parsed.get("warnings", []),
      "formatted": None,
      "metrics": parsed.get("metrics", {})
    }

    if mode == "format":
      out["formatted"] = self.adapter.format(parsed.get("ast") or src, opts).get("formatted")

    return OracleResponse(
      kind=out["@kind"], v=out["@v"], lang_id=out["lang_id"], mode=out["mode"],
      ok=out["ok"], score=out["score"], ast=out["ast"], errors=out["errors"],
      warnings=out["warnings"], formatted=out["formatted"], metrics=out["metrics"]
    )
```

## 6) GAS reference shape (what GAS can do)

GAS cannot run heavy parsers, but it **can**:

* enforce schema presence
* run lightweight checks
* call external oracle runtimes (optional) while keeping ABI stable

### GAS oracle “broker” (ABI-stable)

```javascript
// CODE_ORACLE_ABI_v1 (GAS broker)
// route=oracle.run.v1&lang_id=js&mode=parse
function api_oracle_run_v1_(e) {
  var req = JSON.parse(e.postData.contents);

  // lightweight: check required fields
  if (!req.lang_id || !req.mode || !req.input || !req.input.source) {
    return ok_({
      "@kind": "code.oracle.response.v1", "@v": 1,
      lang_id: req.lang_id || "",
      mode: req.mode || "",
      ok: false,
      score: { legality: 0, style: 0, completeness: 0, risk: 0 },
      ast: null,
      errors: [{ code: "E_SCHEMA", message: "missing required fields", severity: "error", span: null }],
      warnings: [],
      formatted: null,
      metrics: {}
    });
  }

  // If you have an external oracle service, forward here (optional).
  // Otherwise, provide a minimal heuristic “oracle-lite”.
  var res = oracleLite_(req);

  return ok_(res);
}

function oracleLite_(req) {
  var src = String(req.input.source);

  // super minimal heuristics
  var errors = [];
  var open = (src.match(/\{/g) || []).length;
  var close = (src.match(/\}/g) || []).length;
  if (open !== close) errors.push({ code: "E_PARSE", message: "brace mismatch", severity: "error", span: null });

  var ok = errors.length === 0;

  return {
    "@kind": "code.oracle.response.v1",
    "@v": 1,
    "lang_id": req.lang_id,
    "mode": req.mode,
    "ok": ok,
    "score": {
      "legality": ok ? 1.0 : Math.max(0, 1.0 - errors.length * 0.2),
      "style": 1.0,
      "completeness": ok ? 0.9 : 0.2,
      "risk": 0.0
    },
    "ast": null,
    "errors": errors,
    "warnings": [],
    "formatted": null,
    "metrics": { "oracle": "lite", "len": src.length }
  };
}
```

## Next routes (natural)

If you want to wire this end-to-end, the next two “must haves” are:

1. **LANGPACK_LOADER_GAS_v1**  
   Load registry → load pack sheet → assemble pack JSON (grammar/ast/oracle refs)

2. **TREE_SITTER_ORACLE_ADAPTER_v1** (Node/Python)  
   A real implementation of `parse → ast → errors` for many languages using tree-sitter, while keeping the ABI identical.

---

# LANGPACK_LOADER_GAS_v1 (Google Apps Script)

## What it does

* Reads the **Registry Spreadsheet** tabs: `langpacks`, `artifacts`, `capabilities`, `aliases`, `versions`
* Emits **asx.langpack.registry.v1**
* Can also load a **single pack** (by `lang_id`) and return its artifact payloads (where artifacts are located in the pack sheet)

## Expected registry sheet tabs

* `langpacks` (required)
* `artifacts` (required)
* `capabilities` (required)
* `aliases` (optional)
* `versions` (optional)

## GAS code: `langpack_loader.gs`

```javascript
// ============================================================================
// LANGPACK_LOADER_GAS_v1
// Sheet-backed langpack registry loader + per-pack artifact resolver
// ============================================================================

function LANGPACK_LOADER_GAS_v1_loadRegistry(registrySheetId, opts) {
  opts = opts || {};
  var ss = SpreadsheetApp.openById(registrySheetId);

  var langpacksTab = readTab_(ss, "langpacks");
  var artifactsTab = readTab_(ss, "artifacts");
  var capabilitiesTab = readTab_(ss, "capabilities");
  var aliasesTab = readTab_(ss, "aliases");      // optional
  var versionsTab = readTab_(ss, "versions");    // optional

  if (!langpacksTab) throw new Error("Registry missing langpacks tab");
  if (!artifactsTab) throw new Error("Registry missing artifacts tab");
  if (!capabilitiesTab) throw new Error("Registry missing capabilities tab");

  // ---- build langpacks base ----
  var packs = {}; // lang_id -> pack record
  for (var i = 0; i < langpacksTab.length; i++) {
    var r = langpacksTab[i];
    var langId = String(r.lang_id || "").trim();
    if (!langId) continue;

    packs[langId] = {
      title: String(r.title || ""),
      family: String(r.family || ""),
      status: String(r.status || "active"),
      pack_sheet_id: String(r.pack_sheet_id || ""),
      default_version: String(r.default_version || ""),
      entry_artifact_id: String(r.entry_artifact_id || ""),
      tags: splitCsv_(r.tags),
      updated_at: String(r.updated_at || ""),
      notes: String(r.notes || ""),
      homepage: String(r.homepage || ""),
      artifacts: {},     // filled later
      capabilities: {}   // filled later
    };
  }

  // ---- attach artifacts ----
  for (var j = 0; j < artifactsTab.length; j++) {
    var a = artifactsTab[j];
    var lid = String(a.lang_id || "").trim();
    if (!lid || !packs[lid]) continue;

    var artifactId = String(a.artifact_id || "").trim();
    if (!artifactId) continue;

    packs[lid].artifacts[artifactId] = {
      kind: String(a.kind || ""),
      version: String(a.version || ""),
      location: {
        kind: String(a.location_kind || ""),
        ref: String(a.location_ref || "")
      },
      hash: String(a.hash || ""),
      required: toBool_(a.required)
    };
  }

  // ---- attach capabilities (max level if multiple rows) ----
  for (var k = 0; k < capabilitiesTab.length; k++) {
    var c = capabilitiesTab[k];
    var lid2 = String(c.lang_id || "").trim();
    if (!lid2 || !packs[lid2]) continue;

    var cap = String(c.cap || "").trim();
    if (!cap) continue;

    var level = Number(c.level || 0);
    var cur = Number(packs[lid2].capabilities[cap] || 0);
    if (level > cur) packs[lid2].capabilities[cap] = level;
  }

  // ---- aliases ----
  var aliases = {};
  if (aliasesTab) {
    for (var z = 0; z < aliasesTab.length; z++) {
      var al = aliasesTab[z];
      var alias = String(al.alias || "").trim().toLowerCase();
      var lid3 = String(al.lang_id || "").trim();
      if (alias && lid3 && packs[lid3]) aliases[alias] = lid3;
    }
  }

  // ---- versions ----
  var versions = [];
  if (versionsTab) {
    for (var v = 0; v < versionsTab.length; v++) {
      var row = versionsTab[v];
      var lid4 = String(row.lang_id || "").trim();
      if (!lid4 || !packs[lid4]) continue;
      versions.push({
        lang_id: lid4,
        from_version: String(row.from_version || ""),
        to_version: String(row.to_version || ""),
        migrator_artifact_id: String(row.migrator_artifact_id || ""),
        notes: String(row.notes || "")
      });
    }
  }

  // ---- output ----
  return {
    "@kind": "asx.langpack.registry.v1",
    "@v": 1,
    registry_sheet_id: registrySheetId,
    generated_at: new Date().toISOString(),
    langpacks: packs,
    aliases: aliases,
    versions: versions
  };
}

/**
 * Loads one langpack's artifact payloads from its pack sheet.
 * - Resolves artifact locations
 * - If location_kind=sheet_tab, reads that tab and returns rows
 * - If location_kind=json_url, returns the URL only (consumer fetches)
 * - If location_kind=inline_json, reads from pack "inline" tab (optional)
 */
function LANGPACK_LOADER_GAS_v1_loadPack(registrySheetId, langId, opts) {
  opts = opts || {};
  var reg = LANGPACK_LOADER_GAS_v1_loadRegistry(registrySheetId);
  var lid = resolveLangId_(reg, langId);
  if (!lid || !reg.langpacks[lid]) throw new Error("Unknown lang_id: " + langId);

  var packMeta = reg.langpacks[lid];
  if (!packMeta.pack_sheet_id) throw new Error("Missing pack_sheet_id for lang_id: " + lid);

  var ssPack = SpreadsheetApp.openById(packMeta.pack_sheet_id);

  var artifacts = {};
  var missingRequired = [];

  var keys = Object.keys(packMeta.artifacts || {});
  for (var i = 0; i < keys.length; i++) {
    var aid = keys[i];
    var spec = packMeta.artifacts[aid];

    var lk = spec.location.kind;
    var lr = spec.location.ref;

    if (!lk) {
      if (spec.required) missingRequired.push(aid);
      continue;
    }

    if (lk === "sheet_tab") {
      var rows = readTab_(ssPack, lr);
      if (rows === null) {
        if (spec.required) missingRequired.push(aid);
      } else {
        artifacts[aid] = { spec: spec, payload: { kind: "sheet_rows", rows: rows } };
      }
    } else if (lk === "json_url") {
      artifacts[aid] = { spec: spec, payload: { kind: "json_url", url: lr } };
    } else if (lk === "inline_json") {
      // optional: pack sheet has tab named "inline" with columns: key, json
      var inline = readTab_(ssPack, "inline") || [];
      var found = null;
      for (var j = 0; j < inline.length; j++) {
        if (String(inline[j].key || "").trim() === lr) { found = String(inline[j].json || ""); break; }
      }
      if (!found) {
        if (spec.required) missingRequired.push(aid);
      } else {
        artifacts[aid] = { spec: spec, payload: { kind: "inline_json", json: found } };
      }
    } else {
      // unknown kind, keep reference
      artifacts[aid] = { spec: spec, payload: { kind: "unknown_location_kind", ref: spec.location } };
    }
  }

  return {
    "@kind": "asx.langpack.loaded.v1",
    "@v": 1,
    registry_sheet_id: registrySheetId,
    lang_id: lid,
    pack_sheet_id: packMeta.pack_sheet_id,
    loaded_at: new Date().toISOString(),
    meta: {
      title: packMeta.title,
      family: packMeta.family,
      status: packMeta.status,
      default_version: packMeta.default_version,
      entry_artifact_id: packMeta.entry_artifact_id,
      capabilities: packMeta.capabilities
    },
    artifacts: artifacts,
    missing_required: missingRequired
  };
}

// ------------------------------ helpers ------------------------------

function resolveLangId_(reg, langId) {
  var lid = String(langId || "").trim().toLowerCase();
  if (reg.langpacks[lid]) return lid;
  if (reg.aliases && reg.aliases[lid]) return reg.aliases[lid];
  return lid;
}

function readTab_(ss, name) {
  var sh = ss.getSheetByName(name);
  if (!sh) return null;

  var v = sh.getDataRange().getValues();
  if (v.length < 2) return [];

  var h = v[0].map(String);
  var out = [];

  for (var r = 1; r < v.length; r++) {
    var o = {};
    for (var c = 0; c < h.length; c++) o[h[c]] = v[r][c];

    // skip empty row
    var any = false;
    for (var k in o) {
      if (String(o[k] || "").trim()) { any = true; break; }
    }
    if (any) out.push(o);
  }
  return out;
}

function splitCsv_(x) {
  return String(x || "")
    .split(",")
    .map(function(s) { return s.trim(); })
    .filter(Boolean);
}

function toBool_(x) {
  var s = String(x || "").trim().toLowerCase();
  return (s === "true" || s === "1" || s === "yes" || s === "y");
}
```

### Optional GAS router endpoints (if you want it as WebApp)

Add to `doGet/doPost`:

```javascript
function doGet(e) {
  var route = String(e.parameter.route || "");
  var registry = e.parameter.registry_sheet_id;

  if (route === "langpack.registry.load.v1") {
    var out = LANGPACK_LOADER_GAS_v1_loadRegistry(registry);
    return json_(out);
  }

  if (route === "langpack.pack.load.v1") {
    var out2 = LANGPACK_LOADER_GAS_v1_loadPack(registry, e.parameter.lang_id);
    return json_(out2);
  }

  return json_({ ok: false, error: "unknown route" });
}

function json_(o) {
  return ContentService.createTextOutput(JSON.stringify(o))
    .setMimeType(ContentService.MimeType.JSON);
}
```

---

# TREE_SITTER_ORACLE_ADAPTER_v1 (Node)

This is a **real** CODE_ORACLE_ABI_v1 oracle that:

* parses with tree-sitter
* emits a compact AST
* emits errors (tree-sitter does error nodes; plus optional brace heuristics)
* scores legality/style/completeness/risk deterministically

It’s designed as a **multi-language adapter**:

* `lang_id` → module that returns a tree-sitter Language object

## File tree

```
tree-sitter-oracle/
├─ package.json
├─ src/
│  ├─ index.js
│  ├─ abi.js
│  ├─ score.js
│  ├─ ast.js
│  ├─ errors.js
│  └─ languages/
│     ├─ js.js
│     ├─ ts.js
│     ├─ py.js
│     ├─ go.js
│     └─ rust.js
└─ README.md
```

## 1) package.json

```json
{
  "name": "tree-sitter-oracle-adapter-v1",
  "version": "1.0.0",
  "type": "module",
  "main": "src/index.js",
  "scripts": {
    "oracle:stdin": "node src/index.js --stdin",
    "oracle:http": "node src/index.js --http 8787"
  },
  "dependencies": {
    "tree-sitter": "^0.22.6",
    "tree-sitter-javascript": "^0.21.4",
    "tree-sitter-typescript": "^0.21.2",
    "tree-sitter-python": "^0.21.0",
    "tree-sitter-go": "^0.21.0",
    "tree-sitter-rust": "^0.21.1"
  }
}
```

> Add more languages later by installing the relevant `tree-sitter-<lang>` package and adding a module in `src/languages/`.

## 2) src/abi.js — ABI normalize/emit

```js
export function normalizeRequest(req) {
  if (!req || typeof req !== "object") throw new Error("Request must be an object");
  if (req["@kind"] && req["@kind"] !== "code.oracle.request.v1") {
    // allow missing @kind in early testing, but reject wrong kinds
    throw new Error("Wrong @kind");
  }

  const lang_id = String(req.lang_id || "").trim().toLowerCase();
  const mode = String(req.mode || "parse").trim();

  const source = req?.input?.source;
  if (typeof source !== "string") throw new Error("input.source must be a string");

  const options = req.options || {};
  return {
    lang_id,
    mode,
    source,
    filename: String(req?.input?.filename || ""),
    context: req?.input?.context || {},
    options: {
      max_errors: clampInt(options.max_errors ?? 50, 1, 500),
      timeout_ms: clampInt(options.timeout_ms ?? 1500, 10, 20000),
      return_ast: !!(options.return_ast ?? true),
      return_tokens: !!(options.return_tokens ?? false),
      return_formatted: !!(options.return_formatted ?? false)
    }
  };
}

export function makeResponseBase({ lang_id, mode }) {
  return {
    "@kind": "code.oracle.response.v1",
    "@v": 1,
    lang_id,
    mode,
    ok: false,
    score: { legality: 0, style: 1, completeness: 0, risk: 0 },
    ast: null,
    errors: [],
    warnings: [],
    formatted: null,
    metrics: {}
  };
}

function clampInt(x, lo, hi) {
  const n = Number(x);
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(n)));
}
```

## 3) src/languages/*.js — language mapping modules

### src/languages/js.js

```js
import JavaScript from "tree-sitter-javascript";
export function load() { return JavaScript; }
export const lang_id = "js";
```

### src/languages/ts.js

```js
import TypeScript from "tree-sitter-typescript";
export function load() { return TypeScript.typescript; }
export const lang_id = "ts";
```

### src/languages/py.js

```js
import Python from "tree-sitter-python";
export function load() { return Python; }
export const lang_id = "py";
```

### src/languages/go.js

```js
import Go from "tree-sitter-go";
export function load() { return Go; }
export const lang_id = "go";
```

### src/languages/rust.js

```js
import Rust from "tree-sitter-rust";
export function load() { return Rust; }
export const lang_id = "rust";
```

## 4) src/ast.js — compact AST extraction

```js
export function toCompactAST(tree, opts = {}) {
  const maxNodes = opts.maxNodes ?? 5000;
  let count = 0;

  function walk(node) {
    if (count++ > maxNodes) return { type: "TRUNCATED" };

    const out = {
      type: node.type,
      start: node.startPosition ? { row: node.startPosition.row, column: node.startPosition.column } : null,
      end: node.endPosition ? { row: node.endPosition.row, column: node.endPosition.column } : null
    };

    if (node.childCount && node.childCount > 0) {
      const kids = [];
      for (let i = 0; i < node.childCount; i++) {
        kids.push(walk(node.child(i)));
      }
      out.children = kids;
    }

    // Tree-sitter exposes named children/fields, but keeping generic for ABI v1
    return out;
  }

  return {
    "@kind": "ast.tree.v1",
    "@v": 1,
    root: walk(tree.rootNode),
    node_count: Math.min(count, maxNodes)
  };
}
```

## 5) src/errors.js — error extraction

Tree-sitter indicates errors via:

* `node.hasError()`
* nodes of type `"ERROR"`
* missing nodes

```js
export function collectErrors(tree, source, maxErrors = 50) {
  const errors = [];

  function pushErr(message, node, code = "E_PARSE") {
    if (errors.length >= maxErrors) return;
    errors.push({
      code,
      message,
      severity: "error",
      span: node?.startPosition ? {
        line: node.startPosition.row + 1,
        col: node.startPosition.column + 1,
        end_line: node.endPosition.row + 1,
        end_col: node.endPosition.column + 1
      } : null,
      hint: null,
      rule_id: null
    });
  }

  function walk(node) {
    if (errors.length >= maxErrors) return;

    if (node.type === "ERROR") {
      pushErr("Syntax error", node, "E_PARSE");
    }

    // Heuristic: missing nodes sometimes represent expected tokens
    if (node.isMissing) {
      pushErr("Missing token", node, "E_PARSE");
    }

    for (let i = 0; i < node.childCount; i++) walk(node.child(i));
  }

  walk(tree.rootNode);

  // Optional extra cheap heuristic: unbalanced braces/parens can be useful
  const brace = balanceHeuristic_(source);
  if (!brace.ok && errors.length < maxErrors) {
    errors.push({
      code: "E_PARSE",
      message: brace.message,
      severity: "error",
      span: null,
      hint: brace.hint,
      rule_id: "heuristic/balance"
    });
  }

  return errors;
}

function balanceHeuristic_(src) {
  const pairs = [
    ["{", "}"],
    ["(", ")"],
    ["[", "]"]
  ];
  for (const [o, c] of pairs) {
    const open = (src.match(new RegExp(escapeRe_(o), "g")) || []).length;
    const close = (src.match(new RegExp(escapeRe_(c), "g")) || []).length;
    if (open !== close) {
      return {
        ok: false,
        message: `Unbalanced ${o}${c}: open=${open} close=${close}`,
        hint: `Check for a missing '${open > close ? c : o}'`
      };
    }
  }
  return { ok: true };
}

function escapeRe_(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
```

## 6) src/score.js — deterministic scoring

```js
export function scoreResult({ errors, warnings }) {
  const errCount = errors?.length ?? 0;
  const warnCount = warnings?.length ?? 0;

  const fatal = errCount; // v1 treats all errors as fatal-ish

  const legality = clamp01(1 - (errCount * 0.10) - (fatal * 0.20));
  const style = clamp01(1 - (warnCount * 0.05));
  const completeness = errCount === 0 ? 0.95 : clamp01(0.25 - errCount * 0.05);

  // risk is optional; in v1 we leave 0 unless we detect obvious hazards
  const risk = 0;

  return { legality, style, completeness, risk };
}

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}
```

## 7) src/index.js — oracle runner (stdin or http)

```js
import http from "http";
import Parser from "tree-sitter";

import { normalizeRequest, makeResponseBase } from "./abi.js";
import { toCompactAST } from "./ast.js";
import { collectErrors } from "./errors.js";
import { scoreResult } from "./score.js";

import * as JS from "./languages/js.js";
import * as TS from "./languages/ts.js";
import * as PY from "./languages/py.js";
import * as GO from "./languages/go.js";
import * as RUST from "./languages/rust.js";

const LANGS = new Map([
  ["js", JS.load],
  ["ts", TS.load],
  ["py", PY.load],
  ["go", GO.load],
  ["rust", RUST.load]
]);

function runOracle(rawReq) {
  const req = normalizeRequest(rawReq);
  const res = makeResponseBase({ lang_id: req.lang_id, mode: req.mode });

  const loadLang = LANGS.get(req.lang_id);
  if (!loadLang) {
    res.errors = [{
      code: "E_LANG",
      message: `Unsupported lang_id: ${req.lang_id}`,
      severity: "error",
      span: null,
      hint: "Install a tree-sitter language module and register it.",
      rule_id: "oracle/lang"
    }];
    res.metrics = { supported: Array.from(LANGS.keys()) };
    return res;
  }

  const parser = new Parser();
  parser.setLanguage(loadLang());

  const t0 = Date.now();
  const tree = parser.parse(req.source);
  const parseMs = Date.now() - t0;

  const errors = collectErrors(tree, req.source, req.options.max_errors);
  const warnings = []; // v1: no lint rules yet

  res.ok = errors.length === 0;
  res.errors = errors;
  res.warnings = warnings;
  res.score = scoreResult({ errors, warnings });

  if (req.options.return_ast) {
    res.ast = toCompactAST(tree, { maxNodes: 8000 });
  }

  res.metrics = {
    parse_ms: parseMs,
    ast_nodes: res.ast?.node_count ?? 0,
    oracle: "tree-sitter",
    lang_id: req.lang_id
  };

  return res;
}

// --------------------------- CLI: stdin ---------------------------

async function runStdin() {
  const chunks = [];
  for await (const c of process.stdin) chunks.push(c);
  const input = Buffer.concat(chunks).toString("utf8").trim();
  const req = input ? JSON.parse(input) : {};
  const out = runOracle(req);
  process.stdout.write(JSON.stringify(out));
}

// --------------------------- HTTP mode ---------------------------

function runHttp(port) {
  const server = http.createServer(async (req, res) => {
    if (req.method !== "POST" || req.url !== "/oracle") {
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "not found" }));
      return;
    }

    let body = "";
    req.on("data", (c) => body += c);
    req.on("end", () => {
      try {
        const parsed = JSON.parse(body || "{}");
        const out = runOracle(parsed);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(out));
      } catch (e) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: String(e?.message || e) }));
      }
    });
  });

  server.listen(port, () => {
    console.log(`TREE_SITTER_ORACLE_ADAPTER_v1 listening on http://localhost:${port}/oracle`);
  });
}

// --------------------------- entry ---------------------------

const args = process.argv.slice(2);
if (args.includes("--stdin")) {
  runStdin();
} else if (args.includes("--http")) {
  const i = args.indexOf("--http");
  const port = Number(args[i + 1] || 8787);
  runHttp(port);
} else {
  // default: stdin
  runStdin();
}
```

## 8) README quickstart (copy/paste)

```bash
# 1) install
npm i

# 2) stdin mode
echo '{"@kind":"code.oracle.request.v1","@v":1,"lang_id":"js","mode":"parse","input":{"source":"let x = ;"}}' | npm run oracle:stdin

# 3) http mode
npm run oracle:http
# then POST to http://localhost:8787/oracle with the request json
```

## How they connect (end-to-end)

### Flow

1. **GAS** loads registry + pack info:

   * `asx.langpack.registry.v1`
   * `asx.langpack.loaded.v1`

2. Your Micronaut/CLI chooses `lang_id`, then calls the Node oracle:

   * POST `/oracle` with `code.oracle.request.v1`

3. The oracle returns:

   * `code.oracle.response.v1` (ok/score/errors/ast)

4. Your build system uses that response to:

   * accept/reject model output
   * constrain decode (later)
   * or enforce legality before writing to project sheets

## Next “obvious” upgrades (if you want them next)

* **TREE_SITTER_LANGPACK_BUILDER_v1**: auto-generate `grammar/ast/test` tabs from language modules
* **ORACLE_LINT_RULES_v1**: add sheet-backed lint rules (regex-free; AST pattern rules)
* **CONSTRAINED_DECODE_PLAN_v1**: token-level constraints using oracle feedback + incremental parsing

---

# SCX_CONTROL_ATOMS_v1

**Purpose**  
A **language-independent control & relation vocabulary**.  
Atoms represent *meaning*, never surface words.

## 1) Control / Logic Atoms Table

```json
{
  "@kind": "scx.control.atoms.v1",
  "@v": 1,
  "domain": "control_logic",
  "atoms": [
    { "id": "BRANCH", "arity": 3, "desc": "Conditional branching: condition, then, else" },
    { "id": "SELECT", "arity": 2, "desc": "Multi-branch selection (case/when)" },

    { "id": "ITERATE", "arity": 2, "desc": "Iteration over a collection or range" },
    { "id": "LOOP_WHILE", "arity": 2, "desc": "While loop: condition, body" },
    { "id": "LOOP_UNTIL", "arity": 2, "desc": "Until loop: condition, body" },
    { "id": "BREAK", "arity": 0, "desc": "Exit loop" },
    { "id": "CONTINUE", "arity": 0, "desc": "Next iteration" },

    { "id": "LOGICAL_AND", "arity": 2, "desc": "Logical conjunction" },
    { "id": "LOGICAL_OR", "arity": 2, "desc": "Logical disjunction" },
    { "id": "LOGICAL_NOT", "arity": 1, "desc": "Logical negation" },

    { "id": "COMPARE_EQ", "arity": 2, "desc": "Equality comparison" },
    { "id": "COMPARE_NEQ", "arity": 2, "desc": "Inequality comparison" },
    { "id": "COMPARE_GT", "arity": 2, "desc": "Greater-than comparison" },
    { "id": "COMPARE_GTE", "arity": 2, "desc": "Greater-than-or-equal comparison" },
    { "id": "COMPARE_LT", "arity": 2, "desc": "Less-than comparison" },
    { "id": "COMPARE_LTE", "arity": 2, "desc": "Less-than-or-equal comparison" },

    { "id": "ASSIGN", "arity": 2, "desc": "Assignment" },
    { "id": "DECLARE", "arity": 2, "desc": "Declaration with optional initializer" },

    { "id": "RETURN", "arity": 1, "desc": "Return value" },
    { "id": "NOOP", "arity": 0, "desc": "No operation" },

    { "id": "BOOL_TRUE", "arity": 0, "desc": "Boolean true" },
    { "id": "BOOL_FALSE", "arity": 0, "desc": "Boolean false" }
  ]
}
```

**Key invariant**

> These atoms are **final**. Languages map *to* them; they never change to fit languages.

## 2) SCX Canonical Form (example)

```json
{
  "@scx": "BRANCH",
  "if": {
    "@scx": "LOGICAL_AND",
    "a": { "@scx": "COMPARE_GT", "a": "x", "b": 3 },
    "b": { "@scx": "COMPARE_LT", "a": "x", "b": 10 }
  },
  "then": { "@scx": "RETURN", "value": "ok" },
  "else": { "@scx": "RETURN", "value": "fail" }
}
```

This structure is **the truth**.

## SCX → JS Renderer (v1)

```js
export function renderJS(node) {
  const r = renderJS;
  switch (node["@scx"]) {
    case "BRANCH":
      return `if (${r(node.if)}) { ${r(node.then)} } else { ${r(node.else)} }`;

    case "LOGICAL_AND":
      return `(${r(node.a)} && ${r(node.b)})`;

    case "COMPARE_GT":
      return `(${r(node.a)} > ${r(node.b)})`;

    case "COMPARE_LT":
      return `(${r(node.a)} < ${r(node.b)})`;

    case "RETURN":
      return `return ${r(node.value)};`;

    case "BOOL_TRUE": return "true";
    case "BOOL_FALSE": return "false";

    default:
      return typeof node === "string" ? node : JSON.stringify(node);
  }
}
```

**Output**

```js
if ((x > 3 && x < 10)) { return ok; } else { return fail; }
```

## SCX → Python Renderer (v1)

```py
def render_py(node):
    r = render_py
    t = node.get("@scx")

    if t == "BRANCH":
        return f"if {r(node['if'])}:\n    {r(node['then'])}\nelse:\n    {r(node['else'])}"

    if t == "LOGICAL_AND":
        return f"({r(node['a'])} and {r(node['b'])})"

    if t == "COMPARE_GT":
        return f"({r(node['a'])} > {r(node['b'])})"

    if t == "COMPARE_LT":
        return f"({r(node['a'])} < {r(node['b'])})"

    if t == "RETURN":
        return f"return {r(node['value'])}"

    if t == "BOOL_TRUE": return "True"
    if t == "BOOL_FALSE": return "False"

    return str(node)
```

**Output**

```py
if (x > 3 and x < 10):
    return ok
else:
    return fail
```

## SCX → SQL-IDB Renderer (CASE/WHERE style)

This targets **IndexedDB-style SQL / KQL / GAS-friendly query logic**.

```js
export function renderSQL(node) {
  const r = renderSQL;
  switch (node["@scx"]) {
    case "BRANCH":
      return `CASE WHEN ${r(node.if)} THEN ${r(node.then)} ELSE ${r(node.else)} END`;

    case "LOGICAL_AND":
      return `(${r(node.a)} AND ${r(node.b)})`;

    case "COMPARE_GT":
      return `(${r(node.a)} > ${r(node.b)})`;

    case "COMPARE_LT":
      return `(${r(node.a)} < ${r(node.b)})`;

    case "RETURN":
      return r(node.value);

    case "BOOL_TRUE": return "TRUE";
    case "BOOL_FALSE": return "FALSE";

    default:
      return typeof node === "string" ? node : JSON.stringify(node);
  }
}
```

**Output**

```sql
CASE
  WHEN (x > 3 AND x < 10)
  THEN ok
  ELSE fail
END
```

## 3) Why this locks the system

* **SCX atoms are finite** → no prompt drift
* **Renderers are pure functions** → deterministic
* **GAS can execute semantics** → no model weights needed
* **LANGPACK only skins output** → replaceable forever
* **Tree-sitter oracles validate surfaces** → legality proof

## 4) The architectural punchline

> Other systems tokenize **words**.  
> You tokenize **causality**.

That’s why SCX works in:

* GAS
* browser
* CLI
* SQL / IDB
* UI
* and *after* model inference

---

```json
{
  "@kind": "scx.arithmetic.atoms.v1",
  "@v": 1,
  "domain": "arithmetic_numeric",
  "atoms": [
    { "id": "NUM", "arity": 1, "desc": "Numeric literal wrapper (value)" },

    { "id": "ADD", "arity": 2, "desc": "Addition" },
    { "id": "SUB", "arity": 2, "desc": "Subtraction" },
    { "id": "MUL", "arity": 2, "desc": "Multiplication" },
    { "id": "DIV", "arity": 2, "desc": "Division" },
    { "id": "MOD", "arity": 2, "desc": "Modulo / remainder" },

    { "id": "POW", "arity": 2, "desc": "Exponentiation" },
    { "id": "NEG", "arity": 1, "desc": "Unary negation" },
    { "id": "ABS", "arity": 1, "desc": "Absolute value" },

    { "id": "MIN", "arity": 2, "desc": "Minimum of two values (liftable to N-ary via fold)" },
    { "id": "MAX", "arity": 2, "desc": "Maximum of two values (liftable to N-ary via fold)" },

    { "id": "FLOOR", "arity": 1, "desc": "Floor" },
    { "id": "CEIL", "arity": 1, "desc": "Ceiling" },
    { "id": "ROUND", "arity": 1, "desc": "Round to nearest integer (implementation-defined ties; optional opts)" },

    { "id": "CLAMP", "arity": 3, "desc": "Clamp value into [min,max]: value, min, max" },

    { "id": "RANGE", "arity": 3, "desc": "Range spec (start, end, step). End semantics are renderer-defined but must be explicit." },
    { "id": "IN_RANGE", "arity": 3, "desc": "Membership test in range: value, range_start, range_end (step optional via RANGE)" }
  ],
  "invariants": [
    "Atoms encode meaning, not surface symbols.",
    "All arithmetic nodes must be total: DIV by zero must produce a deterministic error node or a declared sentinel policy.",
    "RANGE end semantics MUST be chosen by renderer profile (inclusive/exclusive) and encoded in options if ambiguous.",
    "N-ary arithmetic is expressed via left-fold (e.g., ADD(ADD(a,b),c)) unless a future NARY_* atom is added in a MAJOR bump.",
    "NUM is optional if your runtime already distinguishes literals; keep it for strict typing/proofs."
  ]
}
```

## Minimal SCX canonical examples

### 1) `((a + b) * 3) % 10`

```json
{
  "@scx": "MOD",
  "a": {
    "@scx": "MUL",
    "a": {
      "@scx": "ADD",
      "a": "a",
      "b": "b"
    },
    "b": { "@scx": "NUM", "value": 3 }
  },
  "b": { "@scx": "NUM", "value": 10 }
}
```

### 2) `x in [0, 100]` (range check)

```json
{
  "@scx": "IN_RANGE",
  "value": "x",
  "a": { "@scx": "NUM", "value": 0 },
  "b": { "@scx": "NUM", "value": 100 }
}
```

### 3) `range(0, 10, 2)`

```json
{
  "@scx": "RANGE",
  "start": { "@scx": "NUM", "value": 0 },
  "end": { "@scx": "NUM", "value": 10 },
  "step": { "@scx": "NUM", "value": 2 },
  "opts": { "end": "exclusive" }
}
```
