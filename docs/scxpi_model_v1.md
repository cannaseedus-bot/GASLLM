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
