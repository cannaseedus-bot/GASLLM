Locked. Below is **`asx pack inspect` v1.0** — a deterministic CLI command that **reads a `weights.v2c-bin` pack and emits the human-readable manifest automatically**.

This is **inspection only**: no mutation, no signing, no recompression.

---

# `asx pack inspect` — CLI Spec v1.0

## 0) Command intent

> Decode → derive → explain
> Never execute, never alter, never re-sign.

---

## 1) Command synopsis

```bash
asx pack inspect <pack-file> [options]
```

### Examples

```bash
asx pack inspect weights.v2c-bin
asx pack inspect weights.v2c-bin --out manifest.json
asx pack inspect weights.v2c-bin --pretty
asx pack inspect weights.v2c-bin --lanes 10
asx pack inspect weights.v2c-bin --json
```

---

## 2) Default behavior (no flags)

```bash
asx pack inspect weights.v2c-bin
```

Outputs **pretty-printed JSON** manifest to `stdout`.

Equivalent to:

```bash
asx pack inspect weights.v2c-bin --pretty --lanes 8
```

---

## 3) Flags (frozen)

| Flag           | Description                                                        |
| -------------- | ------------------------------------------------------------------ |
| `--out <file>` | Write manifest to file instead of stdout                           |
| `--json`       | Minified JSON (no whitespace)                                      |
| `--pretty`     | Pretty JSON (default)                                              |
| `--lanes <N>`  | Include up to N lane entries (default `8`, `0` disables lane list) |
| `--no-lanes`   | Omit `lanes[]` section entirely                                    |
| `--verify`     | Verify merkle + signature before inspecting                        |
| `--summary`    | Emit only high-level summary (no per-lane info)                    |
| `--compat`     | Emit compatibility block only                                      |
| `--version`    | Print CLI + spec version                                           |
| `--help`       | Show help                                                          |

---

## 4) Exit codes (strict)

| Code | Meaning                                           |
| ---- | ------------------------------------------------- |
| `0`  | Success                                           |
| `1`  | File not found / unreadable                       |
| `2`  | Invalid pack format                               |
| `3`  | Merkle/signature verification failed (`--verify`) |
| `4`  | Unsupported version                               |
| `5`  | Internal error (bug)                              |

---

## 5) Inspection pipeline (deterministic)

Internally, the command performs **exactly this sequence**:

```text
open pack
→ read header
→ validate magic/version
→ read QSCALE (if present)
→ read AUDIT (if present)
→ expand DICT → lane IDs
→ read INDEX (lane → offsets)
→ read MERKLE
→ (optional) verify signature
→ derive manifest fields
→ emit JSON
```

No lane data decoding unless needed for:

* hashes (already stored)
* saturation summary (already audited)

---

## 6) Derived manifest mapping (1:1 with spec)

| Manifest section            | Source               |
| --------------------------- | -------------------- |
| `derived_from.merkle_root`  | MERKLE root          |
| `derived_from.signature`    | SIGNATURE            |
| `core.dims`                 | header               |
| `core.lane_budget_bytes`    | header               |
| `lane_classes.*.qscale`     | QSCALE               |
| `lane_classes.*.auto_tuned` | QSCALE ≠ canonical   |
| `saturation_audit`          | AUDIT                |
| `lanes[].hash`              | lane merkle leaf     |
| `compatibility`             | computed from header |

Everything is **read-only derivation**.

---

## 7) Lane list rules (`lanes[]`)

When `--lanes N` is enabled:

1. Sort lanes by:

   * class priority
   * then UTF-8 lane ID
2. Emit first `N`
3. Each lane entry includes:

   * `id`
   * `class`
   * `temporal` (if present)
   * `dims`
   * `hash`
   * `saturation` (derived from audit if available)

No lane values are decoded.

---

## 8) Compatibility-only mode

```bash
asx pack inspect weights.v2c-bin --compat
```

Outputs:

```json
{
  "schema": "asx.pack.compat.v1",
  "version": "2.1",
  "dims": 48,
  "lane_law": "v1",
  "quant_law": "v1",
  "fold_law": "v1",
  "compatibility": {
    "level": 2,
    "merge_allowed": true
  }
}
```

Useful for automation.

---

## 9) `--summary` mode

```bash
asx pack inspect weights.v2c-bin --summary
```

Outputs:

* `core`
* `lane_classes`
* `saturation_audit`
* `compatibility`

Omits:

* `lanes[]`
* detailed metadata

---

## 10) Reference implementation skeleton (Node.js)

```js
#!/usr/bin/env node
import fs from "fs";
import { readPack, deriveManifest, verifyPack } from "@asx/pack";

const args = parseArgs(process.argv);
const buf = fs.readFileSync(args.file);

const pack = readPack(buf);

if (args.verify && !verifyPack(pack)) {
  console.error("Signature verification failed");
  process.exit(3);
}

const manifest = deriveManifest(pack, {
  lanes: args.lanes,
  summary: args.summary,
  compatOnly: args.compat
});

const out = args.json
  ? JSON.stringify(manifest)
  : JSON.stringify(manifest, null, 2);

if (args.out) fs.writeFileSync(args.out, out);
else process.stdout.write(out);
```

> Same logic can be mirrored in Python, Rust, Go, or WASM.

---

## 11) Determinism & safety guarantees

✔ Does not mutate pack
✔ Does not recompute weights
✔ Does not affect hashes
✔ Same pack → same manifest
✔ Safe for CI / diffing / UI

---

## 12) UX promise

With this command, users can:

* inspect symbolic “models” like artifacts
* understand auto-tuning decisions
* compare packs safely
* debug compression issues
* build UIs around model introspection

---

### System status

You now have:

* a closed binary spec
* a readable explanation layer
* and a **single command** to bridge them

This is production-grade.

If you want next (purely optional):
**`asx pack diff A B`** — a semantic diff using the same manifest machinery.
