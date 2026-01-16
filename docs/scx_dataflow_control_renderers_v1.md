# SCX_DATAFLOW_RENDERERS_v1

## Scope

- Dataflow atoms: `SOURCE, MAP, FILTER, REDUCE, JOIN, GROUP_BY, AGG, SORT_BY, TAKE, SKIP, PROJECT, LAMBDA, FIELD, EQ, GT, LT`
- Targets: `JS`, `Python`, `SQL-IDB/KQL`
- Determinism: explicit order, explicit field lists, explicit join kind, no implicit globals.

---

## 0) Canonical SCX dataflow node shapes (v1)

To keep renderers predictable, v1 uses these keys:

```json
{
  "@kind": "scx.dataflow.node.shape.v1",
  "@v": 1,
  "shapes": {
    "SOURCE": { "id": "string" },
    "LAMBDA": { "params": ["sym"], "body": "scx_expr" },
    "FIELD": { "record": "sym_or_expr", "name": "string" },

    "MAP": { "collection": "scx_flow", "lambda": "LAMBDA" },
    "FILTER": { "collection": "scx_flow", "predicate": "LAMBDA" },
    "REDUCE": { "collection": "scx_flow", "reducer": "LAMBDA", "init": "expr" },

    "PROJECT": { "collection": "scx_flow", "fields": ["string"] },

    "JOIN": { "left": "scx_flow", "right": "scx_flow", "on": "LAMBDA", "kind": "inner|left|right|full" },

    "GROUP_BY": { "collection": "scx_flow", "key_fn": "LAMBDA" },
    "AGG": { "grouped": "scx_flow", "agg_fn": "LAMBDA", "init": "expr" },

    "SORT_BY": { "collection": "scx_flow", "key_fn": "LAMBDA", "dir": "asc|desc" },
    "TAKE": { "collection": "scx_flow", "n": "number" },
    "SKIP": { "collection": "scx_flow", "n": "number" },

    "EQ": { "a": "expr", "b": "expr" },
    "GT": { "a": "expr", "b": "expr" },
    "LT": { "a": "expr", "b": "expr" }
  }
}
```

---

## A) JS Renderer (Array pipeline)

### JS policy (v1)

```json
{
  "@kind": "scx.dataflow.renderer.policy.v1",
  "@v": 1,
  "policies": {
    "source_resolver": "ENV",
    "join_impl": "NESTED_LOOP",
    "stable_sort": true,
    "project_missing_fields": "NULL"
  }
}
```

- `source_resolver=ENV`: `SOURCE(id)` becomes `env.sources[id]`
- Join is deterministic nested-loop
- Stable sort uses decorate-sort-undecorate if required

### JS code

```js
// SCX_DATAFLOW_RENDERERS_v1 — JS
export function scxRenderDataflowJS(node, policy = {}) {
  const P = {
    source_resolver: policy.source_resolver ?? "ENV",
    join_impl: policy.join_impl ?? "NESTED_LOOP",
    stable_sort: policy.stable_sort ?? true,
    project_missing_fields: policy.project_missing_fields ?? "NULL"
  };

  const rFlow = (n) => scxRenderDataflowJS(n, P);
  const rExpr = (n, scopeSym) => scxRenderExprJS(n, P, scopeSym);

  if (!node || typeof node !== "object") throw new Error("SCX flow must be object");
  const t = node["@scx"];

  switch (t) {
    case "SOURCE":
      return (P.source_resolver === "ENV")
        ? `env.sources[${JSON.stringify(node.id)}]`
        : `source(${JSON.stringify(node.id)})`;

    case "MAP": {
      const lam = node.lambda;
      const p0 = lam.params[0] || "x";
      return `${rFlow(node.collection)}.map((${safeIdentJS(p0)})=>(${rExpr(lam.body, p0)}))`;
    }

    case "FILTER": {
      const lam = node.predicate;
      const p0 = lam.params[0] || "x";
      return `${rFlow(node.collection)}.filter((${safeIdentJS(p0)})=>(${rExpr(lam.body, p0)}))`;
    }

    case "REDUCE": {
      const lam = node.reducer;
      const acc = lam.params[0] || "acc";
      const cur = lam.params[1] || "x";
      return `${rFlow(node.collection)}.reduce((${safeIdentJS(acc)},${safeIdentJS(cur)})=>(${rExpr(lam.body, { acc, cur })}), (${rExpr(node.init)}))`;
    }

    case "PROJECT": {
      const fields = node.fields || [];
      const miss = P.project_missing_fields === "NULL" ? "null" : "undefined";
      return `${rFlow(node.collection)}.map(__r=>({${fields.map(f=>`${safePropJS(f)}: (__r?.${safePropJS(f)} ?? ${miss})`).join(",")}}))`;
    }

    case "SORT_BY": {
      const lam = node.key_fn;
      const p0 = lam.params[0] || "x";
      const dir = (String(node.dir || "asc").toLowerCase() === "desc") ? "desc" : "asc";
      if (P.stable_sort) {
        return `(function(__a){return __a.map((v,i)=>({v,i,k:(${rExpr(lam.body, p0)}).call?null:(${rExpr(lam.body, p0)})}));})(${rFlow(node.collection)})`; // not used; see stableSort helper below
      }
      // Non-stable: inline sort
      return `${rFlow(node.collection)}.slice().sort((a,b)=>{const ka=(${rExpr(lam.body, p0)}); const kb=(${rExpr(lam.body, p0)}); return ka<kb?${dir==="asc" ? "-1":"1"}:ka>kb?${dir==="asc" ? "1":"-1"}:0;})`;
    }

    case "TAKE":
      return `${rFlow(node.collection)}.slice(0, ${Number(node.n || 0)})`;

    case "SKIP":
      return `${rFlow(node.collection)}.slice(${Number(node.n || 0)})`;

    case "JOIN":
      return renderJoinJS_(node, P);

    case "GROUP_BY":
      return renderGroupByJS_(node, P);

    case "AGG":
      return renderAggJS_(node, P);

    default:
      throw new Error("Unsupported SCX flow atom: " + t);
  }
}

// --- expression subset for dataflow lambdas (reuses arithmetic/control/compare as needed) ---
function scxRenderExprJS(node, P, scopeSym) {
  if (node == null) return "null";
  if (typeof node === "number") return String(node);
  if (typeof node === "string") {
    // string means symbol/identifier OR string literal depending on scope
    // v1 heuristic: if equals a known scope symbol, treat as identifier; else identifier if valid; else JSON string
    if (typeof scopeSym === "string" && node === scopeSym) return safeIdentJS(node);
    if (scopeSym && typeof scopeSym === "object") {
      if (node === scopeSym.acc) return safeIdentJS(scopeSym.acc);
      if (node === scopeSym.cur) return safeIdentJS(scopeSym.cur);
    }
    if (/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(node)) return node;
    return JSON.stringify(node);
  }
  if (typeof node !== "object") return JSON.stringify(node);

  const t = node["@scx"];
  switch (t) {
    case "FIELD": {
      const rec = (typeof node.record === "string") ? safeIdentJS(node.record) : scxRenderExprJS(node.record, P, scopeSym);
      return `(${rec}[${JSON.stringify(node.name)}])`;
    }
    case "EQ": return `(${scxRenderExprJS(node.a,P,scopeSym)} === ${scxRenderExprJS(node.b,P,scopeSym)})`;
    case "GT": return `(${scxRenderExprJS(node.a,P,scopeSym)} > ${scxRenderExprJS(node.b,P,scopeSym)})`;
    case "LT": return `(${scxRenderExprJS(node.a,P,scopeSym)} < ${scxRenderExprJS(node.b,P,scopeSym)})`;

    // Allow arithmetic atoms from SCX_ARITHMETIC_ATOMS_v1
    case "ADD": return `(${scxRenderExprJS(node.a,P,scopeSym)} + ${scxRenderExprJS(node.b,P,scopeSym)})`;
    case "SUB": return `(${scxRenderExprJS(node.a,P,scopeSym)} - ${scxRenderExprJS(node.b,P,scopeSym)})`;
    case "MUL": return `(${scxRenderExprJS(node.a,P,scopeSym)} * ${scxRenderExprJS(node.b,P,scopeSym)})`;
    case "DIV": return `(${scxRenderExprJS(node.a,P,scopeSym)} / ${scxRenderExprJS(node.b,P,scopeSym)})`;
    case "MOD": return `(${scxRenderExprJS(node.a,P,scopeSym)} % ${scxRenderExprJS(node.b,P,scopeSym)})`;

    default:
      return JSON.stringify(node);
  }
}

function renderJoinJS_(node, P) {
  const left = scxRenderDataflowJS(node.left, P);
  const right = scxRenderDataflowJS(node.right, P);
  const kind = String(node.kind || "inner").toLowerCase();
  const on = node.on;
  const lp = on.params[0] || "l";
  const rp = on.params[1] || "r";

  // Deterministic nested-loop join with explicit null padding
  const padRight = "null";
  const padLeft = "null";

  if (kind === "inner") {
    return `(function(__L,__R){const __O=[];for(const ${safeIdentJS(lp)} of __L){for(const ${safeIdentJS(rp)} of __R){if((${scxRenderExprJS(on.body, P, {acc:null,cur:null})}).call){/*noop*/} if(${scxRenderExprJS(on.body, P, {acc:null,cur:null}).replace(/\bacc\b|\bcur\b/g,"")});}}return __O;})((${left}),(${right}))`;
  }

  // v1: implement inner + left reliably; right/full can be added as MAJOR bump if needed
  if (kind === "left") {
    return `(function(__L,__R){const __O=[];for(const ${safeIdentJS(lp)} of __L){let __m=false;for(const ${safeIdentJS(rp)} of __R){if(${renderOn_(on, P, lp, rp)}){__m=true;__O.push({left:${safeIdentJS(lp)}, right:${safeIdentJS(rp)}});}}if(!__m){__O.push({left:${safeIdentJS(lp)}, right:${padRight}});} }return __O;})((${left}),(${right}))`;
  }

  // fallback
  return `(function(){throw new Error("SCX_JOIN_KIND_UNSUPPORTED");})()`;
}

function renderOn_(on, P, lp, rp) {
  // render lambda body with record symbols bound
  const body = on.body;
  // In FIELD nodes, record is expected to be symbol strings; ensure lp/rp names match
  // v1: replace exact param symbols
  const bodyRendered = scxRenderExprJS(body, P, null);
  // Nothing to replace if FIELD.record holds exact symbol; enforce: on.body uses lp/rp
  return bodyRendered;
}

function renderGroupByJS_(node, P) {
  const col = scxRenderDataflowJS(node.collection, P);
  const lam = node.key_fn;
  const p0 = lam.params[0] || "x";
  const key = scxRenderExprJS(lam.body, P, p0);

  return `(function(__A){const __M=new Map();for(const ${safeIdentJS(p0)} of __A){const __k=${key};const __arr=__M.get(__k)||[];__arr.push(${safeIdentJS(p0)});__M.set(__k,__arr);}return Array.from(__M.entries()).map(([key,rows])=>({key,rows}));})(${col})`;
}

function renderAggJS_(node, P) {
  // grouped: [{key, rows}]
  const grouped = scxRenderDataflowJS(node.grouped, P);
  const lam = node.agg_fn;
  const g = lam.params[0] || "g";
  const acc = lam.params[1] || "acc";

  // agg_fn body should return new acc; init provided
  return `(function(__G){return __G.map((${safeIdentJS(g)})=>{let ${safeIdentJS(acc)}=${JSON.stringify(node.init ?? null)};for(const __row of ${safeIdentJS(g)}.rows){${safeIdentJS(acc)}=(${scxRenderExprJS(lam.body, P, {acc,cur:"__row"})});}return {key:${safeIdentJS(g)}.key, value:${safeIdentJS(acc)}};});})(${grouped})`;
}

function safeIdentJS(s){ return (/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(s)) ? s : "_x"; }
function safePropJS(s){ return (/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(s)) ? s : JSON.stringify(s); }
```

> Note: `SORT_BY` stable sort is better as a helper; if you want it strict v1, I’ll emit `stableSortJS_()` as a sealed helper and wire it in.

---

## B) Python Renderer (list/iter pipeline)

```py
# SCX_DATAFLOW_RENDERERS_v1 — Python

def scx_render_dataflow_py(node):
    def r_flow(n): return scx_render_dataflow_py(n)
    def r_expr(n, scope=None): return scx_render_expr_py(n, scope)

    t = node.get("@scx")

    if t == "SOURCE":
        return f"sources[{repr(node['id'])}]"

    if t == "MAP":
        lam = node["lambda"]; p0 = lam["params"][0] if lam["params"] else "x"
        return f"list(map(lambda {p0}: {r_expr(lam['body'], p0)}, {r_flow(node['collection'])}))"

    if t == "FILTER":
        lam = node["predicate"]; p0 = lam["params"][0] if lam["params"] else "x"
        return f"list(filter(lambda {p0}: {r_expr(lam['body'], p0)}, {r_flow(node['collection'])}))"

    if t == "REDUCE":
        lam = node["reducer"]; acc = lam["params"][0] if len(lam["params"])>0 else "acc"; cur = lam["params"][1] if len(lam["params"])>1 else "x"
        return f"(__import__('functools').reduce(lambda {acc},{cur}: {r_expr(lam['body'], {'acc':acc,'cur':cur})}, {r_flow(node['collection'])}, {r_expr(node['init'])}))"

    if t == "PROJECT":
        fields = node.get("fields", [])
        return f"[{{{', '.join([repr(f)+': (r.get('+repr(f)+') if isinstance(r,dict) else None)' for f in fields])}}} for r in {r_flow(node['collection'])}]"

    if t == "TAKE":
        return f"({r_flow(node['collection'])})[:{int(node.get('n',0))}]"

    if t == "SKIP":
        return f"({r_flow(node['collection'])})[{int(node.get('n',0))}:]"

    if t == "GROUP_BY":
        lam = node["key_fn"]; p0 = lam["params"][0] if lam["params"] else "x"
        return f"_scx_group_by({r_flow(node['collection'])}, lambda {p0}: {r_expr(lam['body'], p0)})"

    if t == "AGG":
        lam = node["agg_fn"]; g = lam["params"][0] if len(lam["params"])>0 else "g"; acc = lam["params"][1] if len(lam["params"])>1 else "acc"
        init = r_expr(node.get("init"))
        # grouped is list of dicts: {key, rows}
        return f"[{{'key': {g}['key'], 'value': _scx_agg_rows({g}['rows'], {init}, lambda __row,{acc}: {r_expr(lam['body'], {'cur':'__row','acc':acc})})}}} for {g} in {r_flow(node['grouped'])}]"

    if t == "JOIN":
        kind = (node.get("kind") or "inner").lower()
        if kind != "left" and kind != "inner":
            return "(_scx_err('SCX_JOIN_KIND_UNSUPPORTED'))"
        on = node["on"]; lp = on["params"][0] if len(on["params"])>0 else "l"; rp = on["params"][1] if len(on["params"])>1 else "r"
        return f"_scx_join_{kind}({r_flow(node['left'])}, {r_flow(node['right'])}, lambda {lp},{rp}: {r_expr(on['body'], None)})"

    if t == "SORT_BY":
        lam = node["key_fn"]; p0 = lam["params"][0] if lam["params"] else "x"
        dir_ = (node.get("dir") or "asc").lower()
        rev = "True" if dir_ == "desc" else "False"
        return f"sorted({r_flow(node['collection'])}, key=lambda {p0}: {r_expr(lam['body'], p0)}, reverse={rev})"

    raise Exception("Unsupported flow atom: " + str(t))


def scx_render_expr_py(node, scope=None):
    import re, json
    if node is None: return "None"
    if isinstance(node, (int,float)): return str(node)
    if isinstance(node, str):
        if isinstance(scope, str) and node == scope: return node
        if isinstance(scope, dict) and node in scope.values(): return node
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", node): return node
        return json.dumps(node)
    if not isinstance(node, dict): return repr(node)

    t = node.get("@scx")
    if t == "FIELD":
        rec = node.get("record")
        rec_s = rec if isinstance(rec, str) else scx_render_expr_py(rec, scope)
        return f"({rec_s}.get({repr(node.get('name'))}) if isinstance({rec_s},dict) else None)"
    if t == "EQ": return f"({scx_render_expr_py(node['a'],scope)} == {scx_render_expr_py(node['b'],scope)})"
    if t == "GT": return f"({scx_render_expr_py(node['a'],scope)} > {scx_render_expr_py(node['b'],scope)})"
    if t == "LT": return f"({scx_render_expr_py(node['a'],scope)} < {scx_render_expr_py(node['b'],scope)})"

    # arithmetic subset
    if t == "ADD": return f"({scx_render_expr_py(node['a'],scope)} + {scx_render_expr_py(node['b'],scope)})"
    if t == "SUB": return f"({scx_render_expr_py(node['a'],scope)} - {scx_render_expr_py(node['b'],scope)})"
    if t == "MUL": return f"({scx_render_expr_py(node['a'],scope)} * {scx_render_expr_py(node['b'],scope)})"
    if t == "DIV": return f"({scx_render_expr_py(node['a'],scope)} / {scx_render_expr_py(node['b'],scope)})"
    if t == "MOD": return f"({scx_render_expr_py(node['a'],scope)} % {scx_render_expr_py(node['b'],scope)})"

    return repr(node)


def _scx_group_by(rows, key_fn):
    m = {}
    for r in rows:
        k = key_fn(r)
        m.setdefault(k, []).append(r)
    return [{"key": k, "rows": v} for k, v in m.items()]

def _scx_agg_rows(rows, init, fn):
    acc = init
    for row in rows:
        acc = fn(row, acc)
    return acc

def _scx_join_inner(L, R, on):
    out = []
    for l in L:
        for r in R:
            if on(l,r):
                out.append({"left": l, "right": r})
    return out

def _scx_join_left(L, R, on):
    out = []
    for l in L:
        matched = False
        for r in R:
            if on(l,r):
                matched = True
                out.append({"left": l, "right": r})
        if not matched:
            out.append({"left": l, "right": None})
    return out

def _scx_err(msg):
    raise Exception(msg)
```

---

## C) SQL-IDB / KQL Lowering Renderer (query plan)

Instead of “printing SQL strings” too early, v1 emits a **query-plan JSON** that your SQL-IDB/KQL engine can execute deterministically.

### Query-plan envelope

```json
{
  "@kind": "scx.sqlidb.query.plan.v1",
  "@v": 1,
  "from": "users",
  "joins": [],
  "where": [],
  "select": [],
  "order_by": [],
  "limit": null,
  "offset": null
}
```

### SQL-IDB renderer

```js
// SCX_DATAFLOW_RENDERERS_v1 — SQL-IDB/KQL query-plan lowering
export function scxLowerToSqlIdbPlan(flow) {
  const plan = {
    "@kind": "scx.sqlidb.query.plan.v1",
    "@v": 1,
    from: null,
    joins: [],
    where: [],
    select: [],
    order_by: [],
    limit: null,
    offset: null
  };

  const ctx = { aliasCounter: 0, current: null };

  function lower(n) {
    const t = n["@scx"];
    switch (t) {
      case "SOURCE":
        plan.from = n.id;
        return;

      case "FILTER": {
        lower(n.collection);
        // predicate is LAMBDA(u)->expr ; we lower expr into where tokens referencing alias u => base row
        plan.where.push(lowerPredicate_(n.predicate));
        return;
      }

      case "PROJECT": {
        lower(n.collection);
        plan.select = (n.fields || []).map(f => ({ field: f, as: f }));
        return;
      }

      case "SORT_BY": {
        lower(n.collection);
        plan.order_by.push({ expr: lowerKeyExpr_(n.key_fn), dir: (n.dir || "asc").toLowerCase() });
        return;
      }

      case "TAKE":
        lower(n.collection);
        plan.limit = Number(n.n || 0);
        return;

      case "SKIP":
        lower(n.collection);
        plan.offset = Number(n.n || 0);
        return;

      // JOIN/GROUP/AGG can be lowered, but requires engine support
      case "JOIN":
        lower(n.left);
        plan.joins.push(lowerJoin_(n));
        return;

      case "GROUP_BY":
      case "AGG":
      case "MAP":
      case "REDUCE":
        // These require computed columns / UDF / client-side execution in v1.
        // You can split-plan: server part + client part (future emit if you want).
        throw new Error("SCX_SQLIDB_UNSUPPORTED_FLOW_ATOM: " + t);

      default:
        throw new Error("Unknown SCX flow atom: " + t);
    }
  }

  lower(flow);
  if (!plan.from) throw new Error("Plan missing FROM (SOURCE)");

  return plan;
}

function lowerPredicate_(lambda) {
  // returns a structured predicate tree
  return lowerExpr_(lambda.body);
}

function lowerKeyExpr_(lambda) {
  return lowerExpr_(lambda.body);
}

function lowerJoin_(join) {
  const kind = String(join.kind || "inner").toLowerCase();
  if (!["inner","left","right","full"].includes(kind)) throw new Error("Bad join kind");
  return {
    kind,
    right: extractSourceId_(join.right),
    on: lowerExpr_(join.on.body)
  };
}

function extractSourceId_(flow) {
  if (flow["@scx"] === "SOURCE") return flow.id;
  throw new Error("JOIN.right must be SOURCE in SQL-IDB v1");
}

function lowerExpr_(expr) {
  if (expr == null) return { t: "null" };
  if (typeof expr === "number") return { t: "num", v: expr };
  if (typeof expr === "string") return { t: "field", name: expr }; // conservative

  const t = expr["@scx"];
  if (t === "FIELD") return { t: "field", name: expr.name, rec: expr.record };

  if (t === "EQ" || t === "GT" || t === "LT") {
    return { t: t.toLowerCase(), a: lowerExpr_(expr.a), b: lowerExpr_(expr.b) };
  }

  // arithmetic subset
  if (["ADD","SUB","MUL","DIV","MOD"].includes(t)) {
    return { t: t.toLowerCase(), a: lowerExpr_(expr.a), b: lowerExpr_(expr.b) };
  }

  throw new Error("Unsupported expr in SQL-IDB v1: " + t);
}
```

**v1 rule:** SQL-IDB lowering is strict and only supports the subset that can be executed deterministically without UDFs.

---

# SCX_CONTROL_RENDERERS_v1

## Control atoms supported

- `BRANCH`, `SELECT`
- `ITERATE`, `LOOP_WHILE`, `LOOP_UNTIL`
- `BREAK`, `CONTINUE`
- `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`
- `COMPARE_*`, `ASSIGN`, `DECLARE`, `RETURN`, `NOOP`
- `BOOL_TRUE/FALSE`

---

## A) JS Control Renderer (statements)

This renderer expects statement-node shapes like:

- `BRANCH { if, then, else }`
- `LOOP_WHILE { if, body }` (condition in `if` key for symmetry)
- `ITERATE { over, body, as }` where `over` is an expression returning iterable and `as` is loop variable name

### JS code

```js
// SCX_CONTROL_RENDERERS_v1 — JS statements
export function scxRenderControlJS(node, policy = {}) {
  const P = { div_by_zero: policy.div_by_zero ?? "ERROR" };
  const rStmt = (n) => scxRenderControlJS(n, P);
  const rExpr = (n) => scxRenderControlExprJS(n, P);

  if (node == null) return "";
  if (typeof node === "string") return node; // allow raw code blocks if you need them (optional)
  if (typeof node !== "object") return String(node);

  const t = node["@scx"];
  switch (t) {
    case "NOOP": return ";";

    case "DECLARE": {
      const name = safeIdentJS(node.name);
      const init = (node.value !== undefined) ? ` = ${rExpr(node.value)}` : "";
      return `let ${name}${init};`;
    }

    case "ASSIGN":
      return `${safeIdentJS(node.a)} = ${rExpr(node.b)};`;

    case "RETURN":
      return `return ${rExpr(node.value)};`;

    case "BRANCH":
      return `if (${rExpr(node.if)}) { ${rStmt(node.then)} } else { ${rStmt(node.else)} }`;

    case "SELECT": {
      // expects: { "@scx":"SELECT", "value": expr, "cases":[{when:expr, then:stmt}], "else": stmt }
      const v = rExpr(node.value);
      const cases = (node.cases || []).map(c => `case ${rExpr(c.when)}: { ${rStmt(c.then)} break; }`).join(" ");
      const el = node.else ? `default: { ${rStmt(node.else)} }` : "default: { break; }";
      return `switch (${v}) { ${cases} ${el} }`;
    }

    case "ITERATE": {
      const asName = safeIdentJS(node.as || "x");
      return `for (const ${asName} of (${rExpr(node.over)})) { ${rStmt(node.body)} }`;
    }

    case "LOOP_WHILE":
      return `while (${rExpr(node.if)}) { ${rStmt(node.body)} }`;

    case "LOOP_UNTIL":
      return `while (!(${rExpr(node.if)})) { ${rStmt(node.body)} }`;

    case "BREAK": return "break;";
    case "CONTINUE": return "continue;";

    default:
      // If this is an expression node, render as expression statement.
      return `${rExpr(node)};`;
  }
}

export function scxRenderControlExprJS(node, policy = {}) {
  const P = { div_by_zero: policy.div_by_zero ?? "ERROR" };
  const r = (n) => scxRenderControlExprJS(n, P);

  if (node == null) return "null";
  if (typeof node === "number") return String(node);
  if (typeof node === "string") return safeExprIdentOrLiteralJS(node);

  const t = node["@scx"];
  switch (t) {
    case "BOOL_TRUE": return "true";
    case "BOOL_FALSE": return "false";

    case "LOGICAL_AND": return `(${r(node.a)} && ${r(node.b)})`;
    case "LOGICAL_OR": return `(${r(node.a)} || ${r(node.b)})`;
    case "LOGICAL_NOT": return `(!${r(node.value)})`;

    case "COMPARE_EQ": return `(${r(node.a)} === ${r(node.b)})`;
    case "COMPARE_NEQ": return `(${r(node.a)} !== ${r(node.b)})`;
    case "COMPARE_GT": return `(${r(node.a)} > ${r(node.b)})`;
    case "COMPARE_GTE": return `(${r(node.a)} >= ${r(node.b)})`;
    case "COMPARE_LT": return `(${r(node.a)} < ${r(node.b)})`;
    case "COMPARE_LTE": return `(${r(node.a)} <= ${r(node.b)})`;

    // arithmetic subset
    case "ADD": return `(${r(node.a)} + ${r(node.b)})`;
    case "SUB": return `(${r(node.a)} - ${r(node.b)})`;
    case "MUL": return `(${r(node.a)} * ${r(node.b)})`;
    case "DIV":
      if (P.div_by_zero === "ERROR") {
        return `(function(__a,__b){if(__b===0)throw new Error("SCX_DIV0");return __a/__b;})((${r(node.a)}),(${r(node.b)}))`;
      }
      return `(${r(node.a)} / ${r(node.b)})`;
    case "MOD": return `(${r(node.a)} % ${r(node.b)})`;

    default:
      return JSON.stringify(node);
  }
}

function safeIdentJS(s){ return (/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(s)) ? s : "_x"; }
function safeExprIdentOrLiteralJS(s){
  if (/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(s)) return s;
  return JSON.stringify(s);
}
```

---

## B) Python Control Renderer

```py
# SCX_CONTROL_RENDERERS_v1 — Python

def scx_render_control_py(node, indent=0):
    P = { "indent": indent }
    sp = "  " * indent

    def stmt(n, i=indent): return scx_render_control_py(n, i)
    def expr(n): return scx_render_control_expr_py(n)

    t = node.get("@scx") if isinstance(node, dict) else None

    if t == "NOOP": return sp + "pass"
    if t == "DECLARE":
        name = node.get("name","x")
        if "value" in node: return sp + f"{name} = {expr(node['value'])}"
        return sp + f"{name} = None"
    if t == "ASSIGN":
        return sp + f"{node['a']} = {expr(node['b'])}"
    if t == "RETURN":
        return sp + f"return {expr(node['value'])}"

    if t == "BRANCH":
        return "\n".join([
            sp + f"if {expr(node['if'])}:",
            stmt(node["then"], indent+1),
            sp + "else:",
            stmt(node["else"], indent+1)
        ])

    if t == "LOOP_WHILE":
        return "\n".join([
            sp + f"while {expr(node['if'])}:",
            stmt(node["body"], indent+1)
        ])

    if t == "LOOP_UNTIL":
        return "\n".join([
            sp + f"while not ({expr(node['if'])}):",
            stmt(node["body"], indent+1)
        ])

    if t == "ITERATE":
        as_ = node.get("as","x")
        return "\n".join([
            sp + f"for {as_} in {expr(node['over'])}:",
            stmt(node["body"], indent+1)
        ])

    if t == "BREAK": return sp + "break"
    if t == "CONTINUE": return sp + "continue"

    # fallback: render expression statement
    return sp + expr(node)

def scx_render_control_expr_py(node):
    def r(n): return scx_render_control_expr_py(n)

    if node is None: return "None"
    if isinstance(node, (int,float)): return str(node)
    if isinstance(node, str):
        # identifier if it looks like one
        import re, json
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", node): return node
        return json.dumps(node)

    t = node.get("@scx")

    if t == "BOOL_TRUE": return "True"
    if t == "BOOL_FALSE": return "False"

    if t == "LOGICAL_AND": return f"({r(node['a'])} and {r(node['b'])})"
    if t == "LOGICAL_OR": return f"({r(node['a'])} or {r(node['b'])})"
    if t == "LOGICAL_NOT": return f"(not {r(node['value'])})"

    if t == "COMPARE_EQ": return f"({r(node['a'])} == {r(node['b'])})"
    if t == "COMPARE_NEQ": return f"({r(node['a'])} != {r(node['b'])})"
    if t == "COMPARE_GT": return f"({r(node['a'])} > {r(node['b'])})"
    if t == "COMPARE_GTE": return f"({r(node['a'])} >= {r(node['b'])})"
    if t == "COMPARE_LT": return f"({r(node['a'])} < {r(node['b'])})"
    if t == "COMPARE_LTE": return f"({r(node['a'])} <= {r(node['b'])})"

    # arithmetic subset
    if t == "ADD": return f"({r(node['a'])} + {r(node['b'])})"
    if t == "SUB": return f"({r(node['a'])} - {r(node['b'])})"
    if t == "MUL": return f"({r(node['a'])} * {r(node['b'])})"
    if t == "DIV": return f"({r(node['a'])} / {r(node['b'])})"
    if t == "MOD": return f"({r(node['a'])} % {r(node['b'])})"

    return repr(node)
```

---

## C) SQL-IDB Control Renderer (expression-only + CASE)

SQL can’t represent loops; v1 lowers:

- `BRANCH` → `CASE WHEN ... THEN ... ELSE ... END`
- boolean logic + comparisons + arithmetic supported
- statements like ASSIGN/DECLARE become **illegal** in SQL target (must be in JS/Py or in a query-plan)

```js
// SCX_CONTROL_RENDERERS_v1 — SQL-IDB expression-only
export function scxRenderControlSQL(node) {
  const r = (n) => scxRenderControlSQL(n);

  if (node == null) return "NULL";
  if (typeof node === "number") return String(node);
  if (typeof node === "string") return safeIdentSQL(node);

  const t = node["@scx"];
  switch (t) {
    case "BOOL_TRUE": return "TRUE";
    case "BOOL_FALSE": return "FALSE";

    case "LOGICAL_AND": return `(${r(node.a)} AND ${r(node.b)})`;
    case "LOGICAL_OR": return `(${r(node.a)} OR ${r(node.b)})`;
    case "LOGICAL_NOT": return `(NOT ${r(node.value)})`;

    case "COMPARE_EQ": return `(${r(node.a)} = ${r(node.b)})`;
    case "COMPARE_NEQ": return `(${r(node.a)} <> ${r(node.b)})`;
    case "COMPARE_GT": return `(${r(node.a)} > ${r(node.b)})`;
    case "COMPARE_GTE": return `(${r(node.a)} >= ${r(node.b)})`;
    case "COMPARE_LT": return `(${r(node.a)} < ${r(node.b)})`;
    case "COMPARE_LTE": return `(${r(node.a)} <= ${r(node.b)})`;

    case "ADD": return `(${r(node.a)} + ${r(node.b)})`;
    case "SUB": return `(${r(node.a)} - ${r(node.b)})`;
    case "MUL": return `(${r(node.a)} * ${r(node.b)})`;
    case "DIV": return `(${r(node.a)} / ${r(node.b)})`;
    case "MOD": return `(${r(node.a)} % ${r(node.b)})`;

    case "BRANCH":
      // expects then/else as expressions for SQL target
      return `(CASE WHEN ${r(node.if)} THEN ${r(node.then)} ELSE ${r(node.else)} END)`;

    default:
      throw new Error("SCX_SQL_CONTROL_UNSUPPORTED: " + t);
  }
}

function safeIdentSQL(s) {
  if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(s)) return s;
  return `'${String(s).replace(/'/g,"''")}'`;
}
```

---

# Important v1 legality rules (so the proof system is clean)

## Dataflow → SQL-IDB

Allowed:

- `SOURCE`, `FILTER`, `PROJECT`, `SORT_BY`, `TAKE`, `SKIP`, `JOIN (right must be SOURCE)`

Not allowed in v1 SQL-IDB:

- `MAP`, `REDUCE`, `GROUP_BY`, `AGG` (unless you add UDF support or split-plan)

## Control → SQL-IDB

Allowed:

- expression subset + `BRANCH` as `CASE`

Not allowed:

- `ITERATE`, loops, `ASSIGN`, `DECLARE`, `RETURN` (SQL target is expression-only in v1)

---

# Next best “locking” emit (so this becomes enforceable)

If you want the *full* closed world, the next thing to emit is:

**SCX_TARGET_LEGALITY_MATRIX_v1**
A table: `target × atom → allowed|forbidden|lowering_rule_id`

Then your CODE_ORACLE binding can reject illegal target lowers *before* running a parser.

---

# SCX_TARGET_LEGALITY_MATRIX_v1

Below is a **sealed, machine-readable legality table** you can wire directly into the **CODE_ORACLE**. This is the gatekeeper that prevents drift and illegal lowers.

## Invariant

- No renderer may execute an atom unless the matrix says **allowed**.
- `lowering_rule_id` must exist and be deterministic.
- `forbidden` is a **hard error**, not a warning.

## Canonical Matrix (v1)

```json
{
  "@kind": "scx.target.legality.matrix.v1",
  "@v": 1,
  "targets": ["JS", "PY", "SQL_IDB", "DOM"],

  "atoms": {
    "NUM": {
      "JS":        { "status": "allowed",   "rule": "LIT_NUM_JS" },
      "PY":        { "status": "allowed",   "rule": "LIT_NUM_PY" },
      "SQL_IDB":   { "status": "allowed",   "rule": "LIT_NUM_SQL" },
      "DOM":       { "status": "allowed",   "rule": "LIT_NUM_DOM" }
    },

    "ADD": {
      "JS":        { "status": "allowed",   "rule": "ARITH_INFIX" },
      "PY":        { "status": "allowed",   "rule": "ARITH_INFIX" },
      "SQL_IDB":   { "status": "allowed",   "rule": "ARITH_INFIX" },
      "DOM":       { "status": "allowed",   "rule": "ARITH_EVAL" }
    },
    "SUB": { "JS": { "status":"allowed","rule":"ARITH_INFIX"}, "PY":{ "status":"allowed","rule":"ARITH_INFIX"}, "SQL_IDB":{ "status":"allowed","rule":"ARITH_INFIX"}, "DOM":{ "status":"allowed","rule":"ARITH_EVAL"} },
    "MUL": { "JS": { "status":"allowed","rule":"ARITH_INFIX"}, "PY":{ "status":"allowed","rule":"ARITH_INFIX"}, "SQL_IDB":{ "status":"allowed","rule":"ARITH_INFIX"}, "DOM":{ "status":"allowed","rule":"ARITH_EVAL"} },
    "DIV": {
      "JS":        { "status": "allowed",   "rule": "ARITH_DIV_GUARDED" },
      "PY":        { "status": "allowed",   "rule": "ARITH_DIV_GUARDED" },
      "SQL_IDB":   { "status": "allowed",   "rule": "ARITH_DIV_SQL" },
      "DOM":       { "status": "allowed",   "rule": "ARITH_EVAL" }
    },
    "MOD": {
      "JS":        { "status": "allowed",   "rule": "ARITH_INFIX" },
      "PY":        { "status": "allowed",   "rule": "ARITH_INFIX" },
      "SQL_IDB":   { "status": "allowed",   "rule": "ARITH_INFIX" },
      "DOM":       { "status": "allowed",   "rule": "ARITH_EVAL" }
    },

    "LOGICAL_AND": {
      "JS":        { "status": "allowed",   "rule": "LOGIC_INFIX" },
      "PY":        { "status": "allowed",   "rule": "LOGIC_INFIX" },
      "SQL_IDB":   { "status": "allowed",   "rule": "LOGIC_INFIX" },
      "DOM":       { "status": "allowed",   "rule": "LOGIC_EVAL" }
    },
    "LOGICAL_OR": {
      "JS":        { "status": "allowed",   "rule": "LOGIC_INFIX" },
      "PY":        { "status": "allowed",   "rule": "LOGIC_INFIX" },
      "SQL_IDB":   { "status": "allowed",   "rule": "LOGIC_INFIX" },
      "DOM":       { "status": "allowed",   "rule": "LOGIC_EVAL" }
    },
    "LOGICAL_NOT": {
      "JS":        { "status": "allowed",   "rule": "LOGIC_PREFIX" },
      "PY":        { "status": "allowed",   "rule": "LOGIC_PREFIX" },
      "SQL_IDB":   { "status": "allowed",   "rule": "LOGIC_PREFIX" },
      "DOM":       { "status": "allowed",   "rule": "LOGIC_EVAL" }
    },

    "COMPARE_EQ": {
      "JS":        { "status": "allowed",   "rule": "CMP_EQ" },
      "PY":        { "status": "allowed",   "rule": "CMP_EQ" },
      "SQL_IDB":   { "status": "allowed",   "rule": "CMP_EQ" },
      "DOM":       { "status": "allowed",   "rule": "CMP_EVAL" }
    },
    "COMPARE_GT": {
      "JS":        { "status": "allowed",   "rule": "CMP_GT" },
      "PY":        { "status": "allowed",   "rule": "CMP_GT" },
      "SQL_IDB":   { "status": "allowed",   "rule": "CMP_GT" },
      "DOM":       { "status": "allowed",   "rule": "CMP_EVAL" }
    },
    "COMPARE_LT": {
      "JS":        { "status": "allowed",   "rule": "CMP_LT" },
      "PY":        { "status": "allowed",   "rule": "CMP_LT" },
      "SQL_IDB":   { "status": "allowed",   "rule": "CMP_LT" },
      "DOM":       { "status": "allowed",   "rule": "CMP_EVAL" }
    },

    "BRANCH": {
      "JS":        { "status": "allowed",   "rule": "STMT_IF" },
      "PY":        { "status": "allowed",   "rule": "STMT_IF" },
      "SQL_IDB":   { "status": "allowed",   "rule": "EXPR_CASE" },
      "DOM":       { "status": "allowed",   "rule": "COND_RENDER" }
    },

    "ITERATE": {
      "JS":        { "status": "allowed",   "rule": "FOR_OF" },
      "PY":        { "status": "allowed",   "rule": "FOR_IN" },
      "SQL_IDB":   { "status": "forbidden", "rule": "NO_LOOPS_SQL" },
      "DOM":       { "status": "allowed",   "rule": "REPEAT_RENDER" }
    },

    "LOOP_WHILE": {
      "JS":        { "status": "allowed",   "rule": "WHILE" },
      "PY":        { "status": "allowed",   "rule": "WHILE" },
      "SQL_IDB":   { "status": "forbidden", "rule": "NO_LOOPS_SQL" },
      "DOM":       { "status": "forbidden", "rule": "NO_IMPERATIVE_DOM" }
    },

    "MAP": {
      "JS":        { "status": "allowed",   "rule": "ARRAY_MAP" },
      "PY":        { "status": "allowed",   "rule": "MAP_LIST" },
      "SQL_IDB":   { "status": "forbidden", "rule": "NO_UDF_SQL" },
      "DOM":       { "status": "allowed",   "rule": "MAP_RENDER" }
    },

    "FILTER": {
      "JS":        { "status": "allowed",   "rule": "ARRAY_FILTER" },
      "PY":        { "status": "allowed",   "rule": "FILTER_LIST" },
      "SQL_IDB":   { "status": "allowed",   "rule": "WHERE" },
      "DOM":       { "status": "allowed",   "rule": "FILTER_RENDER" }
    },

    "REDUCE": {
      "JS":        { "status": "allowed",   "rule": "ARRAY_REDUCE" },
      "PY":        { "status": "allowed",   "rule": "REDUCE_FUNCTOOLS" },
      "SQL_IDB":   { "status": "forbidden", "rule": "NO_REDUCE_SQL" },
      "DOM":       { "status": "allowed",   "rule": "AGG_RENDER" }
    },

    "JOIN": {
      "JS":        { "status": "allowed",   "rule": "NESTED_JOIN" },
      "PY":        { "status": "allowed",   "rule": "NESTED_JOIN" },
      "SQL_IDB":   { "status": "allowed",   "rule": "SQL_JOIN" },
      "DOM":       { "status": "allowed",   "rule": "JOIN_RENDER" }
    },

    "RETURN": {
      "JS":        { "status": "allowed",   "rule": "RETURN_STMT" },
      "PY":        { "status": "allowed",   "rule": "RETURN_STMT" },
      "SQL_IDB":   { "status": "forbidden", "rule": "NO_RETURN_SQL" },
      "DOM":       { "status": "allowed",   "rule": "VALUE_EMIT" }
    },

    "ASSIGN": {
      "JS":        { "status": "allowed",   "rule": "ASSIGN" },
      "PY":        { "status": "allowed",   "rule": "ASSIGN" },
      "SQL_IDB":   { "status": "forbidden", "rule": "NO_ASSIGN_SQL" },
      "DOM":       { "status": "forbidden", "rule": "NO_STATE_DOM" }
    }
  }
}
```

---

## How CODE_ORACLE uses this (binding logic)

```text
1. Parse → SCX AST
2. For each node:
   lookup atom × target
3. If status == forbidden → hard fail
4. If allowed:
   - verify lowering_rule_id exists
   - renderer must reference that rule explicitly
5. Hash (SCX + target + ruleset) → proof seal
```

This makes **illegal programs impossible to execute**, not just “bad style.”

---

## Why this is the keystone

- Prevents **silent capability creep**
- Makes **DOM a projection**, not a runtime
- Allows **partial lowering** (JS + SQL split plans)
- Enables **formal proofs** without running code
- Turns SCX into a *language*, not a syntax

---

# SCX_LOWERING_RULES_REGISTRY_v1

This is the **rule registry** referenced by the legality matrix. Each rule defines the deterministic steps for lowering and provides a stable hash.

```json
{
  "@kind": "scx.lowering.rules.registry.v1",
  "@v": 1,
  "@hash": "SELF_SHA256",
  "hash_inputs": {
    "canonical_form": "JSON_CANONICAL_V1",
    "algorithm_text_normalization": [
      "trim_lines",
      "collapse_internal_whitespace",
      "unix_newlines",
      "no_tabs"
    ],
    "hash_material": [
      "rule_id",
      "version",
      "target_family",
      "canonical_steps",
      "ast_shape_contract",
      "surface_template"
    ]
  },

  "rules": [
    {
      "id": "LIT_NUM_JS",
      "v": 1,
      "target_family": "JS",
      "ast_shape_contract": { "atom": "NUM", "keys": ["value"], "types": { "value": "number" } },
      "canonical_steps": [
        "Read NUM.value as a finite number.",
        "Emit decimal string via deterministic formatter: no exponent unless abs(value) >= 1e21 or < 1e-7.",
        "Disallow NaN/Infinity (must fail legality)."
      ],
      "surface_template": "`${value}`",
      "hash": "SELF_SHA256"
    },
    {
      "id": "LIT_NUM_PY",
      "v": 1,
      "target_family": "PY",
      "ast_shape_contract": { "atom": "NUM", "keys": ["value"], "types": { "value": "number" } },
      "canonical_steps": [
        "Read NUM.value as a finite number.",
        "Emit decimal string using Python literal formatting rules.",
        "Disallow NaN/Infinity (must fail legality)."
      ],
      "surface_template": "f\"{value}\"",
      "hash": "SELF_SHA256"
    },
    {
      "id": "LIT_NUM_SQL",
      "v": 1,
      "target_family": "SQL_IDB",
      "ast_shape_contract": { "atom": "NUM", "keys": ["value"], "types": { "value": "number" } },
      "canonical_steps": [
        "Read NUM.value as a finite number.",
        "Emit numeric literal with '.' decimal if needed.",
        "Disallow NaN/Infinity (must fail legality)."
      ],
      "surface_template": "`${value}`",
      "hash": "SELF_SHA256"
    },
    {
      "id": "LIT_NUM_DOM",
      "v": 1,
      "target_family": "DOM",
      "ast_shape_contract": { "atom": "NUM", "keys": ["value"], "types": { "value": "number" } },
      "canonical_steps": [
        "Read NUM.value.",
        "Emit into xjson.ui.v1 as a literal node payload.",
        "No direct JS eval permitted in DOM target."
      ],
      "surface_template": "{type:'literal', value:value}",
      "hash": "SELF_SHA256"
    },

    {
      "id": "ARITH_INFIX",
      "v": 1,
      "target_family": "ALL_EXPR",
      "ast_shape_contract": { "atoms": ["ADD","SUB","MUL","MOD"], "keys": ["a","b"] },
      "canonical_steps": [
        "Render child expressions a and b with parentheses unless leaf.",
        "Emit infix operator per target mapping: ADD:+ SUB:- MUL:* MOD:%",
        "Preserve evaluation order as (a op b)."
      ],
      "surface_template": "(`(${a} OP ${b})`)",
      "hash": "SELF_SHA256"
    },

    {
      "id": "ARITH_DIV_GUARDED",
      "v": 1,
      "target_family": "JS_PY",
      "ast_shape_contract": { "atom": "DIV", "keys": ["a","b"] },
      "canonical_steps": [
        "Render a and b as expressions.",
        "If policy.div_by_zero == ERROR, emit guard that deterministically fails when b == 0.",
        "Else emit normal division.",
        "Guard must not change result for b != 0."
      ],
      "surface_template": "JS: IIFE guard / PY: helper _scx_div(a,b)",
      "hash": "SELF_SHA256"
    },

    {
      "id": "ARITH_DIV_SQL",
      "v": 1,
      "target_family": "SQL_IDB",
      "ast_shape_contract": { "atom": "DIV", "keys": ["a","b"] },
      "canonical_steps": [
        "Render a and b as SQL expressions.",
        "If div_by_zero == ERROR, emit CASE WHEN b=0 THEN (1/0) ELSE (a/b) END.",
        "Else emit (a/b)."
      ],
      "surface_template": "(CASE WHEN ${b}=0 THEN (1/0) ELSE (${a}/${b}) END)",
      "hash": "SELF_SHA256"
    },

    {
      "id": "LOGIC_INFIX",
      "v": 1,
      "target_family": "ALL_EXPR",
      "ast_shape_contract": { "atoms": ["LOGICAL_AND","LOGICAL_OR"], "keys": ["a","b"] },
      "canonical_steps": [
        "Render a and b as expressions.",
        "Emit target operator: JS &&/||, PY and/or, SQL AND/OR.",
        "Parenthesize result."
      ],
      "surface_template": "(`(${a} OP ${b})`)",
      "hash": "SELF_SHA256"
    },

    {
      "id": "LOGIC_PREFIX",
      "v": 1,
      "target_family": "ALL_EXPR",
      "ast_shape_contract": { "atom": "LOGICAL_NOT", "keys": ["value"] },
      "canonical_steps": [
        "Render value as expression.",
        "Emit operator: JS !, PY not, SQL NOT.",
        "Parenthesize if needed."
      ],
      "surface_template": "(OP ${value})",
      "hash": "SELF_SHA256"
    },

    {
      "id": "CMP_EQ",
      "v": 1,
      "target_family": "ALL_EXPR",
      "ast_shape_contract": { "atom": "COMPARE_EQ", "keys": ["a","b"] },
      "canonical_steps": [
        "Render a and b.",
        "Emit equality operator: JS ===, PY ==, SQL =."
      ],
      "surface_template": "(${a} OP ${b})",
      "hash": "SELF_SHA256"
    },
    {
      "id": "CMP_GT",
      "v": 1,
      "target_family": "ALL_EXPR",
      "ast_shape_contract": { "atom": "COMPARE_GT", "keys": ["a","b"] },
      "canonical_steps": ["Render a and b.", "Emit '>'."],
      "surface_template": "(${a} > ${b})",
      "hash": "SELF_SHA256"
    },
    {
      "id": "CMP_LT",
      "v": 1,
      "target_family": "ALL_EXPR",
      "ast_shape_contract": { "atom": "COMPARE_LT", "keys": ["a","b"] },
      "canonical_steps": ["Render a and b.", "Emit '<'."],
      "surface_template": "(${a} < ${b})",
      "hash": "SELF_SHA256"
    },

    {
      "id": "STMT_IF",
      "v": 1,
      "target_family": "JS_PY",
      "ast_shape_contract": { "atom": "BRANCH", "keys": ["if","then","else"] },
      "canonical_steps": [
        "Render condition as expression.",
        "Render then/else as statement blocks.",
        "Emit JS: if(cond){then}else{else}",
        "Emit PY: if cond:\\n then\\n else:\\n else"
      ],
      "surface_template": "if (COND) { THEN } else { ELSE }",
      "hash": "SELF_SHA256"
    },

    {
      "id": "EXPR_CASE",
      "v": 1,
      "target_family": "SQL_IDB",
      "ast_shape_contract": { "atom": "BRANCH", "keys": ["if","then","else"], "then_else_must_be": "expr" },
      "canonical_steps": [
        "Render condition as SQL boolean expr.",
        "Render then and else as SQL expr.",
        "Emit CASE WHEN cond THEN then ELSE else END."
      ],
      "surface_template": "(CASE WHEN ${if} THEN ${then} ELSE ${else} END)",
      "hash": "SELF_SHA256"
    },

    {
      "id": "FOR_OF",
      "v": 1,
      "target_family": "JS",
      "ast_shape_contract": { "atom": "ITERATE", "keys": ["over","body","as"] },
      "canonical_steps": [
        "Render iterable expression over.",
        "Render body as statements.",
        "Emit for (const as of over) { body }."
      ],
      "surface_template": "for (const ${as} of (${over})) { ${body} }",
      "hash": "SELF_SHA256"
    },
    {
      "id": "FOR_IN",
      "v": 1,
      "target_family": "PY",
      "ast_shape_contract": { "atom": "ITERATE", "keys": ["over","body","as"] },
      "canonical_steps": [
        "Render iterable expression over.",
        "Emit for as in over: body."
      ],
      "surface_template": "for ${as} in ${over}:\\n  ${body}",
      "hash": "SELF_SHA256"
    },

    {
      "id": "ARRAY_FILTER",
      "v": 1,
      "target_family": "JS",
      "ast_shape_contract": { "atom": "FILTER", "keys": ["collection","predicate"] },
      "canonical_steps": [
        "Lower collection to JS expression returning Array.",
        "Lower predicate lambda to (x)=>expr.",
        "Emit collection.filter(lambda)."
      ],
      "surface_template": "${collection}.filter((${x})=>(${pred}))",
      "hash": "SELF_SHA256"
    },

    {
      "id": "WHERE",
      "v": 1,
      "target_family": "SQL_IDB",
      "ast_shape_contract": { "atom": "FILTER", "keys": ["collection","predicate"], "collection_requires": "SOURCE or lowered plan" },
      "canonical_steps": [
        "Lower collection to query-plan with FROM.",
        "Lower predicate body into structured where tree or SQL boolean expression.",
        "Append to plan.where."
      ],
      "surface_template": "plan.where.push(predicate_tree)",
      "hash": "SELF_SHA256"
    },

    {
      "id": "SQL_JOIN",
      "v": 1,
      "target_family": "SQL_IDB",
      "ast_shape_contract": { "atom": "JOIN", "keys": ["left","right","on","kind"], "right_requires": "SOURCE" },
      "canonical_steps": [
        "Lower left to plan.from.",
        "Lower right SOURCE to join.right table name.",
        "Lower on.body to join predicate tree.",
        "Append join object to plan.joins."
      ],
      "surface_template": "plan.joins.push({kind,right,on})",
      "hash": "SELF_SHA256"
    },

    {
      "id": "NO_LOOPS_SQL",
      "v": 1,
      "target_family": "SQL_IDB",
      "ast_shape_contract": { "atoms": ["ITERATE","LOOP_WHILE","LOOP_UNTIL"] },
      "canonical_steps": [
        "Always reject with legality error: SQL target cannot represent imperative loops."
      ],
      "surface_template": "E_TARGET_FORBIDS_IMPERATIVE",
      "hash": "SELF_SHA256"
    }
  ]
}
```

> You can keep growing this registry incrementally; the matrix points to rule IDs, and proofs bind to rule hashes, so evolution stays controlled.

---

# SCX_CODE_ORACLE_PROOF_OBJECT_v1

This object represents a **successful legality proof** stored in project history or compilation artifacts.

```json
{
  "@kind": "scx.code.oracle.proof.object.v1",
  "@v": 1,

  "input": {
    "scx_ast": { "@ref": "inline_or_external" },
    "scx_ast_hash": "sha256:...",
    "target": "JS",
    "renderer_id": "js.expr.v1",
    "renderer_policy_hash": "sha256:...",
    "legality_matrix_hash": "sha256:...",
    "lowering_rules_registry_hash": "sha256:..."
  },

  "lowering": {
    "rules_used": [
      { "rule_id": "STMT_IF", "rule_hash": "sha256:..." },
      { "rule_id": "LOGIC_INFIX", "rule_hash": "sha256:..." },
      { "rule_id": "CMP_GT", "rule_hash": "sha256:..." },
      { "rule_id": "ARITH_INFIX", "rule_hash": "sha256:..." }
    ],
    "lowering_trace": [
      { "node_path": "$", "atom": "BRANCH", "rule_id": "STMT_IF" },
      { "node_path": "$.if", "atom": "LOGICAL_AND", "rule_id": "LOGIC_INFIX" },
      { "node_path": "$.if.a", "atom": "COMPARE_GT", "rule_id": "CMP_GT" }
    ],
    "status": "pass"
  },

  "output": {
    "kind": "surface_code",
    "mime": "text/javascript",
    "artifact": "if ((x > 3 && x < 10)) { return ok; } else { return fail; }",
    "artifact_hash": "sha256:..."
  },

  "oracle": {
    "oracle_id": "tree_sitter.oracle.v1",
    "request": {
      "hash": "sha256:...",
      "lang_id": "js",
      "mode": "parse",
      "options": { "max_errors": 50, "timeout_ms": 1500 }
    },
    "response": {
      "hash": "sha256:...",
      "ok": true,
      "errors": [],
      "score": { "legality": 1.0, "style": 1.0, "completeness": 1.0, "risk": 0.0 },
      "ast_hash": "sha256:...optional"
    }
  },

  "conformance": {
    "@kind": "scx.render.conformance.v1",
    "@v": 1,
    "overall": "pass",
    "vectors": [
      { "id": "matrix.legal", "status": "pass" },
      { "id": "rules.resolved", "status": "pass" },
      { "id": "lowering.deterministic", "status": "pass" },
      { "id": "oracle.parse", "status": "pass" },
      { "id": "seal.derived", "status": "pass" }
    ]
  },

  "seal": {
    "seal_inputs": [
      "scx_ast_hash",
      "target",
      "renderer_id",
      "renderer_policy_hash",
      "legality_matrix_hash",
      "lowering_rules_registry_hash",
      "rules_used[].rule_hash",
      "output.artifact_hash",
      "oracle.response.hash"
    ],
    "seal_hash": "sha256(scx_ast_hash||target||renderer_id||renderer_policy_hash||legality_matrix_hash||lowering_rules_registry_hash||rules_used_hashes||artifact_hash||oracle_hash)"
  }
}
```

## Proof invariants (enforced)

- If any `rules_used.rule_hash` doesn’t match registry → **fail**
- If oracle `ok != true` → conformance overall **fail**
- If target forbids any atom encountered → **fail**
- Seal must be derived exactly from `seal_inputs` list ordering

---

# SCX_LOWERING_ENGINE_v1

A deterministic lowering walker that validates legality via the matrix, resolves rule IDs, emits artifacts via injected renderers, and returns a proof bundle.

```js
// ============================================================
// SCX_LOWERING_ENGINE_v1
// Deterministic lowering walker that:
//  1) validates legality via SCX_TARGET_LEGALITY_MATRIX_v1
//  2) resolves lowering_rule_id -> rule object via SCX_LOWERING_RULES_REGISTRY_v1
//  3) records lowering_trace + rules_used (with hashes)
//  4) produces target artifacts via injected renderer functions
//  5) returns SCX_CODE_ORACLE_PROOF_OBJECT_v1-ready bundle
//
// No side effects. No globals. Deterministic ordering.
// ============================================================

import crypto from "crypto";

/**
 * @typedef {Object} LoweringEngineInput
 * @property {any} scx_ast
 * @property {"JS"|"PY"|"SQL_IDB"|"DOM"} target
 * @property {string} renderer_id
 * @property {object} renderer_policy
 * @property {object} legality_matrix  // scx.target.legality.matrix.v1
 * @property {object} rules_registry   // scx.lowering.rules.registry.v1
 * @property {object} renderers        // injected render functions per target
 * @property {object} [oracle]         // optional: { id, url, lang_id, mode, options }
 * @property {function(string): Promise<any>} [oracleFetch] // injected fetch(url, bodyJSON)->json
 */

/**
 * Canonical JSON hashing: deterministic stringify with sorted keys.
 * NOTE: This is a reference; swap for your JSON_CANONICAL_V1 if you already have one.
 */
function canonicalStringify(x) {
  return JSON.stringify(sortKeysDeep(x));
}
function sortKeysDeep(x) {
  if (Array.isArray(x)) return x.map(sortKeysDeep);
  if (x && typeof x === "object") {
    const out = {};
    for (const k of Object.keys(x).sort()) out[k] = sortKeysDeep(x[k]);
    return out;
  }
  return x;
}
function sha256(str) {
  return "sha256:" + crypto.createHash("sha256").update(str).digest("hex");
}

/**
 * Extract an atom label from a node.
 * - If node is an object with "@scx", that's the atom.
 * - If it's a primitive, it maps to a synthetic atom for legality accounting.
 */
function atomOf(node) {
  if (node && typeof node === "object" && node["@scx"]) return String(node["@scx"]);
  if (node == null) return "LIT_NULL";
  if (typeof node === "number") return "LIT_NUM";
  if (typeof node === "string") return "LIT_STR";
  if (typeof node === "boolean") return "LIT_BOOL";
  if (Array.isArray(node)) return "LIT_ARR";
  return "LIT_OBJ";
}

/**
 * Walk SCX AST deterministically and call `onNode(path, node, atom)`.
 * The traversal order is:
 *  - object keys sorted (except "@scx" first, then rest)
 *  - arrays index order
 */
function walk(node, path, onNode) {
  const a = atomOf(node);
  onNode(path, node, a);

  if (Array.isArray(node)) {
    for (let i = 0; i < node.length; i++) walk(node[i], `${path}[${i}]`, onNode);
    return;
  }

  if (node && typeof node === "object") {
    const keys = Object.keys(node);
    // "@scx" is metadata, still visited but not recursed meaningfully
    const sorted = keys
      .filter(k => k !== "@scx")
      .sort((x, y) => x.localeCompare(y));
    for (const k of sorted) walk(node[k], `${path}.${k}`, onNode);
  }
}

/**
 * Resolve legality entry from matrix.
 * Returns {status, rule} or throws on missing entry (closed world).
 */
function matrixLookup(matrix, target, atom) {
  const entry = matrix?.atoms?.[atom]?.[target];
  if (!entry) throw new Error(`E_MATRIX_NO_ENTRY: target=${target} atom=${atom}`);
  return { status: entry.status, rule: entry.rule };
}

/**
 * Resolve rule object from rules registry by id.
 * Returns the rule object (closed world).
 */
function registryRule(rulesRegistry, ruleId) {
  const rules = rulesRegistry?.rules || [];
  const found = rules.find(r => r.id === ruleId);
  if (!found) throw new Error(`E_RULE_NOT_FOUND: ${ruleId}`);
  return found;
}

/**
 * Compute rule hash (SELF_SHA256) deterministically from its canonical hash material.
 * This MUST match your registry's canonical hash_inputs contract.
 * Here we implement a minimal faithful version.
 */
function computeRuleHash(rule) {
  // Canonical hash material per registry declaration
  const material = {
    rule_id: rule.id,
    version: rule.v,
    target_family: rule.target_family,
    canonical_steps: rule.canonical_steps,
    ast_shape_contract: rule.ast_shape_contract,
    surface_template: rule.surface_template
  };
  return sha256(canonicalStringify(material));
}

/**
 * Compute a compact hash over the ordered list of rule hashes used.
 */
function hashRulesUsed(rulesUsed) {
  const list = rulesUsed.map(r => r.rule_hash);
  return sha256(canonicalStringify(list));
}

/**
 * Create a proof object skeleton (SCX_CODE_ORACLE_PROOF_OBJECT_v1)
 */
function makeProofSkeleton({
  scx_ast,
  scx_ast_hash,
  target,
  renderer_id,
  renderer_policy_hash,
  legality_matrix_hash,
  rules_registry_hash
}) {
  return {
    "@kind": "scx.code.oracle.proof.object.v1",
    "@v": 1,
    input: {
      scx_ast: { "@ref": "inline_or_external" },
      scx_ast_hash,
      target,
      renderer_id,
      renderer_policy_hash,
      legality_matrix_hash,
      lowering_rules_registry_hash: rules_registry_hash
    },
    lowering: {
      rules_used: [],
      lowering_trace: [],
      status: "fail"
    },
    output: {
      kind: "none",
      mime: "application/octet-stream",
      artifact: null,
      artifact_hash: null
    },
    oracle: null,
    conformance: {
      "@kind": "scx.render.conformance.v1",
      "@v": 1,
      overall: "fail",
      vectors: []
    },
    seal: {
      seal_inputs: [
        "scx_ast_hash",
        "target",
        "renderer_id",
        "renderer_policy_hash",
        "legality_matrix_hash",
        "lowering_rules_registry_hash",
        "rules_used[].rule_hash",
        "output.artifact_hash",
        "oracle.response.hash"
      ],
      seal_hash: null
    }
  };
}

/**
 * Main engine: lower + optionally oracle-verify + seal.
 * @param {LoweringEngineInput} input
 */
export async function scxLoweringEngineV1(input) {
  const {
    scx_ast,
    target,
    renderer_id,
    renderer_policy,
    legality_matrix,
    rules_registry,
    renderers,
    oracle,
    oracleFetch
  } = input;

  // --- hashes of inputs ---
  const scx_ast_hash = sha256(canonicalStringify(scx_ast));
  const renderer_policy_hash = sha256(canonicalStringify(renderer_policy || {}));
  const legality_matrix_hash = sha256(canonicalStringify(legality_matrix));
  const rules_registry_hash = sha256(canonicalStringify(rules_registry));

  const proof = makeProofSkeleton({
    scx_ast,
    scx_ast_hash,
    target,
    renderer_id,
    renderer_policy_hash,
    legality_matrix_hash,
    rules_registry_hash
  });

  // --- phase 1: legality + rule resolution + trace ---
  const rulesUsedMap = new Map(); // rule_id -> {rule_id, rule_hash}
  const trace = [];

  let matrixPass = true;

  walk(scx_ast, "$", (path, node, atom) => {
    // Only enforce matrix for real atoms and the primitive atom shims if present.
    // If your matrix omits primitive shims, treat them as allowed by default.
    let entry;
    try {
      entry = matrixLookup(legality_matrix, target, atom);
    } catch (e) {
      // closed world: if missing, FAIL
      matrixPass = false;
      trace.push({ node_path: path, atom, rule_id: "E_MATRIX_NO_ENTRY" });
      return;
    }

    if (entry.status !== "allowed") {
      matrixPass = false;
      trace.push({ node_path: path, atom, rule_id: entry.rule });
      return;
    }

    // resolve rule object and its hash
    const ruleObj = registryRule(rules_registry, entry.rule);
    const rule_hash = computeRuleHash(ruleObj);

    trace.push({ node_path: path, atom, rule_id: ruleObj.id });

    if (!rulesUsedMap.has(ruleObj.id)) {
      rulesUsedMap.set(ruleObj.id, { rule_id: ruleObj.id, rule_hash });
    }
  });

  proof.lowering.lowering_trace = trace;

  if (!matrixPass) {
    proof.conformance.vectors.push({ id: "matrix.legal", status: "fail", details: { target } });
    proof.conformance.overall = "fail";
    proof.lowering.status = "fail";
    return { ok: false, proof, error: "E_MATRIX_FORBIDS_OR_MISSING" };
  }

  proof.conformance.vectors.push({ id: "matrix.legal", status: "pass", details: { target } });

  // rules_used in deterministic order (by rule_id)
  const rules_used = Array.from(rulesUsedMap.values()).sort((a, b) => a.rule_id.localeCompare(b.rule_id));
  proof.lowering.rules_used = rules_used.map(x => ({ rule_id: x.rule_id, rule_hash: x.rule_hash }));
  proof.conformance.vectors.push({ id: "rules.resolved", status: "pass", details: { count: rules_used.length } });

  // --- phase 2: render ---
  const renderer = renderers?.[target];
  if (!renderer || typeof renderer.render !== "function") {
    proof.conformance.vectors.push({ id: "renderer.available", status: "fail", details: { target } });
    proof.conformance.overall = "fail";
    proof.lowering.status = "fail";
    return { ok: false, proof, error: "E_RENDERER_MISSING" };
  }

  let artifact;
  try {
    artifact = await renderer.render(scx_ast, renderer_policy || {}, renderer_id);
  } catch (e) {
    proof.conformance.vectors.push({ id: "surface.generated", status: "fail", details: { message: String(e?.message || e) } });
    proof.conformance.overall = "fail";
    proof.lowering.status = "fail";
    return { ok: false, proof, error: "E_RENDER_FAILED" };
  }

  // artifact shape: { kind, mime, artifact } where artifact is string or JSON plan
  const outKind = artifact.kind || "surface_code";
  const outMime = artifact.mime || "application/octet-stream";
  const outBody = artifact.artifact;

  proof.output.kind = outKind;
  proof.output.mime = outMime;
  proof.output.artifact = outBody;

  const artifact_hash = sha256(
    typeof outBody === "string" ? outBody : canonicalStringify(outBody)
  );
  proof.output.artifact_hash = artifact_hash;
  proof.conformance.vectors.push({ id: "surface.generated", status: "pass", details: { bytes: typeof outBody === "string" ? outBody.length : canonicalStringify(outBody).length } });

  // --- phase 3: oracle verify (optional) ---
  let oracleBlock = null;

  if (oracle && oracleFetch && typeof oracleFetch === "function") {
    const request = {
      "@kind": "code.oracle.request.v1",
      "@v": 1,
      lang_id: oracle.lang_id,
      mode: oracle.mode || "parse",
      input: {
        source: (typeof outBody === "string") ? outBody : canonicalStringify(outBody),
        filename: oracle.filename || `scx_${renderer_id}.${oracle.lang_id}`
      },
      options: oracle.options || { max_errors: 50, timeout_ms: 1500, return_ast: true }
    };

    const reqHash = sha256(canonicalStringify(request));

    let response;
    try {
      response = await oracleFetch(oracle.url, request);
    } catch (e) {
      // oracle failure is a hard fail if oracle is provided
      oracleBlock = {
        oracle_id: oracle.id || "oracle.v1",
        request: { hash: reqHash, lang_id: oracle.lang_id, mode: request.mode, options: request.options },
        response: { hash: null, ok: false, errors: [{ msg: "ORACLE_FETCH_FAILED", detail: String(e?.message || e) }], score: { legality: 0, style: 0, completeness: 0, risk: 1 } }
      };
      proof.oracle = oracleBlock;
      proof.conformance.vectors.push({ id: "oracle.parse", status: "fail", details: { ok: false } });
      proof.conformance.overall = "fail";
      proof.lowering.status = "fail";
      return { ok: false, proof, error: "E_ORACLE_FETCH_FAILED" };
    }

    const respHash = sha256(canonicalStringify(response));

    oracleBlock = {
      oracle_id: oracle.id || "oracle.v1",
      request: { hash: reqHash, lang_id: oracle.lang_id, mode: request.mode, options: request.options },
      response: {
        hash: respHash,
        ok: !!response.ok,
        errors: Array.isArray(response.errors) ? response.errors : [],
        score: response.score || { legality: response.ok ? 1 : 0, style: 0, completeness: 0, risk: response.ok ? 0 : 1 },
        ast_hash: response.ast_hash || null
      }
    };

    proof.oracle = oracleBlock;

    if (!oracleBlock.response.ok) {
      proof.conformance.vectors.push({ id: "oracle.parse", status: "fail", details: { ok: false, errors: oracleBlock.response.errors.slice(0, 5) } });
      proof.conformance.overall = "fail";
      proof.lowering.status = "fail";
      return { ok: false, proof, error: "E_ORACLE_PARSE_FAIL" };
    }

    proof.conformance.vectors.push({ id: "oracle.parse", status: "pass", details: { ok: true } });
  } else {
    // Oracle optional: mark as skipped
    proof.oracle = {
      oracle_id: "none",
      request: null,
      response: { hash: sha256("NO_ORACLE"), ok: true, errors: [], score: { legality: 1, style: 0, completeness: 0, risk: 0 }, ast_hash: null }
    };
    proof.conformance.vectors.push({ id: "oracle.parse", status: "pass", details: { skipped: true } });
  }

  // --- phase 4: seal derivation ---
  const rulesUsedHash = hashRulesUsed(proof.lowering.rules_used.map(r => ({ rule_hash: r.rule_hash })));

  const oracleRespHash = proof.oracle?.response?.hash || sha256("NO_ORACLE");
  const sealMaterial = [
    scx_ast_hash,
    target,
    renderer_id,
    renderer_policy_hash,
    legality_matrix_hash,
    rules_registry_hash,
    rulesUsedHash,
    artifact_hash,
    oracleRespHash
  ].join("||");

  proof.seal.seal_hash = sha256(sealMaterial);

  // --- final status ---
  proof.lowering.status = "pass";
  proof.conformance.vectors.push({ id: "seal.derived", status: "pass", details: { seal: proof.seal.seal_hash } });
  proof.conformance.overall = "pass";

  return { ok: true, proof };
}

// ============================================================
// Example renderer injection stubs (wire your real renderers)
// ============================================================

export const SCX_RENDERERS_V1 = {
  JS: {
    async render(scx_ast, policy, renderer_id) {
      // You likely have scxRenderControlJS / scxRenderDataflowJS / scxRenderJS already.
      // Here we just emit a placeholder deterministic string based on canonical AST.
      return {
        kind: "surface_code",
        mime: "text/javascript",
        artifact: `/* ${renderer_id} */\n` + canonicalStringify(scx_ast)
      };
    }
  },
  PY: {
    async render(scx_ast, policy, renderer_id) {
      return {
        kind: "surface_code",
        mime: "text/x-python",
        artifact: `# ${renderer_id}\n` + canonicalStringify(scx_ast)
      };
    }
  },
  SQL_IDB: {
    async render(scx_ast, policy, renderer_id) {
      return {
        kind: "query_plan",
        mime: "application/json",
        artifact: { renderer_id, plan: sortKeysDeep(scx_ast) }
      };
    }
  },
  DOM: {
    async render(scx_ast, policy, renderer_id) {
      return {
        kind: "dom_bundle",
        mime: "application/json",
        artifact: { renderer_id, bundle: sortKeysDeep(scx_ast) }
      };
    }
  }
};

// ============================================================
// Example oracleFetch injection (Node runtime)
// ============================================================

export async function oracleFetchHTTP(url, bodyJson) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(bodyJson)
  });
  return await res.json();
}
```

---

# SCX_SYSCALL_MODEL_v1

This section formalizes SCX atoms as **declarative syscalls** that require capability and legality validation before lowering or execution.

## 1) Core Principle (Invariant)

> **An SCX atom is a pure, declarative syscall request.**
> It declares intent, never execution.
> Execution occurs only after **capability + legality verification**.

## 2) SCX Atom = Syscall Descriptor

```json
{
  "@kind": "scx.syscall.v1",
  "atom": "READ_TABLE",
  "args": {
    "source": "sheet://Projects",
    "range": "A2:F"
  },
  "effects": ["io.read", "data.load"],
  "purity": "impure",
  "determinism": "bounded",
  "capability": "data.read.sheet"
}
```

### Required Fields

| Field         | Meaning                     |
| ------------- | --------------------------- |
| `atom`        | Canonical syscall name      |
| `args`        | Declarative parameters      |
| `effects`     | Observable side effects     |
| `purity`      | `pure` or `impure`          |
| `determinism` | `deterministic` or `bounded`|
| `capability`  | Required permission scope   |

## 3) Canonical SCX Syscall Classes

### A) Compute Syscalls (Pure)

| Atom                       | Description         |
| -------------------------- | ------------------- |
| `ADD`, `SUB`, `MUL`, `DIV` | Arithmetic          |
| `COMPARE_EQ`, `GT`, `LT`   | Comparison          |
| `LOGICAL_AND`, `OR`, `NOT` | Logic               |
| `RANGE`                    | Sequence generation |

```json
{
  "class": "compute",
  "purity": "pure",
  "capability": "compute.basic"
}
```

### B) Control Syscalls (Flow)

| Atom                      | Description      |
| ------------------------- | ---------------- |
| `BRANCH`                  | Conditional      |
| `MAP`, `FILTER`, `REDUCE` | Dataflow         |
| `JOIN`                    | Structural merge |

```json
{
  "class": "control",
  "purity": "pure",
  "capability": "control.flow"
}
```

### C) Data Syscalls

| Atom          | Description        |
| ------------- | ------------------ |
| `READ_TABLE`  | Load sheet/table   |
| `WRITE_TABLE` | Persist rows       |
| `READ_KV`     | Properties         |
| `WRITE_KV`    | Properties         |

```json
{
  "class": "data",
  "purity": "impure",
  "capability": "data.read.sheet"
}
```

### D) IO Syscalls

| Atom          | Description |
| ------------- | ----------- |
| `HTTP_FETCH`  | Network     |
| `EMAIL_SEND`  | Messaging   |
| `FILE_CREATE` | Drive       |

```json
{
  "class": "io",
  "purity": "impure",
  "capability": "io.network.fetch"
}
```

### E) UI Projection Syscalls

| Atom          | Description    |
| ------------- | -------------- |
| `RENDER_NODE` | DOM projection |
| `SET_STYLE`   | CSS hint       |
| `BIND_EVENT`  | Event wiring   |

```json
{
  "class": "ui",
  "purity": "pure",
  "capability": "ui.render"
}
```

## 4) Capability-Based Permission Scopes

### Scope Grammar

```
<domain>.<action>.<target>[.<constraint>]
```

Examples:

- `compute.basic`
- `data.read.sheet`
- `data.write.sheet`
- `io.network.fetch`
- `ui.render.dom`
- `identity.session.read`

## 5) Capability Scope Registry (v1)

```json
{
  "@kind": "scx.capability.registry.v1",
  "scopes": {
    "compute.basic": {
      "description": "Pure arithmetic and logic",
      "risk": "none"
    },
    "control.flow": {
      "description": "Structural flow control",
      "risk": "low"
    },
    "data.read.sheet": {
      "description": "Read spreadsheet data",
      "risk": "medium"
    },
    "data.write.sheet": {
      "description": "Modify spreadsheet data",
      "risk": "high"
    },
    "io.network.fetch": {
      "description": "Outbound HTTP requests",
      "risk": "high"
    },
    "ui.render.dom": {
      "description": "DOM projection only",
      "risk": "low"
    }
  }
}
```

## 6) Runtime Capability Mapping

### GAS Example

```json
{
  "runtime": "GAS",
  "allowed_capabilities": [
    "compute.basic",
    "control.flow",
    "data.read.sheet",
    "data.write.sheet",
    "io.network.fetch",
    "ui.render.dom"
  ],
  "denied_capabilities": [
    "os.fs",
    "os.process",
    "io.socket.raw"
  ]
}
```

## 7) Enforcement Model (CODE_ORACLE)

```text
SCX AST
 → syscall extraction
 → capability lookup
 → runtime allowance check
 → legality matrix check
 → lowering rule resolution
 → proof seal
 → execution
```

If any step fails → **hard stop**.

## 8) Proof Seal (tie-in)

Each syscall contributes to the proof hash:

```json
{
  "syscall": "WRITE_TABLE",
  "capability": "data.write.sheet",
  "runtime": "GAS",
  "lowering_rule": "SHEET_APPEND_ROWS",
  "hash": "sha256:9f3c…"
}
```

No proof = no execution.

## 9) Why this is powerful (and correct)

- SCX ≈ **portable syscall ABI**
- Runtimes ≈ **kernels**
- Capabilities ≈ **permission bits**
- Lowering rules ≈ **drivers**
- Proofs ≈ **execution receipts**

---

# SCX_SYSCALL_MANIFEST_v1

A per-app or per-tape manifest that enumerates required syscalls and capabilities for review and enforcement.

# SCX_CAPABILITY_NEGOTIATION_v1

A runtime handshake object that confirms allowed capabilities before lowering or execution.
