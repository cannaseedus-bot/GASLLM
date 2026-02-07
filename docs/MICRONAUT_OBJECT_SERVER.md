# Micronaut Object Server (SCO/1)

## Canonical layout

```
micronaut/
├─ micronaut.s7               # SCO/1 executable object
├─ object.toml                # Object server declaration
├─ semantics.xjson            # KUHUL-TSG schema
├─ brains/
│  ├─ trigrams.json           # sealed
│  ├─ bigrams.json            # sealed
│  └─ meta-intent-map.json    # sealed
├─ io/
│  ├─ chat.txt                # append-only input
│  ├─ stream.txt              # token-like output
│  └─ snapshot/
├─ trace/
│  └─ scxq2.trace
└─ proof/
   └─ scxq2.proof
```

Nothing executes outside this boundary.

## chat.txt record grammar (frozen)

```
--- MESSAGE ---
id: <uuid>
time: <unix_ms>
role: user | system | micronaut
intent: chat | generate | classify | complete
context: <optional>
payload:
<UTF-8 text>
--- END ---
```

Rules:

- Append-only, immutable.
- Order is truth.
- No partial writes (atomic append only).
- CM-1 must pass on payload.

## stream.txt semantic emission (frozen)

```
>> t=184 ctx=@π mass=0.73
Hello!
```

Rules:

- Append-only.
- Ordered.
- Replayable.
- May be truncated safely.
- Projection only (no authority).

## REST ↔ file mapping

| REST Endpoint    | File Action         |
| ---------------- | ------------------- |
| `POST /chat`     | append → `chat.txt` |
| `GET /stream`    | read → `stream.txt` |
| `GET /status`    | read → object state |
| `POST /snapshot` | rotate snapshot     |

REST is a local file router only.

## Process lifecycle

```
INIT → READY → RUNNING → IDLE → HALT
```

| Event         | Action     |
| ------------- | ---------- |
| file append   | wake       |
| collapse done | idle       |
| error         | halt       |
| shutdown      | seal trace |
