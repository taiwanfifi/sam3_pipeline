# GUIDE 3: Feature Extraction & Caching

`extract.py` pre-computes prompt features so that inference (`infer.py` / `infer_multi.py`)
only needs the vision-encoder + decoder — no text or geometry encoder at runtime.

## Workflow

```
config.json ──> extract.py ──> features/{class_name}/ ──> infer.py
 (classes)       (once)         features.npy, mask.npy     (per frame)
```

Run `extract.py` **once** after any config change. Inference reads the cached `.npy` files.

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/extract.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json
```

## What It Loads

Extract loads **all three** TRT engines from the `engines` path in config.json:

| Engine | Used for | Size (typical) |
|--------|----------|----------------|
| text-encoder.engine | Text prompts → 32-token embeddings | ~680 MB |
| vision-encoder.engine | Reference images → FPN feature maps | ~875 MB |
| geometry-encoder.engine | Bounding boxes → geometry embeddings | ~18 MB |

Resolution is **auto-detected** from the vision-encoder engine profile shape.
You never need to specify it manually.

## Output: `features/` Directory

```
features/
├── person/
│   ├── features.npy     # [1, prompt_len, 256] float32
│   ├── mask.npy          # [1, prompt_len] bool
│   └── meta.json         # {"hash": "...", "prompt_len": 32, "prompt_type": "text", ...}
├── hand/
│   ├── features.npy
│   ├── mask.npy
│   └── meta.json
├── screwdriver/          # "both" type: text + geometry concatenated
│   ├── features.npy      # [1, 34, 256] (32 text + 2 geo)
│   ├── mask.npy
│   └── meta.json
└── _meta.json            # {"max_prompt_len": 34, "num_classes": 4, "class_names": [...]}
```

All classes are padded to the same `max_prompt_len` so the decoder can batch them.

**File ownership**: These files are created by Docker (root user). You cannot delete
them from the host directly — use `docker exec` instead (see [Cache Invalidation](#cache-invalidation)).

## Caching Mechanism

Each class has a **content hash** stored in `meta.json`. On the next run, extract.py
compares the hash — if it matches, the class is skipped ("cached").

### What the hash includes

```python
hash = SHA256(
    prompt_type,              # "text", "image", or "both"
    text,                     # the text string (if any)
    reference_image_bytes,    # raw file content of each reference image
    boxes,                    # bounding box coordinates (JSON)
    labels,                   # positive/negative labels (JSON)
)
```

### What the hash does NOT include

- **Engine resolution** — not part of the hash
- **Engine variant** (Q50, Q200, INT8, etc.) — not part of the hash
- Class name — not part of the hash (only used for directory naming)

This means: if you only change the `engines` path in config.json (e.g. switch from
`b8_q50_r560` to `b8_q50_r448`), the hash stays the same and extract.py will
report all classes as "cached".

## Resolution Sensitivity by Prompt Type

This is the critical point for switching engines:

| Prompt type | Resolution-dependent? | Safe to cache across resolutions? |
|-------------|----------------------|-----------------------------------|
| `text` | No | Yes — text encoder has no resolution input |
| `image` | **Yes** | **No** — geometry encoder uses vision-encoder FPN outputs |
| `both` | **Yes** (geometry part) | **No** — the geometry half depends on resolution |

### Why geometry features are resolution-dependent

The `image` and `both` prompt types go through this pipeline:

```
reference image ──> vision-encoder (resolution-dependent!) ──> FPN feature maps
                                                                    │
bounding boxes ──> geometry-encoder <───────────────────────────────┘
                        │
                   geometry features [1, N+1, 256]
```

The vision-encoder resizes the reference image to the engine's resolution (448, 560, 672,
840, or 1008) and produces FPN feature maps at that scale. The geometry-encoder then
processes bounding boxes relative to those feature maps. Different resolutions produce
different FPN outputs, so the geometry features are **not interchangeable** across resolutions.

Text features only go through the text-encoder, which has no image/resolution input,
so they are always valid regardless of which engine resolution you use.

## Switching Engine Resolution

### If all classes are `text` type only

Just change the `engines` path in config.json and run extract. Everything will be cached
and you can immediately run inference. No extra steps needed.

```bash
# Edit config.json: "engines": "engines/b8_q50_r560"  →  "engines/b8_q50_r448"
# Run extract (all cached, instant)
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/extract.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json
```

### If any class uses `image` or `both` type

You must **manually invalidate the cache** for those classes before running extract.
Otherwise the stale geometry features (computed at the old resolution) will be used
with the new resolution engine, producing incorrect results.

```bash
# 1. Change engines path in config.json
# 2. Delete meta.json for geometry-dependent classes
docker exec william_tensorrt rm \
  /root/VisionDSL/models/sam3_pipeline/features/screwdriver/meta.json

# 3. Re-run extract (text classes cached, geometry classes re-extracted)
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/extract.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json
```

## Cache Invalidation

### When extract.py automatically re-extracts

- Class name changed (new directory, no cached files)
- Prompt type changed (`"text"` → `"both"`)
- Text content changed (`"person"` → `"people"`)
- Reference image file changed (different bytes)
- Bounding box coordinates changed
- Labels changed
- `meta.json` missing or deleted

### When you must manually invalidate

- **Switched engine resolution** with `image` or `both` classes
- You suspect corrupted features

### How to invalidate

Delete the `meta.json` for the affected class(es) via Docker:

```bash
# Single class
docker exec william_tensorrt rm \
  /root/VisionDSL/models/sam3_pipeline/features/{class_name}/meta.json

# All classes (nuclear option)
docker exec william_tensorrt bash -c \
  'rm /root/VisionDSL/models/sam3_pipeline/features/*/meta.json'
```

**Why Docker?** The `features/` directory is created by the Docker container running
as root. Host-side `rm` will fail with "Permission denied". Always use `docker exec`.

## Bug Fix: Hardcoded Resolution (2026-02-19)

**Problem**: `encode_image()` originally had a hardcoded `resize((1008, 1008))`:

```python
# BEFORE (broken for non-1008 engines)
resized = np.array(PILImage.fromarray(rgb).resize((1008, 1008), PILImage.BILINEAR))
```

When using the 448 engine, the vision-encoder expected 448x448 input but received
1008x1008, causing a TensorRT shape mismatch error:

```
IExecutionContext::setInputShape: Error Code 3: API Usage Error
(Static dimension mismatch while setting input shape for images.
 Set dimensions are [1,3,1008,1008]. Expected dimensions are [-1,3,448,448].)
```

**Fix**: Auto-detect resolution from the engine's optimization profile:

```python
# AFTER (works with any resolution)
ve_min = self.vision_engine.get_tensor_profile_shape("images", 0)[0]
self.image_size = ve_min[2]   # e.g. 448, 560, 672, 840, 1008

S = self.image_size
resized = np.array(PILImage.fromarray(rgb).resize((S, S), PILImage.BILINEAR))
```

This fix ensures extract.py works correctly with any engine resolution without
manual configuration.

## Stale Class Warning

If the `features/` directory contains subdirectories for classes no longer in
config.json, extract.py prints a warning:

```
WARNING: features/blow_gun/ is not in config — stale?
```

These stale directories don't cause errors but waste disk space. To clean up:

```bash
docker exec william_tensorrt rm -rf \
  /root/VisionDSL/models/sam3_pipeline/features/blow_gun
```

## Quick Reference: Resolution Switch Checklist

1. Edit `config.json` → change `"engines"` path
2. If any class is `image` or `both`:
   - `docker exec william_tensorrt rm .../features/{class}/meta.json`
3. Run `extract.py`
4. Run `infer.py` or `infer_multi.py`
