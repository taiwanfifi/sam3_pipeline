"""
SAM3 Prompt Feature Extraction

Pre-computes prompt features for each class in config.json, saves to features/ dir.
At inference time only vision-encoder + decoder are needed — no text/geo encoder.

Prompt types:
  text:  text encoder  -> [1, 32, 256] features + [1, 32] mask
  image: vision encoder + geometry encoder -> [1, N+1, 256] features + [1, N+1] mask
  both:  concatenate text + geometry on axis=1

All classes are padded to a uniform prompt_len for batched decoding.

Usage (inside sam3_trt container):
  python3 extract.py --config config.json
"""
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path

import cv2
from PIL import Image as PILImage
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

LOG = trt.Logger(trt.Logger.WARNING)
MAX_CLASSES = 4


# ---------------------------------------------------------------------------
# TensorRT helpers
# ---------------------------------------------------------------------------

def load_engine(path: str):
    with open(path, "rb") as f:
        return trt.Runtime(LOG).deserialize_cuda_engine(f.read())


def make_buffers(engine):
    """Allocate host + device buffers for every I/O tensor (dynamic dims → 1)."""
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = [max(d, 1) for d in engine.get_tensor_shape(name)]
        np_dtype = trt.nptype(engine.get_tensor_dtype(name))
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        n = max(int(np.prod(shape)), 1)
        host = cuda.pagelocked_empty(n, np_dtype)
        gpu = cuda.mem_alloc(host.nbytes)
        buf = {"name": name, "host": host, "gpu": gpu,
               "shape": tuple(shape), "dtype": np_dtype, "capacity": n}
        (inputs if is_input else outputs).append(buf)
    return inputs, outputs


def fill(buf, arr: np.ndarray):
    """Copy ndarray into a pagelocked buffer, growing if needed."""
    flat = arr.astype(buf["dtype"]).ravel()
    n = flat.size
    if n > buf["capacity"]:
        buf["gpu"].free()
        cap = int(n * 1.5)
        buf["host"] = cuda.pagelocked_empty(cap, buf["dtype"])
        buf["gpu"] = cuda.mem_alloc(buf["host"].nbytes)
        buf["capacity"] = cap
    np.copyto(buf["host"][:n], flat)


def execute(ctx, inputs, outputs, stream):
    """Set shapes → H2D → execute → D2H → sync → return numpy list."""
    for b in inputs:
        ctx.set_input_shape(b["name"], b["shape"])
        ctx.set_tensor_address(b["name"], b["gpu"])
        cuda.memcpy_htod_async(b["gpu"], b["host"], stream)
    for b in outputs:
        ctx.set_tensor_address(b["name"], b["gpu"])
    ctx.execute_async_v3(stream_handle=stream.handle)
    for b in outputs:
        b["shape"] = tuple(ctx.get_tensor_shape(b["name"]))
    for b in outputs:
        cuda.memcpy_dtoh_async(b["host"], b["gpu"], stream)
    stream.synchronize()
    return [b["host"][:int(np.prod(b["shape"]))].reshape(b["shape"]) for b in outputs]


# ---------------------------------------------------------------------------
# Content hash for caching
# ---------------------------------------------------------------------------

def content_hash(cls_cfg: dict, base_dir: Path) -> str:
    h = hashlib.sha256()
    h.update(cls_cfg["prompt_type"].encode())
    if cls_cfg.get("text"):
        h.update(cls_cfg["text"].encode())
    for ref in cls_cfg.get("references", []):
        p = base_dir / ref["image"]
        if p.exists():
            h.update(p.read_bytes())
        h.update(json.dumps(ref.get("boxes", []), sort_keys=True).encode())
        h.update(json.dumps(ref.get("labels", []), sort_keys=True).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Extractor: loads 3 TRT engines, produces per-class prompt features
# ---------------------------------------------------------------------------

class Extractor:
    def __init__(self, engine_dir: Path, tokenizer_path: str):
        trt.init_libnvinfer_plugins(LOG, "")
        self.stream = cuda.Stream()

        print("Loading text-encoder...")
        self.text_engine = load_engine(str(engine_dir / "text-encoder.engine"))
        self.text_ctx = self.text_engine.create_execution_context()
        self.text_in, self.text_out = make_buffers(self.text_engine)

        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)

        print("Loading vision-encoder...")
        self.vision_engine = load_engine(str(engine_dir / "vision-encoder.engine"))
        self.vision_ctx = self.vision_engine.create_execution_context()
        self.vision_in, self.vision_out = make_buffers(self.vision_engine)

        print("Loading geometry-encoder...")
        self.geo_engine = load_engine(str(engine_dir / "geometry-encoder.engine"))
        self.geo_ctx = self.geo_engine.create_execution_context()
        self.geo_in, self.geo_out = make_buffers(self.geo_engine)

        print("All engines loaded.\n")

    # -- text --
    def encode_text(self, text: str):
        """-> (features [1,32,256] f32, mask [1,32] bool)"""
        enc = self.tokenizer.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        attn = np.array([enc.attention_mask], dtype=np.int64)

        fill(self.text_in[0], ids)
        self.text_in[0]["shape"] = ids.shape
        fill(self.text_in[1], attn)
        self.text_in[1]["shape"] = attn.shape

        out = execute(self.text_ctx, self.text_in, self.text_out, self.stream)
        return out[0].astype(np.float32), out[1].astype(np.bool_)

    # -- vision --
    def encode_image(self, image_bgr: np.ndarray):
        """-> dict with fpn_feat_0/1/2, fpn_pos_2"""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = np.array(PILImage.fromarray(rgb).resize((1008, 1008), PILImage.BILINEAR))
        tensor = (resized.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[np.newaxis]

        fill(self.vision_in[0], tensor)
        self.vision_in[0]["shape"] = tensor.shape
        out = execute(self.vision_ctx, self.vision_in, self.vision_out, self.stream)
        return {"fpn_feat_0": out[0], "fpn_feat_1": out[1],
                "fpn_feat_2": out[2], "fpn_pos_2": out[3]}

    # -- geometry --
    def encode_geometry(self, boxes, labels, fpn_feat_2, fpn_pos_2):
        """-> (features [1,N+1,256] f32, mask [1,N+1] bool)

        boxes:  [1, N, 4] f32  (cxcywh normalised)
        labels: [1, N] i64     (1=pos, 0=neg)
        """
        fill(self.geo_in[0], boxes.astype(np.float32))
        self.geo_in[0]["shape"] = boxes.shape
        fill(self.geo_in[1], labels.astype(np.int64))
        self.geo_in[1]["shape"] = labels.shape
        fill(self.geo_in[2], fpn_feat_2)
        self.geo_in[2]["shape"] = fpn_feat_2.shape
        fill(self.geo_in[3], fpn_pos_2)
        self.geo_in[3]["shape"] = fpn_pos_2.shape

        out = execute(self.geo_ctx, self.geo_in, self.geo_out, self.stream)
        return out[0].astype(np.float32), out[1].astype(np.bool_)


# ---------------------------------------------------------------------------
# Per-class feature extraction
# ---------------------------------------------------------------------------

def extract_class(ext: Extractor, cfg: dict, base_dir: Path):
    """-> (features [1, L, 256], mask [1, L] bool)"""
    kind = cfg["prompt_type"]

    if kind == "text":
        feat, mask = ext.encode_text(cfg["text"])
        print(f"    text  -> {feat.shape}")
        return feat, mask

    if kind == "image":
        feat, mask = _geo_features(ext, cfg, base_dir)
        print(f"    image -> {feat.shape}")
        return feat, mask

    if kind == "both":
        tf, tm = ext.encode_text(cfg["text"])
        gf, gm = _geo_features(ext, cfg, base_dir)
        feat = np.concatenate([tf, gf], axis=1)
        mask = np.concatenate([tm, gm], axis=1)
        print(f"    both  -> text {tf.shape[1]} + geo {gf.shape[1]} = {feat.shape}")
        return feat, mask

    raise ValueError(f"Unknown prompt_type: {kind}")


def _geo_features(ext: Extractor, cfg: dict, base_dir: Path):
    """Encode reference images -> concatenated geometry features."""
    parts_f, parts_m = [], []

    for ref in cfg["references"]:
        img = cv2.imread(str(base_dir / ref["image"]))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {base_dir / ref['image']}")

        vision = ext.encode_image(img)
        boxes = np.array(ref["boxes"], dtype=np.float32)[np.newaxis]   # [1,N,4]
        labels = np.array([ref["labels"]], dtype=np.int64)             # [1,N]
        gf, gm = ext.encode_geometry(boxes, labels,
                                     vision["fpn_feat_2"], vision["fpn_pos_2"])
        parts_f.append(gf)
        parts_m.append(gm)

    if len(parts_f) == 1:
        return parts_f[0], parts_m[0]

    # Multiple refs: keep first fully, skip CLS token from subsequent
    merged_f = [parts_f[0]] + [f[:, 1:, :] for f in parts_f[1:]]
    merged_m = [parts_m[0]] + [m[:, 1:]    for m in parts_m[1:]]
    return np.concatenate(merged_f, axis=1), np.concatenate(merged_m, axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 prompt feature extraction")
    parser.add_argument("--config", required=True, help="Path to config.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    base = config_path.parent
    cfg = json.loads(config_path.read_text())

    engine_dir = (base / cfg["engines"]).resolve()
    tokenizer  = str((base / cfg["tokenizer"]).resolve())
    features_dir = base / cfg["features"]
    features_dir.mkdir(parents=True, exist_ok=True)

    classes = cfg["classes"]
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"Max {MAX_CLASSES} classes supported (got {len(classes)})")

    # Warn about stale class directories
    configured_names = {c["name"] for c in classes}
    for child in features_dir.iterdir():
        if child.is_dir() and child.name not in configured_names:
            print(f"  WARNING: features/{child.name}/ is not in config — stale?")

    ext = Extractor(engine_dir, tokenizer)

    all_feat, all_mask = [], []
    max_len = 0

    for c in classes:
        name = c["name"]
        cls_dir = features_dir / name
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Cache check
        h = content_hash(c, base)
        meta_path = cls_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("hash") == h:
                feat = np.load(cls_dir / "features.npy")
                mask = np.load(cls_dir / "mask.npy")
                print(f"  {name}: cached (len={feat.shape[1]})")
                all_feat.append(feat)
                all_mask.append(mask)
                max_len = max(max_len, feat.shape[1])
                continue

        print(f"  {name}: extracting ({c['prompt_type']})...")
        feat, mask = extract_class(ext, c, base)
        L = feat.shape[1]
        max_len = max(max_len, L)

        np.save(cls_dir / "features.npy", feat)
        np.save(cls_dir / "mask.npy", mask)
        meta_path.write_text(json.dumps({
            "hash": h,
            "prompt_len": L,
            "prompt_type": c["prompt_type"],
            "source": c.get("text", ""),
        }, indent=2))

        all_feat.append(feat)
        all_mask.append(mask)
        print(f"    saved -> {cls_dir}  (len={L})")

    # Pad to uniform length
    print(f"\nPadding to max_len={max_len} ...")
    for i, c in enumerate(classes):
        cls_dir = features_dir / c["name"]
        cur = all_feat[i].shape[1]
        if cur < max_len:
            pad = max_len - cur
            all_feat[i] = np.pad(all_feat[i], ((0,0),(0,pad),(0,0)))
            all_mask[i] = np.pad(all_mask[i], ((0,0),(0,pad)), constant_values=False)
            np.save(cls_dir / "features.npy", all_feat[i])
            np.save(cls_dir / "mask.npy", all_mask[i])
            print(f"  {c['name']}: {cur} -> {max_len}")
        else:
            print(f"  {c['name']}: {max_len} (ok)")

    # Pipeline metadata
    meta = {"max_prompt_len": max_len, "num_classes": len(classes),
            "class_names": [c["name"] for c in classes]}
    (features_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. {len(classes)} classes, max_prompt_len={max_len}")


if __name__ == "__main__":
    main()
