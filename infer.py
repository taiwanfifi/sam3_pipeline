"""
SAM3 Unified Inference Pipeline

GPU-batched multi-class detection with pixel-level masks.
Supports mixed text + image prompts (pre-computed by extract.py).
Only loads vision-encoder + decoder at runtime.

Usage (inside sam3_trt container):
  # Single image
  python3 infer.py --config config.json \
    --images Inputs/demo_3.jpg --output outputs

  # Video with frame sampling
  python3 infer.py --config config.json \
    --video Inputs/media1.mp4 --output outputs --interval 30

  # Include per-class mask PNGs
  python3 infer.py --config config.json \
    --images Inputs/demo_3.jpg --output outputs --masks
"""
import cv2
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

LOG = trt.Logger(trt.Logger.WARNING)
MAX_CLASSES = 4
IMAGE_SIZE = 1008

FPN_SHAPES = [
    (1, 256, 288, 288),
    (1, 256, 144, 144),
    (1, 256, 72, 72),
    (1, 256, 72, 72),
]
FPN_NAMES = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]
OUT_NAMES = ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]

MASK_H, MASK_W = 288, 288
MASK_BYTES = MASK_H * MASK_W * 4
QUERIES = 200


def load_engine(path: str):
    with open(path, "rb") as f:
        return trt.Runtime(LOG).deserialize_cuda_engine(f.read())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """GPU-batched multi-class SAM3 inference with pixel masks."""

    def __init__(self, config_path: str):
        cfg_path = Path(config_path)
        base = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())

        trt.init_libnvinfer_plugins(LOG, "")
        engine_dir = (base / cfg["engines"]).resolve()
        features_dir = base / cfg["features"]
        self.confidence = cfg.get("confidence", 0.3)

        # Class list
        self.classes = [c["name"] for c in cfg["classes"]]
        N = len(self.classes)
        if N > MAX_CLASSES:
            raise ValueError(f"Max {MAX_CLASSES} classes supported (got {N})")
        self.N = N

        # Load prompt metadata
        meta = json.loads((features_dir / "_meta.json").read_text())
        self.P = meta["max_prompt_len"]  # prompt length

        print(f"Pipeline: {N} classes, prompt_len={self.P}")

        # Load pre-computed prompt features
        feats, masks = [], []
        for c in cfg["classes"]:
            d = features_dir / c["name"]
            f = np.load(d / "features.npy").astype(np.float32)
            m = np.load(d / "mask.npy").astype(np.bool_)
            assert f.shape == (1, self.P, 256), f"{c['name']}: shape {f.shape}"
            feats.append(f[0])
            masks.append(m[0])
            print(f"  {c['name']}: loaded ({c.get('prompt_type', 'text')})")

        stacked_feat = np.stack(feats)   # [N, P, 256]
        stacked_mask = np.stack(masks)   # [N, P]

        self.stream = cuda.Stream()

        # --- Vision encoder ---
        print("Loading vision-encoder...")
        self.vision_engine = load_engine(str(engine_dir / "vision-encoder.engine"))
        self.vision_ctx = self.vision_engine.create_execution_context()

        img_elems = 3 * IMAGE_SIZE * IMAGE_SIZE
        self.img_host = cuda.pagelocked_empty(img_elems, np.float32)
        self.img_gpu = cuda.mem_alloc(self.img_host.nbytes)
        self.fpn_gpu = [cuda.mem_alloc(int(np.prod(s)) * 4) for s in FPN_SHAPES]

        # --- Decoder (batched) ---
        print("Loading decoder...")
        self.decoder_engine = load_engine(str(engine_dir / "decoder.engine"))
        self.decoder_ctx = self.decoder_engine.create_execution_context()

        # Batched FPN buffers (N copies)
        self.batch_fpn_gpu = [cuda.mem_alloc(N * int(np.prod(s)) * 4) for s in FPN_SHAPES]

        # Prompt buffers on GPU
        pf_bytes = N * self.P * 256 * 4
        pm_bytes = N * self.P  # bool = 1 byte
        self.prompt_feat_gpu = cuda.mem_alloc(pf_bytes)
        self.prompt_mask_gpu = cuda.mem_alloc(pm_bytes)

        # One-time host → GPU copy of prompt features
        pf_host = cuda.pagelocked_empty(N * self.P * 256, np.float32)
        np.copyto(pf_host, stacked_feat.ravel())
        cuda.memcpy_htod_async(self.prompt_feat_gpu, pf_host, self.stream)

        pm_host = cuda.pagelocked_empty(N * self.P, np.bool_)
        np.copyto(pm_host, stacked_mask.ravel())
        cuda.memcpy_htod_async(self.prompt_mask_gpu, pm_host, self.stream)

        # Output buffers
        self.out_shapes = [
            (N, QUERIES, MASK_H, MASK_W),  # pred_masks
            (N, QUERIES, 4),                # pred_boxes
            (N, QUERIES),                   # pred_logits
            (N, 1),                         # presence_logits
        ]
        self.out_gpu = [cuda.mem_alloc(int(np.prod(s)) * 4) for s in self.out_shapes]
        self.out_host = [
            cuda.pagelocked_empty(int(np.prod(s)), np.float32) for s in self.out_shapes[1:]
        ]
        self.mask_host = cuda.pagelocked_empty(MASK_H * MASK_W, np.float32)

        self.stream.synchronize()
        del pf_host, pm_host

        # VRAM estimate (GPU allocations only)
        vram = img_elems * 4  # img_gpu
        for s in FPN_SHAPES:
            vram += int(np.prod(s)) * 4      # single FPN
            vram += N * int(np.prod(s)) * 4  # batched FPN
        vram += pf_bytes + pm_bytes
        for s in self.out_shapes:
            vram += int(np.prod(s)) * 4
        self.vram_mb = vram / 1024 / 1024

        print(f"Ready: {N} classes, prompt_len={self.P}, buffers ~{self.vram_mb:.0f} MB\n")

    # ----- core inference -----

    def infer(self, image_bgr: np.ndarray):
        """Run batched inference. Returns (boxes, logits, presence, hw).
        Masks stay on GPU — call copy_mask() selectively."""
        hw = image_bgr.shape[:2]
        N, P = self.N, self.P

        # Preprocess
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = np.array(PILImage.fromarray(rgb).resize(
            (IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR))
        tensor = (resized.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        np.copyto(self.img_host, tensor.ravel())
        cuda.memcpy_htod_async(self.img_gpu, self.img_host, self.stream)

        # Vision encoder
        self.vision_ctx.set_input_shape("images", (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.vision_ctx.set_tensor_address("images", self.img_gpu)
        for i, name in enumerate(FPN_NAMES):
            self.vision_ctx.set_tensor_address(name, self.fpn_gpu[i])
        self.vision_ctx.execute_async_v3(stream_handle=self.stream.handle)

        # Replicate FPN on GPU (d2d copy, zero CPU involvement)
        for fi in range(4):
            one = int(np.prod(FPN_SHAPES[fi])) * 4
            cuda.memcpy_dtod_async(self.batch_fpn_gpu[fi], self.fpn_gpu[fi], one, self.stream)
            for bi in range(1, N):
                cuda.memcpy_dtod_async(
                    int(self.batch_fpn_gpu[fi]) + bi * one,
                    self.fpn_gpu[fi], one, self.stream)

        # Decoder: batch=N, prompt_len=P
        for i, name in enumerate(FPN_NAMES):
            self.decoder_ctx.set_input_shape(name, (N, *FPN_SHAPES[i][1:]))
            self.decoder_ctx.set_tensor_address(name, self.batch_fpn_gpu[i])
        self.decoder_ctx.set_input_shape("prompt_features", (N, P, 256))
        self.decoder_ctx.set_tensor_address("prompt_features", self.prompt_feat_gpu)
        self.decoder_ctx.set_input_shape("prompt_mask", (N, P))
        self.decoder_ctx.set_tensor_address("prompt_mask", self.prompt_mask_gpu)
        for i, name in enumerate(OUT_NAMES):
            self.decoder_ctx.set_tensor_address(name, self.out_gpu[i])

        self.decoder_ctx.execute_async_v3(stream_handle=self.stream.handle)

        # Copy back boxes/logits/presence (small); masks stay on GPU
        # out_gpu: [masks, boxes, logits, presence] — skip index 0 (masks)
        # out_host: [boxes, logits, presence] — aligned with out_gpu[1:]
        for i in range(len(self.out_host)):
            n = int(np.prod(self.out_shapes[i + 1]))
            cuda.memcpy_dtoh_async(self.out_host[i][:n], self.out_gpu[i + 1], self.stream)

        self.stream.synchronize()

        boxes    = self.out_host[0][:N*QUERIES*4].reshape(N, QUERIES, 4)
        logits   = self.out_host[1][:N*QUERIES].reshape(N, QUERIES)
        presence = self.out_host[2][:N].reshape(N, 1)
        return boxes, logits, presence, hw

    def copy_mask(self, cls_idx: int, query_idx: int, hw: tuple) -> np.ndarray:
        """Selectively copy one mask from GPU → CPU, resize to original res."""
        h, w = hw
        offset = (cls_idx * QUERIES + query_idx) * MASK_BYTES
        cuda.memcpy_dtoh(self.mask_host, int(self.out_gpu[0]) + offset)
        raw = self.mask_host.reshape(MASK_H, MASK_W)
        return cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR) > 0

    def postprocess(self, boxes, logits, presence, hw, conf=None):
        """Convert raw outputs → list of detections with masks."""
        if conf is None:
            conf = self.confidence
        h, w = hw
        dets = []

        for i, cls in enumerate(self.classes):
            p = 1.0 / (1.0 + np.exp(-presence[i, 0]))
            scores = (1.0 / (1.0 + np.exp(-logits[i]))) * p
            keep = np.where(scores > conf)[0]

            for j in keep:
                box = boxes[i, j].copy()
                box[[0, 2]] *= w
                box[[1, 3]] *= h
                box = np.clip(box, 0, [w, h, w, h])
                mask = self.copy_mask(i, int(j), hw)
                dets.append({"class": cls, "box": box.tolist(),
                             "score": float(scores[j]), "mask": mask})

        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets

    def detect(self, image_bgr: np.ndarray, conf: float = None) -> list:
        """Full pipeline: preprocess → infer → postprocess."""
        boxes, logits, presence, hw = self.infer(image_bgr)
        return self.postprocess(boxes, logits, presence, hw, conf)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

PALETTE = [
    (30, 144, 255), (255, 144, 30), (144, 255, 30), (255, 30, 144),
    (255, 200, 50), (30, 255, 144), (255, 255, 30), (30, 255, 255),
]


def draw_overlay(image_bgr, dets, alpha=0.45):
    """Draw semi-transparent masks + contours + labels."""
    vis = image_bgr.copy()
    if not dets:
        return vis

    # Assign colours per class
    cmap = {}
    for d in dets:
        if d["class"] not in cmap:
            cmap[d["class"]] = PALETTE[len(cmap) % len(PALETTE)]

    for d in reversed(dets):
        c = np.array(cmap[d["class"]], dtype=np.float32)
        m = d["mask"]
        vis[m] = (vis[m].astype(np.float32) * (1 - alpha) + c * alpha).astype(np.uint8)

    for d in reversed(dets):
        m = d["mask"].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, cmap[d["class"]], 2)

    for d in dets:
        x1, y1 = int(d["box"][0]), int(d["box"][1])
        c = cmap[d["class"]]
        label = f"{d['class']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), c, -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def save_masks(dets, hw, out_dir, name):
    """Save per-class binary mask PNGs."""
    h, w = hw
    by_class = {}
    for d in dets:
        c = d["class"]
        if c not in by_class:
            by_class[c] = np.zeros((h, w), dtype=np.uint8)
        by_class[c][d["mask"]] = 255
    for c, mask in by_class.items():
        cv2.imwrite(str(out_dir / f"{name}_mask_{c}.png"), mask)


def timestamp(frame_idx: int, fps: float) -> str:
    ms = int(frame_idx / fps * 1000)
    return f"{ms // 60000:02d}m{ms % 60000 // 1000:02d}s{ms % 1000:03d}ms"


def run_tag():
    """Timestamp tag for output filenames (one per run)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def run_images(pipe, paths, out_dir, conf, with_masks, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{tag}_detections.jsonl"
    times = []

    with open(jsonl, "w") as f:
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"SKIP: {p}")
                continue

            t0 = time.time()
            dets = pipe.detect(img, conf)
            dt = time.time() - t0
            times.append(dt)

            stem = Path(p).stem
            print(f"{stem}: {len(dets)} dets, {dt*1000:.0f} ms")
            for d in dets:
                print(f"  {d['class']}: {d['score']:.3f}  {[int(x) for x in d['box']]}")

            cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), draw_overlay(img, dets))
            if with_masks and dets:
                save_masks(dets, img.shape[:2], out_dir, stem)

            row = {"frame": stem, "elapsed_ms": round(dt * 1000, 1),
                   "detections": [{"class": d["class"], "box": d["box"],
                                   "score": d["score"]} for d in dets]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    return times


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------

def run_video(pipe, video_path, out_dir, conf, with_masks, interval, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{tag}_detections.jsonl"
    avi   = out_dir / f"{tag}_output.avi"
    times = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return times

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps / interval
    print(f"Video: {total} frames, {fps:.1f} fps, interval={interval}")
    print(f"Output AVI: {avi.name} @ {out_fps:.1f} fps")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(avi), fourcc, out_fps, (w, h))

    idx = 0
    with open(jsonl, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval != 0:
                idx += 1
                continue
            idx += 1

            t0 = time.time()
            dets = pipe.detect(frame, conf)
            dt = time.time() - t0
            times.append(dt)

            frame_idx = idx - 1  # actual frame number (before increment)
            ts = timestamp(frame_idx, fps)
            print(f"[{ts}] #{frame_idx}: {len(dets)} dets, {dt*1000:.0f} ms")

            overlay = draw_overlay(frame, dets)
            writer.write(overlay)

            if with_masks and dets:
                save_masks(dets, frame.shape[:2], out_dir, ts)

            row = {"frame_idx": frame_idx, "timestamp": ts,
                   "elapsed_ms": round(dt * 1000, 1),
                   "detections": [{"class": d["class"], "box": d["box"],
                                   "score": d["score"]} for d in dets]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    writer.release()
    cap.release()
    print(f"\nProcessed {len(times)}/{total} frames")
    return times


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------

def write_performance(times, out_dir, pipe, tag):
    if not times:
        return
    ms = [t * 1000 for t in times]
    s = sorted(ms)
    p95 = s[max(0, int(len(s) * 0.95) - 1)]
    avg = float(np.mean(ms))
    perf = {
        "frames": len(times), "classes": pipe.N, "prompt_len": pipe.P,
        "avg_ms": round(avg, 1), "min_ms": round(min(ms), 1),
        "max_ms": round(max(ms), 1), "p95_ms": round(p95, 1),
        "est_fps": round(1000 / avg, 1), "buffer_vram_mb": round(pipe.vram_mb, 1),
    }
    path = out_dir / f"{tag}_performance.json"
    path.write_text(json.dumps(perf, indent=2))
    print(f"\nPerformance: avg {perf['avg_ms']:.0f} ms, p95 {perf['p95_ms']:.0f} ms, "
          f"~{perf['est_fps']:.1f} FPS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SAM3 unified inference pipeline")
    ap.add_argument("--config", required=True)

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+")
    src.add_argument("--video")

    ap.add_argument("--output", default="outputs", help="Output directory")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--interval", type=int, default=1, help="Frame interval (video)")
    ap.add_argument("--masks", action="store_true", help="Save per-class mask PNGs")
    args = ap.parse_args()

    out = Path(args.output)
    pipe = Pipeline(args.config)
    conf = args.conf if args.conf is not None else pipe.confidence
    tag = run_tag()

    if args.images:
        times = run_images(pipe, args.images, out, conf, args.masks, tag)
    else:
        times = run_video(pipe, args.video, out, conf, args.masks, args.interval, tag)

    write_performance(times, out, pipe, tag)


if __name__ == "__main__":
    main()
