"""
SAM3 Multi-Camera Inference Pipeline (Plan C v3)

8 cameras processed simultaneously:
  - VE batch=F: all 8 frames in one forward pass
  - Zero-copy FPN: VE output buffers directly used as decoder input
  - Decoder iterates per class (4x), each with batch=F (all frames)
  - Prompts pre-replicated on GPU at init (zero per-round copy)
  - Selective mask copy: only transfer detected masks (~40x less bandwidth)
  - Double-buffered decoder output: decoder N+1 overlaps mask copy N

Resolution and QUERIES are auto-detected from engine files — supports
1008, 840, or any resolution built with the same patch_size=14 ViT.

Usage (inside sam3_trt container):
  # Single video duplicated across 8 cameras
  python3 infer_multi.py --config config.json \
    --video Inputs/media1.mp4 --cameras 8 --output outputs

  # Multiple videos cycled across cameras
  python3 infer_multi.py --config config.json \
    --video Inputs/shop.mp4 Inputs/hair.mp4 Inputs/car.mp4 \
    --cameras 8 --output outputs

  # With grid video output for visual review
  python3 infer_multi.py --config config.json \
    --video Inputs/media1.mp4 --cameras 8 --save-video
"""
import cv2
import json
import time
import argparse
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

LOG = trt.Logger(trt.Logger.WARNING)
MAX_CAMERAS = 8
MAX_CLASSES = 8
PATCH_SIZE = 14

FPN_NAMES = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]
OUT_NAMES = ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]

# Max detections per class iteration across all frames
# 8 frames × ~30 dets/frame worst case = 240
MAX_DETS_PER_ITER = 256


def load_engine(path: str):
    with open(path, "rb") as f:
        return trt.Runtime(LOG).deserialize_cuda_engine(f.read())


# ---------------------------------------------------------------------------
# Multi-Camera Pipeline (Plan C v3)
# ---------------------------------------------------------------------------

class MultiCameraPipeline:
    """Plan C v3: selective mask copy + double-buffered decoder output.

    Key optimisations over v2:
      1. Selective mask copy: only transfer masks for detected queries
         (~5 dets/frame vs 200 queries = ~40x less PCIe bandwidth)
      2. All mask copies queued async, single sync per class iteration
      3. Double-buffered decoder outputs: decoder cls N+1 runs while
         masks from cls N are being copied/processed
    """

    def __init__(self, config_path: str, num_cameras: int = 8):
        cfg_path = Path(config_path)
        base = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())

        trt.init_libnvinfer_plugins(LOG, "")
        engine_dir = (base / cfg["engines"]).resolve()
        features_dir = base / cfg["features"]
        self.confidence = cfg.get("confidence", 0.3)

        self.classes = [c["name"] for c in cfg["classes"]]
        N = len(self.classes)
        if N > MAX_CLASSES:
            raise ValueError(f"Max {MAX_CLASSES} classes (got {N})")
        self.N = N
        self.F = num_cameras

        meta = json.loads((features_dir / "_meta.json").read_text())
        self.P = meta["max_prompt_len"]

        print(f"MultiCameraPipeline: {self.F} cameras, {N} classes, prompt_len={self.P}")

        F = self.F
        P = self.P

        # Two streams: compute (VE+decoder) and copy (mask transfer)
        self.stream_compute = cuda.Stream()
        self.stream_copy = cuda.Stream()

        # --- Pre-replicate prompts on GPU (one-time) ---
        self.prompt_feat_gpu_per_class = []
        self.prompt_mask_gpu_per_class = []

        for c in cfg["classes"]:
            d = features_dir / c["name"]
            f1 = np.load(d / "features.npy").astype(np.float32)
            m1 = np.load(d / "mask.npy").astype(np.bool_)
            assert f1.shape == (1, P, 256), f"{c['name']}: shape {f1.shape}"

            feat_f = np.tile(f1, (F, 1, 1))
            mask_f = np.tile(m1, (F, 1))

            pf_gpu = cuda.mem_alloc(feat_f.nbytes)
            pm_gpu = cuda.mem_alloc(mask_f.nbytes)

            pf_host = cuda.pagelocked_empty(feat_f.size, np.float32)
            np.copyto(pf_host, feat_f.ravel())
            cuda.memcpy_htod_async(pf_gpu, pf_host, self.stream_compute)

            pm_host = cuda.pagelocked_empty(mask_f.size, np.bool_)
            np.copyto(pm_host, mask_f.ravel())
            cuda.memcpy_htod_async(pm_gpu, pm_host, self.stream_compute)

            self.prompt_feat_gpu_per_class.append(pf_gpu)
            self.prompt_mask_gpu_per_class.append(pm_gpu)
            print(f"  {c['name']}: loaded + replicated x{F} ({c.get('prompt_type', 'text')})")

        # --- Vision encoder (batch=F) ---
        print("Loading vision-encoder...")
        self.vision_engine = load_engine(str(engine_dir / "vision-encoder.engine"))
        self.vision_ctx = self.vision_engine.create_execution_context()

        # Probe IMAGE_SIZE from VE engine profile (supports 1008, 840, etc.)
        ve_min = self.vision_engine.get_tensor_profile_shape("images", 0)[0]
        IMAGE_SIZE = ve_min[2]
        self.IMAGE_SIZE = IMAGE_SIZE
        patches = IMAGE_SIZE // PATCH_SIZE
        MASK_H = patches * 4
        MASK_W = patches * 4
        self.MASK_H = MASK_H
        self.MASK_W = MASK_W
        self.MASK_BYTES = MASK_H * MASK_W * 4
        FPN_SHAPES_SINGLE = [
            (256, patches * 4, patches * 4),
            (256, patches * 2, patches * 2),
            (256, patches, patches),
            (256, patches, patches),
        ]
        self.FPN_SHAPES_SINGLE = FPN_SHAPES_SINGLE
        print(f"  Resolution: {IMAGE_SIZE}x{IMAGE_SIZE} (auto-detected)")

        img_elems = F * 3 * IMAGE_SIZE * IMAGE_SIZE
        self.img_host = cuda.pagelocked_empty(img_elems, np.float32)
        self.img_gpu = cuda.mem_alloc(img_elems * 4)

        # FPN [F, 256, H, W] — shared VE output / decoder input (zero-copy)
        self.fpn_gpu = []
        for shape in FPN_SHAPES_SINGLE:
            self.fpn_gpu.append(cuda.mem_alloc(F * int(np.prod(shape)) * 4))

        # --- Decoder (batch=F, iterated per class) ---
        print("Loading decoder...")
        self.decoder_engine = load_engine(str(engine_dir / "decoder.engine"))
        self.decoder_ctx = self.decoder_engine.create_execution_context()

        # Probe QUERIES from decoder output shape (50 or 200 depending on engine)
        self.decoder_ctx.set_input_shape("fpn_feat_0", (1, 256, MASK_H, MASK_W))
        self.decoder_ctx.set_input_shape("fpn_feat_1", (1, 256, patches*2, patches*2))
        self.decoder_ctx.set_input_shape("fpn_feat_2", (1, 256, patches, patches))
        self.decoder_ctx.set_input_shape("fpn_pos_2", (1, 256, patches, patches))
        self.decoder_ctx.set_input_shape("prompt_features", (1, 1, 256))
        self.decoder_ctx.set_input_shape("prompt_mask", (1, 1))
        Q = self.decoder_ctx.get_tensor_shape("pred_boxes")[1]
        self.Q = Q
        print(f"  Decoder QUERIES={Q} (auto-detected from engine)")

        # Double-buffered output: buf[0] and buf[1]
        self.out_shapes = [
            (F, Q, MASK_H, MASK_W),  # pred_masks
            (F, Q, 4),               # pred_boxes
            (F, Q),                   # pred_logits
            (F, 1),                   # presence_logits
        ]
        self.out_gpu = [
            [cuda.mem_alloc(int(np.prod(s)) * 4) for s in self.out_shapes],
            [cuda.mem_alloc(int(np.prod(s)) * 4) for s in self.out_shapes],
        ]
        # Host buffers for boxes/logits/presence (double-buffered)
        self.out_host = [
            [cuda.pagelocked_empty(int(np.prod(s)), np.float32)
             for s in self.out_shapes[1:]],
            [cuda.pagelocked_empty(int(np.prod(s)), np.float32)
             for s in self.out_shapes[1:]],
        ]

        # Selective mask pool: pre-allocated pagelocked buffers
        self.mask_pool = [
            cuda.pagelocked_empty(MASK_H * MASK_W, np.float32)
            for _ in range(MAX_DETS_PER_ITER)
        ]

        # CUDA events for inter-stream synchronisation
        self.event_decoder_done = [cuda.Event() for _ in range(MAX_CLASSES)]
        self.event_copy_done = [cuda.Event(), cuda.Event()]  # per double-buf

        self.stream_compute.synchronize()

        # Pre-allocated numpy buffer for preprocessing
        self._preproc_buf = np.empty(
            (F, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # VRAM estimate
        vram = img_elems * 4
        for shape in FPN_SHAPES_SINGLE:
            vram += F * int(np.prod(shape)) * 4
        for s in self.out_shapes:
            vram += int(np.prod(s)) * 4 * 2  # double buffered
        for _ in range(N):
            vram += F * P * 256 * 4 + F * P
        self.vram_mb = vram / 1024 / 1024

        print(f"Ready: {F} cameras, {N} classes, prompt_len={P}, "
              f"buffers ~{self.vram_mb:.0f} MB (double-buffered)\n")

    def _preprocess_frames(self, frames):
        """Preprocess F frames into img_host using cv2."""
        IMAGE_SIZE = self.IMAGE_SIZE
        for i, fr in enumerate(frames):
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            self._preproc_buf[i] = (
                resized.astype(np.float32) / 127.5 - 1.0
            ).transpose(2, 0, 1)
        np.copyto(self.img_host, self._preproc_buf.ravel())
        cuda.memcpy_htod_async(
            self.img_gpu, self.img_host, self.stream_compute)

    def _launch_decoder(self, cls_idx, buf_idx):
        """Launch decoder for one class on stream_compute."""
        F, P = self.F, self.P
        gpu = self.out_gpu[buf_idx]

        # FPN: zero-copy from VE output
        for i, name in enumerate(FPN_NAMES):
            self.decoder_ctx.set_input_shape(name, (F, *self.FPN_SHAPES_SINGLE[i]))
            self.decoder_ctx.set_tensor_address(name, self.fpn_gpu[i])

        # Prompt: pre-replicated, just set pointer
        self.decoder_ctx.set_input_shape("prompt_features", (F, P, 256))
        self.decoder_ctx.set_tensor_address(
            "prompt_features", self.prompt_feat_gpu_per_class[cls_idx])
        self.decoder_ctx.set_input_shape("prompt_mask", (F, P))
        self.decoder_ctx.set_tensor_address(
            "prompt_mask", self.prompt_mask_gpu_per_class[cls_idx])

        # Output to buf[buf_idx]
        for i, name in enumerate(OUT_NAMES):
            self.decoder_ctx.set_tensor_address(name, gpu[i])

        self.decoder_ctx.execute_async_v3(
            stream_handle=self.stream_compute.handle)
        self.event_decoder_done[cls_idx].record(self.stream_compute)

    def _transfer_and_process(self, cls_idx, cls_name, buf_idx, hws, conf,
                              all_dets):
        """Transfer boxes/logits + selective masks, then postprocess."""
        F, Q = self.F, self.Q
        MASK_H, MASK_W = self.MASK_H, self.MASK_W
        MASK_BYTES = self.MASK_BYTES
        gpu = self.out_gpu[buf_idx]
        host = self.out_host[buf_idx]

        # Wait for decoder to finish on stream_copy
        self.stream_copy.wait_for_event(self.event_decoder_done[cls_idx])

        # Async copy boxes/logits/presence (small)
        for i in range(len(host)):
            n = int(np.prod(self.out_shapes[i + 1]))
            cuda.memcpy_dtoh_async(
                host[i][:n], gpu[i + 1], self.stream_copy)
        self.stream_copy.synchronize()

        boxes = host[0][:F*Q*4].reshape(F, Q, 4)
        logits = host[1][:F*Q].reshape(F, Q)
        presence = host[2][:F].reshape(F, 1)

        # Determine active queries per frame, queue SELECTIVE mask copies
        frame_keeps = []
        frame_scores = []
        pool_idx = 0

        for fi in range(F):
            p_score = 1.0 / (1.0 + np.exp(-presence[fi, 0]))
            scores = (1.0 / (1.0 + np.exp(-logits[fi]))) * p_score
            keep = np.where(scores > conf)[0]
            frame_keeps.append(keep)
            frame_scores.append(scores)

            # Queue async copy for ONLY the detected masks
            for j in keep:
                if pool_idx >= MAX_DETS_PER_ITER:
                    break
                offset = (fi * Q + int(j)) * MASK_BYTES
                cuda.memcpy_dtoh_async(
                    self.mask_pool[pool_idx],
                    int(gpu[0]) + offset,
                    self.stream_copy)
                pool_idx += 1

        # Single sync for ALL selective mask copies
        self.stream_copy.synchronize()
        self.event_copy_done[buf_idx].record(self.stream_copy)

        # Process masks on CPU (GPU is free to run next decoder)
        pool_idx = 0
        for fi in range(F):
            h, w = hws[fi]
            keep = frame_keeps[fi]
            scores = frame_scores[fi]

            if len(keep) == 0:
                continue

            # Vectorised box scaling
            keep_boxes = boxes[fi, keep].copy()
            keep_boxes[:, [0, 2]] *= w
            keep_boxes[:, [1, 3]] *= h
            keep_boxes = np.clip(keep_boxes, 0, [w, h, w, h])

            for ki, j in enumerate(keep):
                if pool_idx >= MAX_DETS_PER_ITER:
                    break
                raw = self.mask_pool[pool_idx].reshape(MASK_H, MASK_W)
                mask = cv2.resize(
                    raw, (w, h), interpolation=cv2.INTER_LINEAR) > 0
                all_dets[fi].append({
                    "class": cls_name,
                    "box": keep_boxes[ki].tolist(),
                    "score": float(scores[j]),
                    "mask": mask,
                })
                pool_idx += 1

    def detect_batch(self, frames: list, conf: float = None) -> list:
        """Full pipeline with double-buffered decoder + selective mask copy.

        Timeline (4 classes):
          stream_compute: [VE] [Dec0->buf0] [Dec1->buf1] [Dec2->buf0] [Dec3->buf1]
          stream_copy:          [copy buf0]  [copy buf1] [copy buf0] [copy buf1]
          CPU:                  [process 0]  [process 1] [process 2] [process 3]

        Decoder N+1 overlaps with mask copy + CPU processing of class N.
        """
        if conf is None:
            conf = self.confidence
        F = self.F
        IMAGE_SIZE = self.IMAGE_SIZE
        hws = [fr.shape[:2] for fr in frames]
        all_dets = [[] for _ in range(F)]

        # 1. Preprocess all F frames -> GPU
        self._preprocess_frames(frames)

        # 2. Vision encoder batch=F
        self.vision_ctx.set_input_shape("images", (F, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.vision_ctx.set_tensor_address("images", self.img_gpu)
        for i, name in enumerate(FPN_NAMES):
            self.vision_ctx.set_tensor_address(name, self.fpn_gpu[i])
        self.vision_ctx.execute_async_v3(
            stream_handle=self.stream_compute.handle)

        # 3. Pipelined decoder: launch cls 0
        self._launch_decoder(0, buf_idx=0)

        # 4. For each subsequent class: transfer prev + launch next
        for cls_idx in range(1, self.N):
            prev_buf = (cls_idx - 1) % 2
            curr_buf = cls_idx % 2

            # Ensure this buffer's previous copy is done before decoder writes
            if cls_idx >= 2:
                self.stream_compute.wait_for_event(
                    self.event_copy_done[curr_buf])

            # Launch decoder for current class
            self._launch_decoder(cls_idx, buf_idx=curr_buf)

            # Meanwhile: transfer + process previous class's results
            self._transfer_and_process(
                cls_idx - 1, self.classes[cls_idx - 1],
                prev_buf, hws, conf, all_dets)

        # 5. Process last class
        last_buf = (self.N - 1) % 2
        self._transfer_and_process(
            self.N - 1, self.classes[self.N - 1],
            last_buf, hws, conf, all_dets)

        for fi in range(F):
            all_dets[fi].sort(key=lambda d: d["score"], reverse=True)

        return all_dets


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

PALETTE = [
    (30, 144, 255), (255, 144, 30), (144, 255, 30), (255, 30, 144),
    (255, 200, 50), (30, 255, 144), (255, 255, 30), (30, 255, 255),
]


def draw_overlay(image_bgr, dets, alpha=0.45):
    vis = image_bgr.copy()
    if not dets:
        return vis

    cmap = {}
    for d in dets:
        if d["class"] not in cmap:
            cmap[d["class"]] = PALETTE[len(cmap) % len(PALETTE)]

    for d in reversed(dets):
        c = np.array(cmap[d["class"]], dtype=np.float32)
        m = d["mask"]
        if m is not None:
            vis[m] = (vis[m].astype(np.float32) * (1 - alpha)
                      + c * alpha).astype(np.uint8)

    for d in reversed(dets):
        m = d.get("mask")
        if m is not None:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, cmap[d["class"]], 2)

    for d in dets:
        x1, y1 = int(d["box"][0]), int(d["box"][1])
        c = cmap[d["class"]]
        label = f"{d['class']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), c, -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def compose_grid(frames, all_dets, cam_labels, F, cols, cell_w, cell_h):
    """Compose all camera overlays into a single grid image."""
    rows = (F + cols - 1) // cols
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for fi in range(F):
        overlay = draw_overlay(frames[fi], all_dets[fi])
        cell = cv2.resize(overlay, (cell_w, cell_h))
        # Camera label with shadow
        label = f"[{fi}] {cam_labels[fi]}"
        cv2.putText(cell, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        r, c = divmod(fi, cols)
        grid[r * cell_h:(r + 1) * cell_h,
             c * cell_w:(c + 1) * cell_w] = cell
    return grid


def run_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def avi_to_mp4(avi_path: Path) -> Path:
    """Convert AVI (MJPG) to MP4 (H.264) via ffmpeg, delete AVI on success."""
    mp4_path = avi_path.with_suffix(".mp4")
    cmd = ["ffmpeg", "-y", "-i", str(avi_path),
           "-c:v", "libx264", "-crf", "23", "-preset", "fast",
           "-movflags", "+faststart", "-an", str(mp4_path)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and mp4_path.exists():
        avi_path.unlink()
        return mp4_path
    print(f"  WARNING: ffmpeg failed for {avi_path.name}, keeping AVI")
    return avi_path


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------

def run_multi_video(pipe, video_paths, out_dir, conf, interval, tag,
                    save_video=False, save_cameras=False):
    """Multi-camera video inference with real different video sources.

    Each video_path gets its own cv2.VideoCapture.  If fewer videos than
    cameras, videos are cycled to fill all F slots.  The master clock is
    driven by the first video; when any camera's video ends it loops from
    the beginning.

    Phase 1: inference + real-time AVI direct write (MJPG)
    Phase 2: batch convert all AVI → MP4 (H.264), delete AVIs
    """
    F = pipe.F
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{tag}_detections.jsonl"
    times = []

    # Open one capture per camera, cycling videos if < F
    caps = []
    cam_labels = []
    for i in range(F):
        vp = video_paths[i % len(video_paths)]
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"ERROR: cannot open {vp}")
            return times
        caps.append(cap)
        cam_labels.append(Path(vp).stem)

    # Master clock from first video
    src_fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    total = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / src_fps
    min_advance = interval / src_fps

    unique_vids = list(dict.fromkeys(str(p) for p in video_paths))
    print(f"Cameras: {F} ({len(unique_vids)} unique videos, cycled)")
    for i, label in enumerate(cam_labels):
        print(f"  cam[{i}]: {label}")
    print(f"Master clock: {total} frames, {src_fps:.1f} fps, {duration:.1f}s")
    if save_video:
        print("Grid video output: ENABLED (real-time AVI)")
    if save_cameras:
        print("Individual camera video output: ENABLED (real-time AVI)")
    print()

    # --- AVI writers (real-time direct write) ---
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    avi_paths = []  # collect all AVI paths for batch conversion

    grid_writer = None
    if save_video:
        CELL_W, CELL_H = 480, 270
        cols = min(F, 4)
        grid_w = cols * CELL_W
        grid_h = ((F + cols - 1) // cols) * CELL_H
        grid_avi = out_dir / f"{tag}_grid.avi"
        grid_writer = cv2.VideoWriter(str(grid_avi), fourcc, src_fps,
                                      (grid_w, grid_h))
        avi_paths.append(grid_avi)

    cam_writers = []
    if save_cameras:
        for ci in range(F):
            cam_avi = out_dir / f"{tag}_cam{ci}_{cam_labels[ci]}.avi"
            cam_w = int(caps[ci].get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_h = int(caps[ci].get(cv2.CAP_PROP_FRAME_HEIGHT))
            cam_writers.append(
                cv2.VideoWriter(str(cam_avi), fourcc, src_fps, (cam_w, cam_h)))
            avi_paths.append(cam_avi)

    video_clock = 0.0
    next_idx = 0
    n_out = 0
    frame_idx = -1

    def read_frame(cap):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return frame if ok else None

    # --- Phase 1: inference + real-time AVI output ---
    with open(jsonl, "w") as f:
        while True:
            # Master camera drives the loop
            ok, master_frame = caps[0].read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx < next_idx:
                # Skip on all cameras to stay in sync
                for cap in caps[1:]:
                    cap.read()
                continue

            # Read from all cameras (non-master may loop)
            frames = [master_frame]
            for cap in caps[1:]:
                fr = read_frame(cap)
                if fr is None:
                    fr = master_frame
                frames.append(fr)

            t0 = time.time()
            all_dets = pipe.detect_batch(frames, conf)
            dt = time.time() - t0
            times.append(dt)

            total_dets = sum(len(d) for d in all_dets)
            per_cam = dt * 1000 / F
            det_str = " ".join(f"{cam_labels[i]}={len(all_dets[i])}"
                               for i in range(F))
            print(f"#{frame_idx}: {dt*1000:.0f} ms total ({per_cam:.0f} ms/cam), "
                  f"{total_dets} dets [{det_str}]")

            # Write grid overlay directly to AVI
            if grid_writer is not None:
                grid = compose_grid(frames, all_dets, cam_labels,
                                    F, cols, CELL_W, CELL_H)
                grid_writer.write(grid)

            # Write individual camera overlays directly to AVI
            if cam_writers:
                for ci in range(F):
                    overlay = draw_overlay(frames[ci], all_dets[ci])
                    cam_writers[ci].write(overlay)

            row = {
                "frame_idx": frame_idx,
                "elapsed_ms": round(dt * 1000, 1),
                "per_camera_ms": round(per_cam, 1),
                "cameras": F,
                "camera_sources": cam_labels,
                "detections_per_camera": {
                    cam_labels[i]: len(all_dets[i]) for i in range(F)
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_out += 1

            video_clock += max(dt, min_advance)
            next_idx = int(video_clock * src_fps)
            if next_idx <= frame_idx:
                next_idx = frame_idx + 1

    # Release all resources
    for cap in caps:
        cap.release()
    if grid_writer is not None:
        grid_writer.release()
    for w in cam_writers:
        w.release()
    print(f"\nProcessed {n_out}/{total} master frames")

    # --- Phase 2: batch convert all AVI → MP4 ---
    if avi_paths:
        print(f"\nConverting {len(avi_paths)} AVI → MP4 ...")
        for avi in avi_paths:
            if avi.exists():
                out_path = avi_to_mp4(avi)
                sz_mb = out_path.stat().st_size / 1048576
                print(f"  {out_path.name} ({sz_mb:.0f} MB)")

    return times


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------

def write_performance(times, out_dir, pipe, tag, video_paths=None):
    if not times:
        return
    warm = times[1:] if len(times) > 1 else times
    ms = [t * 1000 for t in warm]
    s = sorted(ms)
    p95 = s[max(0, int(len(s) * 0.95) - 1)]
    avg = float(np.mean(ms))
    F = pipe.F
    perf = {
        "cameras": F,
        "video_sources": [Path(p).name for p in video_paths] if video_paths else [],
        "classes": pipe.N,
        "queries": pipe.Q,
        "image_size": pipe.IMAGE_SIZE,
        "prompt_len": pipe.P,
        "frames_processed": len(times),
        "total_avg_ms": round(avg, 1),
        "per_camera_avg_ms": round(avg / F, 1),
        "min_ms": round(min(ms), 1),
        "max_ms": round(max(ms), 1),
        "p95_ms": round(p95, 1),
        "total_fps": round(1000 / avg * F, 1),
        "per_camera_fps": round(1000 / avg, 1),
        "buffer_vram_mb": round(pipe.vram_mb, 1),
    }
    path = out_dir / f"{tag}_performance.json"
    path.write_text(json.dumps(perf, indent=2))
    print(f"\nPerformance ({F} cameras, {pipe.N} classes, excl. warmup):")
    print(f"  Total:      avg {perf['total_avg_ms']:.0f} ms/round, "
          f"p95 {perf['p95_ms']:.0f} ms")
    print(f"  Per camera: avg {perf['per_camera_avg_ms']:.0f} ms, "
          f"~{perf['per_camera_fps']:.1f} FPS/camera")
    print(f"  Throughput: ~{perf['total_fps']:.1f} total FPS ({F} cameras)")
    print(f"  Buffers:    ~{perf['buffer_vram_mb']:.0f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="SAM3 multi-camera inference (Plan C v3)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", nargs="+", required=True,
                    help="Video file(s). If fewer than --cameras, videos are "
                         "cycled to fill all camera slots.")
    ap.add_argument("--cameras", type=int, default=8)
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--interval", type=int, default=1)
    ap.add_argument("--save-video", action="store_true",
                    help="Save grid overlay video (2x4 layout) for visual "
                         "review. Adds rendering overhead -- omit for speed "
                         "benchmarks.")
    ap.add_argument("--save-cameras", action="store_true",
                    help="Save individual camera overlay videos (one AVI per "
                         "camera). Can be combined with --save-video.")
    args = ap.parse_args()

    out = Path(args.output)
    pipe = MultiCameraPipeline(args.config, num_cameras=args.cameras)
    conf = args.conf if args.conf is not None else pipe.confidence
    tag = run_tag()

    times = run_multi_video(pipe, args.video, out, conf, args.interval, tag,
                            save_video=args.save_video,
                            save_cameras=args.save_cameras)
    write_performance(times, out, pipe, tag, video_paths=args.video)


if __name__ == "__main__":
    main()
