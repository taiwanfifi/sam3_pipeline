"""
Build INT8 Vision Encoder TensorRT engine with entropy calibration.

Usage (inside TensorRT container):
  # Extract calibration frames from video, then build engine
  python3 setup/build_int8_ve.py \
    --onnx setup/onnx_q200/vision-encoder.onnx \
    --video Inputs/shop.mp4 \
    --output engines/b8_q50_int8/vision-encoder.engine \
    --num-frames 100 \
    --image-size 1008

  # Use existing calibration cache (skip frame extraction)
  python3 setup/build_int8_ve.py \
    --onnx setup/onnx_q200/vision-encoder.onnx \
    --output engines/b8_q50_int8/vision-encoder.engine \
    --cache setup/ve_int8_calib.cache \
    --image-size 1008
"""
import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image as PILImage

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 — initializes CUDA context


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class VECalibrator(trt.IInt8EntropyCalibrator2):
    """Entropy calibration using video frames, matching infer.py preprocessing."""

    def __init__(self, video_path, image_size, num_frames=100, cache_path=None):
        super().__init__()
        self.image_size = image_size
        self.cache_path = cache_path
        self.batch_size = 1  # calibrate one frame at a time
        self.current_idx = 0

        # Try loading cache first
        if cache_path and os.path.isfile(cache_path):
            print(f"[Calibrator] Loading cache: {cache_path}")
            with open(cache_path, "rb") as f:
                self._cache = f.read()
            self.frames = []  # no frames needed
        else:
            self._cache = None
            self.frames = self._extract_frames(video_path, num_frames)
            print(f"[Calibrator] Extracted {len(self.frames)} calibration frames")

        # GPU buffer for single image [1, 3, IMAGE_SIZE, IMAGE_SIZE]
        self.nbytes = 1 * 3 * image_size * image_size * 4  # float32
        self.d_input = cuda.mem_alloc(self.nbytes)

    def _extract_frames(self, video_path, num_frames):
        """Uniformly sample frames from video."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // num_frames)
        frames = []
        idx = 0
        while len(frames) < num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            idx += step
        cap.release()
        return frames

    def _preprocess(self, bgr):
        """Match infer.py preprocessing exactly."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = np.array(PILImage.fromarray(rgb).resize(
            (self.image_size, self.image_size), PILImage.BILINEAR))
        tensor = (resized.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        return np.ascontiguousarray(tensor)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= len(self.frames):
            return None
        frame = self.frames[self.current_idx]
        self.current_idx += 1
        tensor = self._preprocess(frame)
        cuda.memcpy_htod(self.d_input, tensor.tobytes())
        if self.current_idx % 20 == 0:
            print(f"  calibrating... {self.current_idx}/{len(self.frames)}")
        return [int(self.d_input)]

    def read_calibration_cache(self):
        return self._cache

    def write_calibration_cache(self, cache):
        self._cache = cache
        if self.cache_path:
            with open(self.cache_path, "wb") as f:
                f.write(cache)
            print(f"[Calibrator] Cache saved: {self.cache_path}")


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------

def build_int8_engine(onnx_path, output_path, calibrator, image_size):
    """Build INT8+FP16 engine for Vision Encoder with dynamic batch."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"\nParsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            sys.exit(1)
    print("  ONNX parsed OK")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    # Dynamic batch: min=1, opt=8, max=8 (same as FP16 build)
    profile = builder.create_optimization_profile()
    profile.set_shape("images",
                      min=(1, 3, image_size, image_size),
                      opt=(8, 3, image_size, image_size),
                      max=(8, 3, image_size, image_size))
    config.add_optimization_profile(profile)

    print(f"\nBuilding INT8 engine (this takes 10-30 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: Engine build failed!")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    buf = bytes(serialized)
    with open(output_path, "wb") as f:
        f.write(buf)
    size_mb = len(buf) / 1024 / 1024
    print(f"\nEngine saved: {output_path} ({size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build INT8 Vision Encoder engine")
    ap.add_argument("--onnx", required=True, help="Vision encoder ONNX path")
    ap.add_argument("--output", required=True, help="Output .engine path")
    ap.add_argument("--video", help="Video for calibration frame extraction")
    ap.add_argument("--num-frames", type=int, default=100, help="Calibration frames")
    ap.add_argument("--image-size", type=int, default=1008, help="Input resolution")
    ap.add_argument("--cache", help="Calibration cache path (read/write)")
    args = ap.parse_args()

    cache_path = args.cache or os.path.splitext(args.output)[0] + "_calib.cache"

    if not args.video and not (args.cache and os.path.isfile(args.cache)):
        print("ERROR: --video required (no calibration cache found)")
        sys.exit(1)

    calibrator = VECalibrator(
        video_path=args.video or "",
        image_size=args.image_size,
        num_frames=args.num_frames,
        cache_path=cache_path,
    )

    build_int8_engine(args.onnx, args.output, calibrator, args.image_size)


if __name__ == "__main__":
    main()
