#!/bin/bash
# SAM3 Pipeline Benchmark: r448 vs r560, single vs 8-cam, with/without video
# Run inside Docker: docker exec william_tensorrt bash /root/VisionDSL/models/sam3_pipeline/benchmark.sh
set -e
cd /root/VisionDSL/models/sam3_pipeline

VIDEOS="Inputs/shop.mp4 Inputs/hair.mp4 Inputs/car.mp4"
OUT=outputs/benchmark

echo "=============================================="
echo " SAM3 Benchmark: r448 vs r560"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================="
echo ""

# -----------------------------------------------
# Round 1: Performance only (no video output)
# -----------------------------------------------
echo "====== ROUND 1: Performance Only (JSON) ======"
echo ""

echo "--- [1/4] r448 single camera ---"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > /tmp/gpu_before.txt
python3 infer.py --config config_r448.json --video Inputs/shop.mp4 --output $OUT/r448_single_perf
echo ""

echo "--- [2/4] r448 8 cameras ---"
python3 infer_multi.py --config config_r448.json --video $VIDEOS --cameras 8 --output $OUT/r448_8cam_perf
echo ""

echo "--- [3/4] r560 single camera ---"
python3 infer.py --config config_r560.json --video Inputs/shop.mp4 --output $OUT/r560_single_perf
echo ""

echo "--- [4/4] r560 8 cameras ---"
python3 infer_multi.py --config config_r560.json --video $VIDEOS --cameras 8 --output $OUT/r560_8cam_perf
echo ""

# -----------------------------------------------
# Round 2: With video output (debug mode)
# -----------------------------------------------
echo "====== ROUND 2: With Video Output (Debug) ======"
echo ""

echo "--- [5/8] r448 single camera + video ---"
python3 infer.py --config config_r448.json --video Inputs/shop.mp4 --output $OUT/r448_single_video --save-video
echo ""

echo "--- [6/8] r448 8 cameras + video ---"
python3 infer_multi.py --config config_r448.json --video $VIDEOS --cameras 8 --output $OUT/r448_8cam_video --save-video --save-cameras
echo ""

echo "--- [7/8] r560 single camera + video ---"
python3 infer.py --config config_r560.json --video Inputs/shop.mp4 --output $OUT/r560_single_video --save-video
echo ""

echo "--- [8/8] r560 8 cameras + video ---"
python3 infer_multi.py --config config_r560.json --video $VIDEOS --cameras 8 --output $OUT/r560_8cam_video --save-video --save-cameras
echo ""

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo "=============================================="
echo " BENCHMARK COMPLETE - Results Summary"
echo "=============================================="
echo ""
echo "Round 1 (Performance Only):"
for d in r448_single_perf r448_8cam_perf r560_single_perf r560_8cam_perf; do
    echo "--- $d ---"
    cat $OUT/$d/*_performance.json 2>/dev/null || echo "  (no result)"
    echo ""
done

echo "Round 2 (With Video):"
for d in r448_single_video r448_8cam_video r560_single_video r560_8cam_video; do
    echo "--- $d ---"
    cat $OUT/$d/*_performance.json 2>/dev/null || echo "  (no result)"
    echo ""
done
