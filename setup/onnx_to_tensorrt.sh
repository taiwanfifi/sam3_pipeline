#!/bin/bash
# =========================================================================
#  ONNX → TensorRT 引擎轉換腳本
# =========================================================================
#
#  用途：
#    將 4 個 ONNX 模型轉換為 TensorRT 引擎（.engine 檔案）。
#    TensorRT 引擎是 NVIDIA 針對你的 GPU 高度優化的推論格式，
#    速度比直接跑 PyTorch 快很多倍，但只能在同型 GPU 上使用。
#
#  重要觀念：
#    - .engine 檔案綁定特定 GPU 架構，換 GPU 就要重新轉換
#    - .onnx 檔案是通用格式，任何 GPU 都能用，只需匯出一次
#    - trtexec 是 TensorRT 容器內建的命令列轉換工具
#
#  trtexec 常用參數說明：
#    --fp16          使用半精度浮點數 (FP16) 加速，精度損失極小
#    --onnx          輸入的 ONNX 模型路徑
#    --saveEngine    輸出的 .engine 檔案路徑
#    --minShapes     最小維度組合（引擎優化的下界）
#    --optShapes     最常使用的維度組合（引擎會重點針對這組優化）
#    --maxShapes     最大維度組合（引擎優化的上界）
#
#  動態維度 (Dynamic Shapes) 概念：
#    TensorRT 引擎在轉換時需要知道每個輸入的維度範圍。
#    例如 batch 維度設定 min=1, opt=8, max=8，
#    表示這個引擎最少處理 1 個類別、最多 8 個、最常處理 8 個。
#    如果實際 batch 超過 max，引擎會報錯。
#
#  MAX_CLASSES 設定：
#    所有引擎的 maxShapes batch 維度決定了最大可用類別數。
#    預設值為 8。如需調整，修改下方所有 maxShapes 和 optShapes
#    裡的 batch 數字（例如把所有 8 改成 4 或 16）。
#    batch 越大 → VRAM 越高（引擎會為最大值預留記憶體）。
#    詳見 setup/README.md 的「最大類別數」章節。
#
#  使用方式（在 TensorRT 容器內執行）：
#    bash onnx_to_tensorrt.sh <onnx 資料夾路徑> [輸出資料夾路徑]
#
#  範例：
#    bash onnx_to_tensorrt.sh /root/sam3_pipeline/setup/onnx
#    bash onnx_to_tensorrt.sh /root/sam3_pipeline/setup/onnx /root/sam3_pipeline/engines/b8_q200
#
#  轉換時間：
#    約 5~15 分鐘（視 GPU 而定），大部分時間花在 Vision Encoder。
#
#  輸出位置：
#    引擎會儲存到上一層的 engines/ 資料夾（../engines/）
# =========================================================================

set -e  # 任何指令失敗就立即停止

ONNX_DIR="${1:?用法: bash onnx_to_tensorrt.sh <onnx 資料夾路徑> [輸出資料夾路徑]}"
OUT_DIR="${2:-$(dirname "$0")/../engines/b8_q200}"

# --- 前置檢查 ---
# 確認所有 ONNX 檔案都存在，避免轉到一半才發現少了檔案
ONNX_FILES=("vision-encoder" "text-encoder" "geometry-encoder" "decoder")
for f in "${ONNX_FILES[@]}"; do
    if [ ! -f "$ONNX_DIR/$f.onnx" ]; then
        echo "錯誤: 找不到 $ONNX_DIR/$f.onnx"
        echo "請先執行 export_sam3_to_onnx.py 產生 ONNX 檔案"
        exit 1
    fi
done

mkdir -p "$OUT_DIR"

echo "ONNX 來源: $ONNX_DIR"
echo "輸出位置: $OUT_DIR"
echo ""

# -------------------------------------------------------------------------
# [1/4] Vision Encoder — 影像編碼器
# -------------------------------------------------------------------------
# 輸入：images [B, 3, 1008, 1008]  （RGB 圖片）
# 輸出：3 層 FPN 特徵 + 1 個位置編碼
#
# batch 維度：min=1, max=8（= MAX_CLASSES）
# 這是最大的模型（~880MB ONNX），轉換時間最長
# -------------------------------------------------------------------------
echo "=== [1/4] Vision Encoder ==="
trtexec --fp16 \
    --onnx="$ONNX_DIR/vision-encoder.onnx" \
    --saveEngine="$OUT_DIR/vision-encoder.engine" \
    --minShapes=images:1x3x1008x1008 \
    --optShapes=images:8x3x1008x1008 \
    --maxShapes=images:8x3x1008x1008

# -------------------------------------------------------------------------
# [2/4] Text Encoder — 文字編碼器
# -------------------------------------------------------------------------
# 輸入：input_ids [B, 32]      （token ID 序列）
#       attention_mask [B, 32]  （有效 token 遮罩）
#
# 32 是 CLIP tokenizer 的固定 token 長度。
# batch 維度用於同時編碼多個類別的文字。
# -------------------------------------------------------------------------
echo "=== [2/4] Text Encoder ==="
trtexec --fp16 \
    --onnx="$ONNX_DIR/text-encoder.onnx" \
    --saveEngine="$OUT_DIR/text-encoder.engine" \
    --minShapes=input_ids:1x32,attention_mask:1x32 \
    --optShapes=input_ids:8x32,attention_mask:8x32 \
    --maxShapes=input_ids:8x32,attention_mask:8x32

# -------------------------------------------------------------------------
# [3/4] Geometry Encoder — 幾何編碼器
# -------------------------------------------------------------------------
# 輸入：input_boxes [B, num_boxes, 4]        （正規化 cxcywh 框座標）
#       input_boxes_labels [B, num_boxes]     （標籤：1=正, 0=負）
#       fpn_feat_2 [B, 256, 72, 72]          （Vision Encoder 的 FPN 特徵）
#       fpn_pos_2 [B, 256, 72, 72]           （位置編碼）
#
# num_boxes：每個類別最多 20 個參考框。
# 72x72 是 1008/14 的 patch 數量（固定不變）。
# -------------------------------------------------------------------------
echo "=== [3/4] Geometry Encoder ==="
trtexec --fp16 \
    --onnx="$ONNX_DIR/geometry-encoder.onnx" \
    --saveEngine="$OUT_DIR/geometry-encoder.engine" \
    --minShapes=input_boxes:1x1x4,input_boxes_labels:1x1,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72 \
    --optShapes=input_boxes:1x8x4,input_boxes_labels:1x8,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72 \
    --maxShapes=input_boxes:8x20x4,input_boxes_labels:8x20,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72

# -------------------------------------------------------------------------
# [4/4] Decoder — 解碼器
# -------------------------------------------------------------------------
# 輸入：fpn_feat_0 [B, 256, 288, 288]            （高解析度 FPN）
#       fpn_feat_1 [B, 256, 144, 144]            （中解析度 FPN）
#       fpn_feat_2 [B, 256,  72,  72]            （低解析度 FPN）
#       fpn_pos_2  [B, 256,  72,  72]            （位置編碼）
#       prompt_features [B, prompt_len, 256]      （prompt 特徵）
#       prompt_mask [B, prompt_len]               （有效位遮罩）
#
# B = 類別數量（最多 8，由 maxShapes 決定）— 每個類別獨立解碼
# prompt_len = prompt token 數（最多 60）：
#   - 文字 prompt: 32 tokens
#   - 幾何 prompt: num_boxes*2 + 1 tokens
#   - 組合 prompt: 文字 + 幾何 tokens
# -------------------------------------------------------------------------
echo "=== [4/4] Decoder ==="
trtexec --fp16 \
    --onnx="$ONNX_DIR/decoder.onnx" \
    --saveEngine="$OUT_DIR/decoder.engine" \
    --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x1x256,prompt_mask:1x1 \
    --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x33x256,prompt_mask:1x33 \
    --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,prompt_features:8x60x256,prompt_mask:8x60

echo ""
echo "轉換完成！4 個引擎已儲存到 $OUT_DIR"
ls -lh "$OUT_DIR"/*.engine
