# Decoder QUERIES 優化實驗

DETR decoder 內部有 200 個 learned object queries，每個 query 負責偵測一個物件。
這些 queries 會直接影響 decoder 的 output buffer 大小 — 尤其是 `pred_masks [B, Q, 288, 288]`，在 multi-camera pipeline 裡透過 double-buffering 翻倍，成為 VRAM 的大頭。

本實驗透過在 ONNX 匯出時加入 topK 篩選，把 decoder output 從 200 降到 50，測量 VRAM 和速度影響。

---

## 原理

```
DETR decoder (固定 200 queries)
    ↓
pred_logits [B, 200]     ← 每個 query 的匹配分數
    ↓
topK (K=50)              ← 取分數最高的 50 個
    ↓
pred_masks  [B, 50, 288, 288]   ← 只對 top-50 做 mask decoder
pred_boxes  [B, 50, 4]
pred_logits [B, 50]
```

**關鍵：DETR 內部仍然用完整 200 queries 做 cross-attention，偵測品質不受影響。**
topK 只裁剪了 output — 少了 150 個分數最低（幾乎都是背景）的候選，mask decoder 也只需要處理 50 個。

## 影響範圍

| 元件 | Q200 | Q50 | 說明 |
|------|------|-----|------|
| ONNX 匯出 | `--num-queries 200` | `--num-queries 50` | 只需重匯出 decoder |
| TensorRT engine | decoder.engine | decoder.engine（重建） | VE/TE/GE 不受影響 |
| Output buffer | `[F, 200, 288, 288]` | `[F, 50, 288, 288]` | 縮小 4 倍 |
| Double buffer | ×2 | ×2 | 總共省 ~1.5 GB |
| Engine activation | ~6.7 GB | ~6.3 GB | 內部 attention 也縮小 |
| 偵測品質 | baseline | 相同 | 0 漏偵 |
| 速度 | baseline | 相同 | bottleneck 在 VE |

## 實驗結果（RTX 5090, 8 cameras, 4 classes）

### VRAM

| 指標 | Q200 | Q50 | 差異 |
|------|------|-----|------|
| **實際 VRAM（nvidia-smi）** | **8,718 MB** | **7,500 MB** | **-1,218 MB (-14%)** |
| Buffer 估算（code 內計算） | 1,998 MB | 1,238 MB | -760 MB |
| Engine weights + activation | ~6,720 MB | ~6,262 MB | -458 MB |

Buffer 層的 760 MB 來自：
- double-buffer masks: 2 × `[8, 200, 288, 288]` × 4B = 2,548 MB → 2 × `[8, 50, 288, 288]` × 4B = 638 MB
- double-buffer boxes/logits: 差異很小（~KB 級別）

Engine activation 的 458 MB 來自 DETR decoder 中 cross-attention 的 Q 維度縮小（200→50）。

### 速度

| 指標 | Q200 | Q50 |
|------|------|-----|
| Avg latency (8 cameras) | 425.9 ms | 424.9 ms |
| Per camera | 53.2 ms | 53.1 ms |
| Throughput | 18.8 FPS | 18.8 FPS |
| p95 | 491.8 ms | 508.5 ms |

速度幾乎一致。原因：bottleneck 是 vision encoder（batch=8 的 forward pass ~100ms），decoder 本身只佔 ~30ms/class，Q50 vs Q200 的 decoder 差異在 noise 範圍內。

### 偵測品質

逐幀比對兩組結果的 `detections_per_camera`，偵測數完全一致。
監控場景（人、手、物品、工具）每幀實際偵測物件數遠小於 50，topK=50 完全夠用。

---

## 如何重現

### 1. 匯出 Q50 decoder ONNX

```bash
# 在 sam3_pytorch 容器內
docker exec sam3_pytorch python3 \
  /root/sam3_pipeline/setup/export_sam3_to_onnx.py \
  --module decoder \
  --model-path facebook/sam3 \
  --output-dir /root/sam3_pipeline/setup/onnx_q50 \
  --num-queries 50 \
  --device cuda
```

只需匯出 decoder（vision/text/geometry encoder 不受 queries 影響）。

### 2. 建 TensorRT engine

```bash
# 在 tensorrt 容器內
docker exec tensorrt trtexec --fp16 \
  --onnx=/root/.../setup/onnx_q50/decoder.onnx \
  --saveEngine=/root/.../engines/b8_q50/decoder.engine \
  --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x1x256,prompt_mask:1x1 \
  --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x33x256,prompt_mask:1x33 \
  --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,prompt_features:8x60x256,prompt_mask:8x60
```

Input shapes 跟 Q200 完全一樣（queries 只影響 output）。

### 3. 跑 pipeline

```bash
# Q200 (baseline)
docker exec tensorrt python3 infer_multi.py \
  --config config.json --video Inputs/media1.mp4 --cameras 8

# Q50
docker exec tensorrt python3 infer_multi.py \
  --config config_q50.json --video Inputs/media1.mp4 --cameras 8
```

`infer_multi.py` 會自動從 decoder engine 的 output shape 偵測 QUERIES 值，不需要手動指定。

---

## Engine 目錄結構

```
engines/
├── b8_q200/                      ← batch=8, queries=200（預設）
│   ├── vision-encoder.engine     #  885 MB
│   ├── text-encoder.engine       #  679 MB
│   ├── geometry-encoder.engine   #   27 MB
│   └── decoder.engine            #   55 MB
├── b8_q50/                       ← batch=8, queries=50（優化版）
│   ├── vision-encoder.engine     → symlink to b8_q200/
│   ├── text-encoder.engine       → symlink to b8_q200/
│   ├── geometry-encoder.engine   → symlink to b8_q200/
│   └── decoder.engine            #   55 MB（獨立建置）
├── b4_q200/                      ← batch=4, queries=200（低 VRAM 版本）
│   └── ...
└── tokenizer.json                ← 共用
```

命名規則：`b{maxBatch}_q{queries}`

只有 decoder 受 queries 影響，其他 3 個 engine 用 symlink 共用，節省 ~1.6 GB 磁碟空間。

---

## 結論

| 方面 | 結論 |
|------|------|
| VRAM | Q50 省 ~1.2 GB (14%)，有意義但不是決定性的 |
| 速度 | 零差異（bottleneck 在 VE） |
| 品質 | 零損失（topK 只去掉背景候選） |
| 改動量 | 只需重匯出 + 重建 decoder engine |
| 建議 | 監控場景建議直接用 Q50 |

如果 VRAM 不是限制因素，Q200 和 Q50 可以共存。透過 `config.json` 的 `engines` 路徑切換即可。

---

## 延伸：與解析度優化組合

Q50 可以跟降解析度疊加使用。在 8 GB VRAM + 8 路攝影機的部署場景中，
`Q50 + 672 解析度` 或 `Q50 + 560 解析度` 是最佳組合：

| 組合 | 8 路 VRAM | 每路 FPS | Config |
|------|:---:|:---:|------|
| Q200 @ 1008 | 8,718 MB | 2.3 | `config.json` |
| Q50 @ 1008 | 7,500 MB | 2.3 | `config_q50.json` |
| Q50 @ 840 | 12,909 MB | 2.2 | `config_q50_r840.json` |
| **Q50 @ 672** | **4,846 MB** | **4.0** | `config_q50_r672.json` |
| **Q50 @ 560** | **4,060 MB** | **4.6** | `config_q50_r560.json` |

> 840 在 8 路模式下 VRAM 反而比 1008 高（12.9 GB > 7.5 GB），
> 因為 multi-camera 的 FPN buffer 在 batch=8 時放大更多。
> 詳見 [`setup/resolution_guide.md`](setup/resolution_guide.md)。
