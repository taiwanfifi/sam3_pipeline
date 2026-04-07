# 決策：採用 r434 + 2 Classes (PERSON + HAND)

日期：2026-03-20

---

## 結論

- **解析度**：r434（14×31 patches）取代原本的 r448（14×32 patches）
- **Classes**：從 4 class（PERSON, HEAD, HAND, SCREWDRIVER）縮減為 2 class（PERSON, HAND）
- **推薦 config**：`models/sam3_pipeline/config_2cls_r434.json`

---

## 為什麼選 r434

### 單路推論（1 camera, bbox 模式, 含完整 segmentation）

| 解析度 | Patches | FPS | vs r448 |
|:---:|:---:|:---:|:---:|
| r448 | 32 | 11.3 | 基準 |
| **r434** | **31** | **17.0** | **+50%** |
| r420 | 30 | 14.2 | +26% |
| r406 | 29 | 17.3 | +53% |
| r308 | 22 | 18.0 | +59% |

r434 只少 1 個 patch，FPS 直接 +50%。r420 反而比 r434 慢（TensorRT kernel 優化差異）。

### 8 路推論（8 cameras, real-time skip frames 模式）

| 解析度 | avg ms/round | FPS/cam | 總 FPS | VRAM Buffers |
|:---:|:---:|:---:|:---:|:---:|
| r448 | 120 ms | 8.4 | 66.9 | 245 MB |
| **r434** | **116 ms** | **8.6** | **69.2** | **230 MB** |
| r420 | 119 ms | 8.4 | 67.1 | 215 MB |
| r406 | 113 ms | 8.8 | 70.7 | 201 MB |

8 路下差距被 VE batch=8 攤平，但 r434 仍然不輸 r448。

### 偵測品質

r434 vs r448 的 segmentation mask overlay 影片肉眼看不出差異（只差 1 個 patch = 14 pixels）。
比較影片在 `VisionDSL/outputs/resolution_compare/r434_2cls.mp4` 和 `r448_2cls.mp4`。

---

## 2 Classes 的影響

SAM3 decoder 是 batched inference（所有 class 一次 forward pass），不是每個 class 跑一次。
減少 class 數量的影響：

- Vision Encoder（佔 40-60% 時間）：**完全不變**，只跑 1 次
- Decoder batch 從 4→2：**省 ~10-15% decoder 時間**
- 不需要重編 engine，只改 config JSON 的 `classes` 陣列

---

## 關鍵路徑

### Config 檔案

```
# 推薦使用（r434, 2 classes）
VisionDSL/models/sam3_pipeline/config_2cls_r434.json

# 其他可用的 2-class configs
VisionDSL/models/sam3_pipeline/config_2cls_r448.json
VisionDSL/models/sam3_pipeline/config_2cls_r406.json
VisionDSL/models/sam3_pipeline/config_2cls_r420.json
VisionDSL/models/sam3_pipeline/config_2cls_r308.json
VisionDSL/models/sam3_pipeline/config_2cls_r280.json  # 已刪除 engine，僅剩 config

# 原有的 4-class configs（仍可用）
VisionDSL/models/sam3_pipeline/config.json          # r560, 4cls
VisionDSL/models/sam3_pipeline/config_r448.json     # r448, 4cls
VisionDSL/models/sam3_pipeline/config_r560.json     # r560, 4cls
VisionDSL/models/sam3_pipeline/config_bakery.json   # r448, 4cls (bakery)
```

### TensorRT Engines

```
VisionDSL/models/sam3_pipeline/engines/
├── b8_q50_r308/     # 308×308
├── b8_q50_r406/     # 406×406
├── b8_q50_r420/     # 420×420
├── b8_q50_r434/     # 434×434  ← 推薦
├── b8_q50_r448/     # 448×448  (原有)
├── b8_q50_r560/     # 560×560  (原有)
├── b8_q50_r672/     # 672×672  (原有)
└── tokenizer.json

# 已刪除的 engines（解析度太低，品質差且反而更慢）：
# r154, r196, r252, r280
```

### ONNX 檔案（用於重編 engine）

```
VisionDSL/models/sam3_pipeline/setup/
├── onnx_r308_q50/
├── onnx_r406_q50/
├── onnx_r420_q50/
├── onnx_r434_q50/   ← 推薦
└── onnx_r448_q50/   (原有)
```

### 比較影片

```
VisionDSL/outputs/resolution_compare/
├── r448_2cls.mp4    # 基準
├── r434_2cls.mp4    # 推薦（品質與 r448 幾乎一樣）
├── r420_2cls.mp4
├── r406_2cls.mp4
├── r308_2cls.mp4
├── r280_2cls.mp4
├── r252_2cls.mp4
├── r196_2cls.mp4
├── r154_2cls.mp4
└── r154_2cls_v2.mp4
```

### Docker 容器

| 容器名稱 | 用途 | GPU | PyTorch | 掛載路徑 |
|----------|------|:---:|:---:|----------|
| `sam3` | ONNX 匯出（有 PyTorch + SAM3 weights） | ✅ | ✅ | `/root/willy/repos/william/VisionDSL/` |
| `william_sam3` | 同上但**無 GPU** | ❌ | ✅ | `/root/VisionDSL/` |
| `william_tensorrt` | TensorRT 編譯 + 推論 | ✅ | ❌ | `/root/VisionDSL/` |
| `tensorrt` | TensorRT（備用） | ✅ | ✅ | `/root/willy/repos/william/VisionDSL/` |

**ONNX 匯出必須用 `sam3` 容器**（有 GPU + PyTorch + SAM3 model weights）。
**TensorRT 編譯和推論用 `william_tensorrt` 容器**。

### 編譯新解析度的完整流程

```bash
# 1. ONNX 匯出（在 sam3 容器，需 GPU + PyTorch）
docker exec sam3 bash -c "cd /root/willy/repos/william/VisionDSL/models/sam3_pipeline && \
  MODEL=setup/sam3_pretrained/snapshots/3c879f39826c281e95690f02c7821c4de09afae7 && \
  python3 setup/export_sam3_to_onnx.py --module vision --model-path \$MODEL --image-size <RES> --output-dir setup/onnx_r<RES>_q50 --device cuda && \
  python3 setup/export_sam3_to_onnx.py --module decoder --model-path \$MODEL --image-size <RES> --num-queries 50 --output-dir setup/onnx_r<RES>_q50 --device cuda && \
  python3 setup/export_sam3_to_onnx.py --module geometry --model-path \$MODEL --image-size <RES> --output-dir setup/onnx_r<RES>_q50 --device cuda && \
  ln -sf ../onnx_r448_q50/text-encoder.onnx setup/onnx_r<RES>_q50/text-encoder.onnx"

# 2. TensorRT 編譯（在 william_tensorrt 容器）
docker exec william_tensorrt bash -c "cd /root/VisionDSL/models/sam3_pipeline && \
  bash setup/onnx_to_tensorrt.sh setup/onnx_r<RES>_q50 engines/b8_q50_r<RES> <RES>"

# 3. 建立 config JSON（解析度必須是 14 的倍數）
```

### Mask Overlay 繪圖說明

- SAM3 推論**永遠輸出 segmentation mask**（不論有沒有畫到畫面上）
- bbox 是從 mask contour 重新計算的（`_bbox_from_mask()`），比 decoder 原始 bbox 更精準
- mask overlay 繪圖是**純 CPU 操作**（numpy alpha blend + cv2.findContours），不在 GPU 上
- 開啟 `--mask-overlay` 會額外消耗 CPU 時間（resize mask + blend + contour），約降低 30-40% FPS

---

## 實驗記錄

### 被淘汰的低解析度（r154 ~ r280）

| 解析度 | 問題 |
|:---:|------|
| r154 (11 patches) | FPS 反而比 r448 慢（5.7 vs 5.9），false positive 暴增，檔案更大 |
| r196 (14 patches) | 類似 r154，品質崩潰 |
| r252 (18 patches) | 單路 11.2 FPS 最快，但 8 路差距被攤平 |
| r280 (20 patches) | 單路 9.4 FPS，無明顯優勢 |

**結論：低於 r308 的解析度不值得，false positive 反噬效能。**
已刪除 r154/r196/r252/r280 的 engine 和 ONNX 檔案。
