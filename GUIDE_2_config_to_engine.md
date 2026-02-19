# Config 與 Engine 對照指南

## Config 如何對應 Engine

每個 config JSON 檔的 `"engines"` 欄位指向一個 engine 目錄。不同解析度用不同的 config 檔：

```
config_q50_r448.json  →  engines/b8_q50_r448/
config_q50_r560.json  →  engines/b8_q50_r560/
config_q50_r672.json  →  engines/b8_q50_r672/
config_q50_r840.json  →  engines/b8_q50_r840/
```

切換解析度只需要換 `--config` 參數：

```bash
# 用 448 解析度
python3 infer.py --config config_q50_r448.json --video Inputs/hair.mp4 --output outputs

# 用 840 解析度
python3 infer.py --config config_q50_r840.json --video Inputs/hair.mp4 --output outputs
```

Pipeline 會從 engine 檔自動偵測解析度和 QUERIES 數量，不需要額外設定。

---

## 可用的 Engine 目錄

每個目錄必須包含 **4 個** engine 檔案才能完整運作：

| 目錄 | 解析度 | vision-encoder | text-encoder | geometry-encoder | decoder |
|------|--------|---------------|-------------|-----------------|---------|
| `b8_q50_r448` | 448x448 | 875 MB | 680 MB | 18 MB | 57 MB |
| `b8_q50_r560` | 560x560 | 876 MB | 680 MB | 27 MB | 57 MB |
| `b8_q50_r672` | 672x672 | 878 MB | 680 MB | 28 MB | 57 MB |
| `b8_q50_r840` | 840x840 | 885 MB | 680 MB | 28 MB | 57 MB |

### 命名規則

```
b8_q50_r448
│  │   └── 解析度 448x448
│  └────── quantization 參數（INT8 量化用）
└───────── batch=8（最大支援 8 個 class）
```

### 哪些引擎做什麼

| 引擎 | 用途 | 使用時機 |
|------|------|---------|
| vision-encoder | 將圖片編碼成 FPN 特徵圖 | `extract.py`（image prompt）+ `infer.py` 每幀 |
| text-encoder | 將文字 prompt 編碼成特徵向量 | `extract.py`（text prompt），推論時不需要 |
| geometry-encoder | 將 bounding box 編碼成幾何特徵 | `extract.py`（image prompt），推論時不需要 |
| decoder | 用 FPN + prompt 特徵解碼出偵測結果 | `infer.py` 每幀，每個 class 跑一次 |

推論時只載入 **vision-encoder + decoder**，text-encoder 和 geometry-encoder 只在 `extract.py` 預算特徵時使用。

### 解析度差異（8 cameras, 4 classes, Q50, RTX 5090 實測）

| 解析度 | FPN 尺寸 | Avg ms/round | FPS/cam | Total FPS | Buffers |
|--------|---------|:---:|:---:|:---:|:---:|
| **448** | 128/64/32 | **132 ms** | **7.6** | **60.8** | **246 MB** |
| **560** | 160/80/40 | **147 ms** | **6.8** | **54.3** | **383 MB** |
| 672 | 192/96/48 | 253 ms | 4.0 | 31.7 | — |
| 840 | 240/120/60 | 446 ms | 2.2 | 17.9 | — |

448 比 560 快約 10%，VRAM buffer 省 36%。精度差異需肉眼比較偵測影片。

FPN 尺寸計算公式：`patches = 解析度 / 14`，FPN 三層分別是 `patches×4`、`patches×2`、`patches`。

---

## config.json 結構

```json
{
  "engines": "engines/b8_q50_r448",
  "tokenizer": "engines/tokenizer.json",
  "features": "features",
  "confidence": 0.3,
  "classes": [
    {"name": "person", "prompt_type": "text", "text": "person"},
    {"name": "counter", "prompt_type": "text", "text": "counter"},
    {"name": "hand", "prompt_type": "text", "text": "hand"},
    {"name": "blow_gun", "prompt_type": "image", "references": [...]}
  ]
}
```

| 欄位 | 說明 |
|------|------|
| `engines` | engine 目錄路徑（相對於 config 檔所在目錄） |
| `tokenizer` | tokenizer.json 路徑（所有解析度共用同一個） |
| `features` | 預算特徵快取目錄（所有解析度共用） |
| `confidence` | 預設信心閾值，可被 `--conf` 覆蓋 |
| `classes` | 偵測類別列表（最多 8 個） |

**重點：** 不同解析度的 config 只有 `"engines"` 路徑不同，其他欄位（classes、features、confidence）完全一樣。所以切換解析度只是換 config 檔。

---

## 多攝影機（infer_multi.py）詳細說明

### 影片路徑分配

`--video` 接受多個影片路徑，用空格分隔。影片數量不夠 camera 數時會 **cycle 循環**填滿：

```bash
# 方式 A：1 支影片 → 複製到全部 8 路
--video shop.mp4 --cameras 8
# cam[0-7] 全部都是 shop.mp4

# 方式 B：3 支影片 → cycle 填滿 8 路
--video shop.mp4 hair.mp4 car.mp4 --cameras 8
# cam[0]=shop  cam[1]=hair  cam[2]=car
# cam[3]=shop  cam[4]=hair  cam[5]=car
# cam[6]=shop  cam[7]=hair

# 方式 C：8 支影片 → 一對一
--video v0.mp4 v1.mp4 v2.mp4 v3.mp4 v4.mp4 v5.mp4 v6.mp4 v7.mp4 --cameras 8
# cam[0]=v0  cam[1]=v1  ...  cam[7]=v7
```

### 跑法：Adaptive Real-Time Simulation

**不是逐幀處理，而是模擬真實即時串流的「盡力跑」模式。**

運作方式：
1. 以第一支影片的 FPS 作為 master clock（例如 30 FPS）
2. 每一輪推論 8 個 camera 的畫面
3. 推論完成後，根據花費的時間計算下一輪要讀哪一幀
4. 來不及的幀自動跳過，保持跟影片時間同步

```
影片時間軸:  |0  |1  |2  |3  |4  |5  |6  |7  |8  |9  |10 |11 |...
             ↑           ↑           ↑           ↑
           推論0       推論1       推論2       推論3
           (80ms)      (80ms)      (80ms)      (80ms)
```

如果一輪推論花 80ms，而影片 30 FPS（每幀 33ms），那每輪會跳約 2-3 幀。這就是真實部署時的行為——GPU 盡力跑，跑不完就跳幀。

### 輸出檔案

| 檔案 | 是否預設 | 說明 |
|------|---------|------|
| `{timestamp}_detections.jsonl` | 預設輸出 | 每幀每 camera 的偵測統計 |
| `{timestamp}_performance.json` | 預設輸出 | 速度報告（avg/p95/FPS） |
| `{timestamp}_grid.avi` | 需加 `--save-video` | 2x4 grid overlay 影片 |

**JSON 一定會輸出，影片是可選的。**

`--save-video` 會增加渲染 overhead（畫 overlay + 寫檔），純測速度建議不加。

### detections.jsonl 格式

每行一個 JSON，包含該輪所有 camera 的結果：

```json
{
  "frame_idx": 42,
  "elapsed_ms": 85.3,
  "per_camera_ms": 10.7,
  "cameras": 8,
  "camera_sources": ["shop", "hair", "car", "shop", "hair", "car", "shop", "hair"],
  "detections_per_camera": {
    "shop": 12,
    "hair": 8,
    "car": 5
  }
}
```

8 支 camera 的結果都在同一個 JSONL 裡，用 `camera_sources` 區分。

### performance.json 格式

```json
{
  "cameras": 8,
  "video_sources": ["shop.mp4", "hair.mp4", "car.mp4"],
  "classes": 4,
  "queries": 50,
  "image_size": 448,
  "frames_processed": 150,
  "total_avg_ms": 85.3,
  "per_camera_avg_ms": 10.7,
  "p95_ms": 95.0,
  "total_fps": 93.8,
  "per_camera_fps": 11.7,
  "buffer_vram_mb": 125.0
}
```

---

## 補轉缺少的 Engine

如果某個解析度的 engine 目錄不完整（例如缺 decoder），可以單獨補轉。

### 檢查完整性

```bash
for d in engines/b8_q50_r*/; do
  echo "=== $(basename $d) ==="
  ls -lh "$d"
done
```

每個目錄應該有 4 個 `.engine` 檔案。

### 單獨補轉 Decoder（以 448 為例）

先確認 ONNX 檔案存在：

```bash
ls setup/onnx_r448_q50/decoder.onnx
```

在 Docker 內執行 trtexec：

```bash
# 448: patches=32, FPN=128/64/32
docker exec william_tensorrt trtexec --fp16 \
  --onnx=/root/VisionDSL/models/sam3_pipeline/setup/onnx_r448_q50/decoder.onnx \
  --saveEngine=/root/VisionDSL/models/sam3_pipeline/engines/b8_q50_r448/decoder.engine \
  --minShapes=fpn_feat_0:1x256x128x128,fpn_feat_1:1x256x64x64,fpn_feat_2:1x256x32x32,fpn_pos_2:1x256x32x32,prompt_features:1x1x256,prompt_mask:1x1 \
  --optShapes=fpn_feat_0:1x256x128x128,fpn_feat_1:1x256x64x64,fpn_feat_2:1x256x32x32,fpn_pos_2:1x256x32x32,prompt_features:1x33x256,prompt_mask:1x33 \
  --maxShapes=fpn_feat_0:8x256x128x128,fpn_feat_1:8x256x64x64,fpn_feat_2:8x256x32x32,fpn_pos_2:8x256x32x32,prompt_features:8x60x256,prompt_mask:8x60
```

### FPN 尺寸速查表

trtexec 的 shapes 需要正確的 FPN 尺寸。各解析度對照：

| 解析度 | patches | FPN_0 (×4) | FPN_1 (×2) | FPN_2 (×1) |
|--------|---------|-----------|-----------|-----------|
| 448 | 32 | 128 | 64 | 32 |
| 560 | 40 | 160 | 80 | 40 |
| 672 | 48 | 192 | 96 | 48 |
| 840 | 60 | 240 | 120 | 60 |

### 完整重新轉換

如果要重新轉換整組 4 個引擎，用 `onnx_to_tensorrt.sh`：

```bash
docker exec william_tensorrt bash /root/VisionDSL/models/sam3_pipeline/setup/onnx_to_tensorrt.sh \
  /root/VisionDSL/models/sam3_pipeline/setup/onnx_r448_q50 \
  /root/VisionDSL/models/sam3_pipeline/engines/b8_q50_r448 \
  448
```
