# SAM3 統一偵測管線

獨立的多類別物件偵測管線，支援像素級遮罩（mask）輸出。
每個類別可以用**文字**、**圖片（幾何）**、或**兩者組合**來描述要偵測的目標。

> **第一次使用？** 請先看 [`setup/`](setup/) 資料夾 — 裡面有從零開始建立 Docker 環境、
> 轉換 TensorRT 引擎的完整教學。詳見 [`setup/README.md`](setup/README.md)。

## 資料夾結構

```
sam3_pipeline/
├── setup/                    # 環境建置與引擎轉換（第一次使用請先看這裡！）
│   ├── README.md             #   從零開始的逐步教學
│   ├── export_sam3_to_onnx.py #  PyTorch → ONNX 匯出腳本
│   └── onnx_to_tensorrt.sh   #  ONNX → TensorRT 引擎轉換腳本
├── engines/                  # TensorRT 引擎 + tokenizer（完全獨立，不依賴外部路徑）
│   ├── vision-encoder.engine
│   ├── text-encoder.engine
│   ├── geometry-encoder.engine
│   ├── decoder.engine
│   └── tokenizer.json
├── features/                 # extract.py 自動產生的特徵檔
│   ├── {class_name}/
│   │   ├── features.npy      # [1, prompt_len, 256] float32
│   │   ├── mask.npy           # [1, prompt_len] bool
│   │   └── meta.json
│   └── _meta.json
├── Inputs/                   # 放來源影片和圖片
├── references/               # 放參考圖片（image prompt 用）
├── outputs/                  # 偵測結果輸出
├── config.json               # 類別定義
├── config_editor.py          # 視覺化設定編輯器（獨立工具，見下方說明）
├── extract.py                # 第一步：預先計算 prompt 特徵
└── infer.py                  # 第二步：跑偵測推論
```

## 前置準備

**在跑管線之前**，你需要 TensorRT 引擎和 Docker 環境。
如果還沒有，請先照 [`setup/README.md`](setup/README.md) 的教學完成建置。

## 使用流程

所有指令都在 `sam3_trt` Docker 容器內執行（建立方式見 [`setup/`](setup/)）。

### 第一步：設定要偵測的類別

手動編輯 `config.json`，或使用視覺化編輯器（見下方 [Config 編輯器](#config-編輯器)）：

```json
{
  "engines": "engines",
  "tokenizer": "engines/tokenizer.json",
  "features": "features",
  "confidence": 0.3,
  "classes": [
    {"name": "person", "prompt_type": "text", "text": "person"},
    {"name": "hand",   "prompt_type": "text", "text": "hand"}
  ]
}
```

- `confidence`：信心值門檻，低於此分數的偵測會被過濾
- `classes`：最多 4 個類別

### 第二步：產生 prompt 特徵檔

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/extract.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json
```

這一步會：
1. 根據每個類別的 `prompt_type` 跑對應的編碼器
2. 把特徵存到 `features/{class_name}/` 底下
3. 自動快取 — 如果類別沒改變，重跑會跳過（靠內容 hash 判斷）

**什麼時候需要重跑？** 只有在 `config.json` 裡的類別有變動時才需要。例如改了文字、換了參考圖片、或新增/移除類別。

### 第三步：跑偵測

**偵測單張圖片：**

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --images /root/VisionDSL/models/sam3_pipeline/Inputs/demo_3.jpg \
  --output /root/VisionDSL/models/sam3_pipeline/outputs
```

**偵測多張圖片：**

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --images /root/VisionDSL/models/sam3_pipeline/Inputs/demo_3.jpg \
           /root/VisionDSL/models/sam3_pipeline/Inputs/webcam_1.jpg \
  --output /root/VisionDSL/models/sam3_pipeline/outputs
```

**偵測影片（自適應即時模式）：**

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/media1.mp4 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs
```

影片模式模擬即時播放：每次 inference 完成後，影片時鐘推進實際耗時。GPU 忙碌時到達的幀會被丟棄（drop），就像真實的 camera feed。輸出的 AVI 播放時長 = 原片時長，零飄移。

## 輸出檔案說明

每次執行會產生帶時間戳的檔案（`YYYYMMDD_HHMMSS` 前綴），不同次執行不會互相覆蓋。

**圖片模式：**

| 檔案 | 說明 |
|------|------|
| `{name}_overlay.jpg` | 帶有彩色遮罩疊加 + 標籤的圖片 |
| `{name}_mask_{class}.png` | 每個類別的二值遮罩（需加 `--masks`） |
| `{timestamp}_detections.jsonl` | 每張圖片偵測結果（串流 JSONL） |
| `{timestamp}_performance.json` | 效能統計（平均/最小/最大/p95 毫秒、預估 FPS） |

**影片模式：**

| 檔案 | 說明 |
|------|------|
| `{timestamp}_output.avi` | MJPG 影片，逐幀寫入 overlay 畫面 |
| `{timestamp}_detections.jsonl` | 每幀偵測結果（串流 JSONL，支援 `tail -f` 即時讀取） |
| `{timestamp}_performance.json` | 效能統計 |

## 指令參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--config` | （必填） | config.json 路徑 |
| `--images` | — | 圖片路徑（和 --video 二選一） |
| `--video` | — | 影片路徑 |
| `--output` | `outputs` | 輸出目錄 |
| `--conf` | config 裡的值 | 覆蓋信心值門檻 |
| `--interval` | `1` | 最小幀間距；1=GPU 自適應，N=最多每 N 幀處理一幀 |
| `--masks` | 否 | 儲存每個類別的 mask PNG |

## Prompt 類型詳解

### 文字 prompt（text）

用文字描述目標物件。最簡單的方式。

```json
{"name": "person", "prompt_type": "text", "text": "person"}
```

### 圖片 prompt（image）

用參考圖片 + 框選位置來告訴模型「我要找的是這個東西」。
框的座標格式是**正規化的 cxcywh**：`[中心x, 中心y, 寬, 高]`，數值範圍 0.0 到 1.0。
標籤：`1` = 正樣本（這是目標），`0` = 負樣本（這不是）。

```json
{
  "name": "my_cup",
  "prompt_type": "image",
  "references": [
    {"image": "references/cup.jpg", "boxes": [[0.5, 0.5, 0.3, 0.4]], "labels": [1]}
  ]
}
```

參考圖片放在 `references/` 資料夾內。

### 組合 prompt（both）

同時使用文字描述 + 參考圖片，兩者的特徵會串接在一起。
適合用在「文字不夠精確，但加上視覺範例就可以鎖定」的情境。

```json
{
  "name": "red_cup",
  "prompt_type": "both",
  "text": "cup",
  "references": [
    {"image": "references/red_cup.jpg", "boxes": [[0.5, 0.5, 0.3, 0.4]], "labels": [1]}
  ]
}
```

## 限制

| 限制 | 值 |
|------|-----|
| 最多類別數 | 8（decoder 的 batch 維度上限；可調整，見 [`setup/README.md`](setup/README.md)） |
| 最大 prompt token 數 | 60（text=32，每個 geo box=2 tokens） |
| 每個類別最多參考框數 | 20 |

## 效能與 VRAM（RTX 5090 實測）

| 類別數 | 引擎 batch | 平均耗時/幀 | 預估 FPS | VRAM 佔用 |
|--------|-----------|------------|----------|----------|
| 4（3 text + 1 image） | maxShapes=4 | ~70 ms | ~14 | **~5.0 GB** |
| 4（3 text + 1 image） | maxShapes=8 | ~70 ms | ~14 | **~7.5 GB** |

- 首幀因為包含暖機會較慢（~100–350 ms），後續幀穩定
- VRAM 主要被 TensorRT 的 activation memory 吃掉，在引擎載入時就會按 `maxShapes` 預先分配
- 實際用幾個類別對 VRAM 影響極小 — 引擎會為最大 batch 預留記憶體，不管你實際用了幾個
- 要改變最大類別數，需要用不同的 batch 重建引擎（見 [`setup/README.md`](setup/README.md)）

## 整體流程圖

```
config.json  ──>  extract.py  ──>  features/   ──>  infer.py  ──>  outputs/
 （定義類別）     （跑一次）     （快取特徵）     （每幀推論）     （輸出結果）
```

**重點：** `extract.py` 用到 text/geometry encoder，但 `infer.py` 只載入 vision encoder + decoder，所以推論時很快且佔用資源少。

## Config 編輯器

視覺化的 `config.json` 編輯工具。特別適合 **image prompt** — 可以在參考圖片上直接框選目標，自動計算正規化的 cxcywh 座標。

- **零依賴** — 純 Python（`http.server`），單一檔案，不需要 pip install
- **不需要 Docker** — 直接在本機執行
- **完全獨立** — 只讀寫 `config.json` 和提供 references/ 裡的圖片，不影響管線運作

### 使用方式

```bash
# 在本機直接執行（不需要進 Docker）
python3 /home/ubuntu/Documents/willy/repos/william/VisionDSL/models/sam3_pipeline/config_editor.py \
  --config /home/ubuntu/Documents/willy/repos/william/VisionDSL/models/sam3_pipeline/config.json
```

然後在瀏覽器打開 `http://localhost:8080`。

### 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--config` | 腳本同目錄的 `config.json` | config.json 路徑 |
| `--port` | `8080` | HTTP 埠號 |

### 功能

- 新增/刪除類別（最多 8 個），編輯名稱和 prompt 類型
- 文字 prompt：直接編輯文字欄位
- 圖片 prompt：選擇參考圖片，在畫布上拖曳框選 bounding box
- 每個框可以切換正樣本/負樣本標籤
- 正規化的 cxcywh 座標自動計算
- 按鈕或 `Ctrl+S` 儲存，有未儲存狀態提示
- 深色主題 UI

## 常見問題

**Q: 改了類別要重跑哪個？**
A: 只要重跑 `extract.py`，它會自動跳過沒改的類別，只重算改過的。

**Q: image prompt 效果如何？**
A: geometry encoder 原本設計是同張圖片內使用，跨圖片的參考效果需要實測。建議先用 text prompt 確認管線正常，再試 image prompt。

**Q: 可以超過 8 個類別嗎？**
A: 可以。修改 `setup/onnx_to_tensorrt.sh` 裡所有引擎的 maxShapes batch 數字（例如改成 16），重新轉換引擎即可。不需要重新訓練模型，也不需要重新匯出 ONNX。代價是 VRAM 會增加。詳見 [`setup/README.md`](setup/README.md) 的「最大類別數」章節。

**Q: 影片輸出太多檔案怎麼辦？**
A: 用 `--interval` 減少處理幀數。Mask PNG 預設關閉，只有加 `--masks` 才會輸出。
