# SAM3 Pipeline 操作指南

## 整體工作流程

```
config.json  ──>  extract.py  ──>  features/  ──>  infer.py  ──>  outputs/
  (定義類別)       (一次性預算)      (快取)         (每幀推論)      (結果)
```

- `extract.py` 需要 text encoder / vision encoder / geometry encoder（三個引擎）
- `infer.py` 只需要 vision encoder + decoder（兩個引擎），速度快

全部推論指令在 Docker 容器 `william_tensorrt` 裡面執行。

---

## 四個腳本說明

### 1. `config_editor.py` — 視覺化設定編輯器

**在 host 上跑，不需要 Docker，不需要 GPU。**

用途：新增/刪除 class、選擇 prompt type、在參考圖上畫 bounding box、切換正/負樣本。

```bash
# 啟動（預設 port 8080）
python3 /home/ubuntu/Documents/willy/repos/william/VisionDSL/models/sam3_pipeline/config_editor.py \
  --config /home/ubuntu/Documents/willy/repos/william/VisionDSL/models/sam3_pipeline/config.json

# 指定 port
python3 config_editor.py --config config.json --port 9090
```

開瀏覽器到 `http://localhost:8080`：

- 左側 sidebar 管理 classes（最多 8 個）
- 選 prompt type：`text` / `image` / `both`
- image/both 模式下可以選 `references/` 裡的圖片，直接在圖上拖拉畫框
- 自動算出 cxcywh 正規化座標（0.0~1.0）
- 每個框可以切換 pos（正樣本）/ neg（負樣本）
- `Ctrl+S` 或按鈕儲存

---

### 2. `extract.py` — 預算 Prompt Features

**在 Docker 內執行。每次修改 `config.json` 後跑一次。**

沒改過的 class 會自動跳過（根據內容 hash 快取）。

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/extract.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json
```

根據每個 class 的 `prompt_type` 做不同處理：

| prompt_type | 使用的引擎 | 輸出 shape |
|-------------|-----------|-----------|
| `text` | text encoder | `[1, 32, 256]` |
| `image` | vision encoder + geometry encoder | `[1, N+1, 256]`（N = box 數量） |
| `both` | 三個都用，text + geometry 串接 | `[1, 32+N+1, 256]` |

產出檔案存在 `features/{class_name}/`：
- `features.npy` — prompt features
- `mask.npy` — attention mask
- `meta.json` — hash、prompt_len 等 metadata

---

### 3. `infer.py` — 單機推論

**在 Docker 內執行。** 只載入 vision encoder + decoder。

**單張圖片：**

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --images /root/VisionDSL/models/sam3_pipeline/Inputs/demo_3.jpg \
  --output /root/VisionDSL/models/sam3_pipeline/outputs
```

**影片（自適應即時模擬）：**

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/hair.mp4 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs
```

**參數：**

| Flag | 預設 | 說明 |
|------|------|------|
| `--config` | （必填） | config.json 路徑 |
| `--images` | — | 圖片路徑（和 --video 二選一） |
| `--video` | — | 影片路徑 |
| `--output` | `outputs` | 輸出目錄 |
| `--conf` | config 裡的值 | 覆蓋信心閾值 |
| `--interval` | `1` | 最小幀間距；1=GPU 自適應，N=最多每 N 幀處理一次 |
| `--masks` | `false` | 儲存每個 class 的 binary mask PNG |

**輸出檔案（以時間戳命名，不會互相覆蓋）：**

- `{timestamp}_output.avi` — overlay 影片
- `{timestamp}_detections.jsonl` — 每幀偵測結果
- `{timestamp}_performance.json` — 效能統計

---

### 4. `infer_multi.py` — 多攝影機推論

**在 Docker 內執行。** 8 路攝影機同時處理。

```bash
# 單一影片複製到 8 路
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer_multi.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/hair.mp4 \
  --cameras 8 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs

# 多個影片循環填滿（3 支影片 → 8 路：A,B,C,A,B,C,A,B）
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer_multi.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --video Inputs/shop.mp4 Inputs/hair.mp4 Inputs/car.mp4 \
  --cameras 8

# 加上 grid 影片輸出（2x4 排列，方便檢視）
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer_multi.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --video Inputs/hair.mp4 \
  --cameras 8 --save-video
```

| Flag | 預設 | 說明 |
|------|------|------|
| `--config` | （必填） | config.json 路徑 |
| `--video` | （必填） | 影片路徑（可多個，空格分隔） |
| `--cameras` | `8` | 攝影機數量 |
| `--save-video` | `false` | 輸出 2x4 grid overlay AVI |

---

## 三種 Prompt Type

### `text` — 文字描述

用一個詞或短語描述物件，透過 text encoder 產生語意特徵。

```json
{"name": "person", "prompt_type": "text", "text": "person"}
```

**適合：** 通用物件（person、hand、cup、chair 等常見概念）。

### `image` — 圖片參考（geometry）

在參考圖片上用 bounding box 指出目標物件，透過 vision encoder + geometry encoder 產生視覺特徵。

Boxes 格式是 **正規化 cxcywh**：`[center_x, center_y, width, height]`，值域 0.0~1.0。

```json
{
  "name": "my_tool",
  "prompt_type": "image",
  "references": [
    {
      "image": "references/中華汽車/image9.JPG",
      "boxes": [[0.45, 0.75, 0.15, 0.25]],
      "labels": [1]
    }
  ]
}
```

- `labels: [1]` = 正樣本（這是目標物件）
- `labels: [0]` = 負樣本（這不是目標物件）

**適合：** 形狀獨特、不容易跟背景混淆的特殊物件。

### `both` — 混合模式（文字 + 圖片）

文字和圖片特徵串接在一起。**文字提供語意方向，圖片提供視覺參考，互補。**

```json
{
  "name": "blow_gun",
  "prompt_type": "both",
  "text": "air blow gun",
  "references": [
    {
      "image": "references/中華汽車/image9.JPG",
      "boxes": [[0.45, 0.75, 0.15, 0.25]],
      "labels": [1]
    }
  ]
}
```

**適合：** 特殊物件 + 容易跟背景混淆（例如噴槍框太大包到地板）。

---

## 問題排解：Bounding Box 框太大包到背景

### 問題

用 `image` prompt 框噴槍時，因為噴槍很小，bounding box 不可避免地包到大面積的地板/桌面。模型把背景的特徵當成目標，導致偵測到的是地板而不是噴槍。

### 解法一：改用 `both` 模式

把 `prompt_type` 從 `image` 改成 `both`，加上 `text` 欄位：

```json
{
  "name": "blow_gun",
  "prompt_type": "both",
  "text": "air blow gun",
  "references": [
    {
      "image": "references/中華汽車/image13.png",
      "boxes": [[0.21, 0.86, 0.37, 0.16]],
      "labels": [1]
    },
    {
      "image": "references/中華汽車/image2.png",
      "boxes": [[0.24, 0.93, 0.42, 0.14]],
      "labels": [1]
    }
  ]
}
```

text 語意會把模型的注意力拉向「blow gun」這個概念，即使 box 不完美也能正確辨識。

### 解法二：加負樣本排除背景

在同一張參考圖上，額外畫一個框在**地板區域**，label 設為 `0`（negative）：

```json
{
  "image": "references/中華汽車/image9.JPG",
  "boxes": [
    [0.45, 0.75, 0.15, 0.25],
    [0.50, 0.95, 0.80, 0.08]
  ],
  "labels": [1, 0]
}
```

第一個框（label 1）：噴槍的位置
第二個框（label 0）：地板，告訴模型「這不是目標」

### 解法三：多張參考圖 + 緊框

提供多張不同角度的參考圖，每張都盡量畫最緊的框。多個參考的 geometry features 會合併（第一張完整保留，後續的去掉 CLS token 再串接）。

### 最佳實踐（組合使用）

```json
{
  "name": "blow_gun",
  "prompt_type": "both",
  "text": "air blow gun",
  "references": [
    {
      "image": "references/中華汽車/image9.JPG",
      "boxes": [
        [0.45, 0.75, 0.15, 0.25],
        [0.50, 0.95, 0.80, 0.08]
      ],
      "labels": [1, 0]
    },
    {
      "image": "references/中華汽車/image13.png",
      "boxes": [
        [0.21, 0.86, 0.37, 0.16]
      ],
      "labels": [1]
    }
  ]
}
```

三招組合：
1. `both` 模式 → text 錨定語意
2. 負樣本框 → 排除地板
3. 多張參考圖 → 增加視覺多樣性

---

## 什麼時候用哪種模式

| 模式 | 適合場景 | 範例 |
|------|---------|------|
| `text` | 通用物件，一個詞就能描述 | person、hand、cup、chair |
| `image` | 特殊/罕見物件，形狀獨特，box 能框得很精準 | 特定零件、獨特圖案 |
| `both` | 特殊物件 + box 容易包到背景 / 語意模糊 | 噴槍（小物件混在複雜背景中） |

---

## 完整操作範例：從頭偵測噴槍

### Step 1：準備參考圖

把噴槍的截圖放到 `references/` 目錄下：

```
references/
└── 中華汽車/
    ├── image9.JPG    ← 有噴槍的截圖
    ├── image13.png   ← 另一個角度
    └── ...
```

### Step 2：編輯 config.json

用 config_editor 或直接手改：

```bash
# 方法 A：用視覺編輯器
python3 config_editor.py --config config.json
# 瀏覽器開 http://localhost:8080
# 點 blow_gun → prompt type 改 both → 填 text → 選圖畫框 → Save

# 方法 B：直接編輯 config.json（見上面的 JSON 範例）
```

### Step 3：產生 features

```bash
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/extract.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json
```

預期輸出：
```
  person: cached (len=32)
  counter: cached (len=32)
  hand: cached (len=32)
  blow_gun: extracting (both)...
    both  -> text 32 + geo 4 = [1, 36, 256]
    saved -> features/blow_gun  (len=36)
```

### Step 4：跑推論

```bash
# 用 hair.mp4 測試
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/hair.mp4 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs

# 或用單張圖測試（更快看結果）
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config.json \
  --images /root/VisionDSL/models/sam3_pipeline/Inputs/demo_3.jpg \
  --output /root/VisionDSL/models/sam3_pipeline/outputs
```

### Step 5：看結果

輸出在 `outputs/` 目錄：
- `*_overlay.jpg` 或 `*_output.avi` — 帶偵測框和 mask 的視覺化
- `*_detections.jsonl` — JSON 格式偵測結果
- `*_performance.json` — 效能數據

---

## Docker 路徑對照

| Host 路徑 | Docker 路徑 |
|-----------|------------|
| `/home/ubuntu/Documents/willy/repos/william/VisionDSL` | `/root/VisionDSL` |
| `.../VisionDSL/models/sam3_pipeline` | `/root/VisionDSL/models/sam3_pipeline` |
| `.../VisionDSL/models/sam3_pipeline/Inputs/hair.mp4` | `/root/VisionDSL/models/sam3_pipeline/Inputs/hair.mp4` |

---

## 限制

| 項目 | 上限 |
|------|------|
| 最大 class 數量 | 8（decoder batch 維度） |
| 最大 prompt tokens | 60（text=32，每個 geo box=2 tokens） |
| 每個 class 最大 box 數量 | 20 |
