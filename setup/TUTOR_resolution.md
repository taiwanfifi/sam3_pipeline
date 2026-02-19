# 解析度調整指南

這份指南教你理解輸入解析度如何影響 SAM3 的速度、VRAM 和偵測品質，
以及如何為不同部署環境選擇最適合的解析度。

---

## 目錄

1. [解析度如何影響 SAM3](#1-解析度如何影響-sam3)
2. [為什麼降解析度可以省這麼多 VRAM？](#2-為什麼降解析度可以省這麼多-vram)
3. [實測數據](#3-實測數據)
4. [偵測品質比較](#4-偵測品質比較)
5. [怎麼選解析度？決策框架](#5-怎麼選解析度決策框架)
6. [實作：匯出不同解析度的 ONNX + 建 engine](#6-實作匯出不同解析度的-onnx--建-engine)
7. [常見問題](#7-常見問題)

---

## 1. 解析度如何影響 SAM3

SAM3 的 Vision Encoder (VE) 是一個 ViT（Vision Transformer）。
它把輸入圖片切成很多小方塊（patch），每個 patch 14×14 pixels。

```
輸入圖片 1008×1008
    ↓ 切成 14×14 的 patch
72 × 72 = 5,184 個 patch
    ↓ 每個 patch 變成一個 256 維向量
[5184, 256] 的特徵矩陣
    ↓ 32 層 Transformer（包含 Self-Attention）
FPN 輸出（多尺度特徵圖）
```

**關鍵公式：**
```
patches = 解析度 / 14

1008 → 72 patches
 840 → 60 patches
 672 → 48 patches
 560 → 40 patches
```

解析度必須是 14 的倍數（因為 patch_size = 14）。

### 為什麼 patches 數量這麼重要？

因為 Transformer 的 Self-Attention 計算量跟 patches 的**平方**成正比：

```
Self-Attention:
  Q × K^T = [patches², patches²]

  1008: 5184 × 5184 = 26,873,856 個乘法
   840: 3600 × 3600 = 12,960,000 個乘法  (↓52%)
   672: 2304 × 2304 =  5,308,416 個乘法  (↓80%)
   560: 1600 × 1600 =  2,560,000 個乘法  (↓90%)
```

但注意：SAM3 的 VE 有 28 個 windowed attention 層（固定 24×24 窗口）和 4 個 global attention 層。
**只有 global 層的計算量跟總 patches 有關**，windowed 層的計算量是固定的。

所以實際的加速不是簡單的 patches⁴ 比例，但趨勢是對的——解析度越低，速度越快。

---

## 2. 為什麼降解析度可以省這麼多 VRAM？

VRAM 的佔用來自三個部分：

### 2.1 Engine weights（模型權重）

這部分跟解析度**無關**。SAM3 VE 有 ~440M 參數，FP16 下固定佔 ~880 MB。
不管輸入 1008 還是 560，weights 都一樣大。

### 2.2 Activation memory（中間計算結果）

這是最大的變數。TensorRT 需要為每一層的中間結果預留 GPU 記憶體。

```
FPN 特徵圖大小:
  FPN_0 = [batch, 256, patches×4, patches×4]    ← 最大的一層
  FPN_1 = [batch, 256, patches×2, patches×2]
  FPN_2 = [batch, 256, patches,   patches  ]

以 batch=8 為例:
  1008: FPN_0 = [8, 256, 288, 288] = 8 × 256 × 288 × 288 × 2 bytes ≈ 340 MB
   672: FPN_0 = [8, 256, 192, 192] = 8 × 256 × 192 × 192 × 2 bytes ≈ 151 MB
   560: FPN_0 = [8, 256, 160, 160] = 8 × 256 × 160 × 160 × 2 bytes ≈ 105 MB
```

FPN_0 就差了 2 倍多。而 TensorRT 內部的 activation 還有更多中間層，加總起來差距更大。

### 2.3 Pipeline buffers（推論時的 I/O 緩衝）

我們自己配置的 buffer（輸入圖片、FPN 複製、output 等）：

| 解析度 | buffer_vram_mb | 說明 |
|:---:|:---:|---|
| 1008 | 632 MB | 包含 batched FPN、masks、boxes 等 |
| 840 | 439 MB | ↓31% |
| 672 | 281 MB | ↓56% |
| 560 | 195 MB | ↓69% |

### 總 VRAM（nvidia-smi 實測）

```
Engine weights + Activation memory + Pipeline buffers = 總 VRAM

1008: ~880 + ~5550 + 632 = 7,064 MB
 840: ~880 + ~4730 + 439 = 6,050 MB
 672: ~880 + ~3420 + 281 = 4,582 MB
 560: ~880 + ~2800 + 195 = 3,874 MB
```

（Activation memory 是用總 VRAM 反推的估計值）

---

## 3. 實測數據

### 3.1 單路推理（`infer.py`, RTX 5090, 4 classes, FP16, Q50）

| 解析度 | Patches | **VRAM** | **avg_ms** | **FPS** | p95_ms | 處理幀數 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1008** | 72 | 7,064 MB | 62.5 ms | 16.0 | 87.2 | 2,325 |
| **840** | 60 | 6,050 MB | 50.3 ms | 19.9 | 73.6 | 2,890 |
| **672** | 48 | 4,582 MB | 35.1 ms | 28.5 | 59.8 | 3,866 |
| **560** | 40 | 3,874 MB | 28.4 ms | 35.2 | 46.6 | 4,107 |

### 3.2 多路推理（`infer_multi.py`, RTX 5090, 8 cameras, 4 classes, FP16, Q50）

這才是真正的部署場景——8 路攝影機同時跑（3 支不同影片 shop/hair/car 循環填 8 路）。
VE batch=8 一次處理 8 張 frame，decoder 逐 class 迭代。

| 解析度 | 每輪 8 張 avg | 每路 avg | **每路 FPS** | 總 FPS | **實測 VRAM** | P95 | 8GB 可行？ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **840** | 446 ms | 55.8 ms | **2.2** | 17.9 | **12,909 MB** | 612 ms | **不可行** |
| **672** | 253 ms | 31.6 ms | **4.0** | 31.7 | **4,846 MB** | 319 ms | 可行（餘 3.2 GB） |
| **560** | 216 ms | 27.0 ms | **4.6** | 37.0 | **4,060 MB** | 271 ms | 可行（餘 3.9 GB） |

> **重要發現**：840 在 8 路模式下需要 12.9 GB VRAM，8 GB 顯卡跑不了。
> 672 和 560 都能塞進 8 GB，差距只有 15%（4.0 vs 4.6 FPS/cam）。

**多路 benchmark 指令**：

```bash
# 672 解析度，8 路攝影機，3 支影片循環
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer_multi.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_q50_r672.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/shop.mp4 \
         /root/VisionDSL/models/sam3_pipeline/Inputs/hair.mp4 \
         /root/VisionDSL/models/sam3_pipeline/Inputs/car.mp4 \
  --cameras 8 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/multi_r672
```

**量 VRAM**（在另一個 terminal）：
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### 3.3 相對比較（以 1008 為基準）

| 解析度 | VRAM (單路) | 速度 (單路) | VRAM (8路) | 速度 (8路) | 偵測品質 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1008** | 100% | 100% | N/A (超出) | N/A | 最佳 |
| **840** | 86% | 125% | 100% (基準) | 100% | 接近 1008 |
| **672** | 65% | 178% | 38% | 177% | 主要物件 OK |
| **560** | 55% | 220% | 31% | 207% | 大物件 OK |

---

## 4. 偵測品質比較

### 同一張圖（demo_3.jpg, 商店場景）

| 解析度 | 偵測數 | person | hand | blow_gun | counter |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1008 | 55 | 偵測到 | 偵測到 | 很多（含誤判） | 偵測到 |
| 840 | 59 | 偵測到 | 偵測到 | 減少 | 偵測到 |
| 672 | 69 | 偵測到 | 偵測到 | 減少 | 偵測到 |
| 560 | 54 | 偵測到 | 偵測到 | 最少 | 偵測到 |

### 觀察

1. **Person 偵測**：四種解析度都穩定偵測到。因為人在監控畫面中通常夠大（佔畫面 20%+），
   即使 560（40 個 patch），一個人也能佔到 5-10 個 patch，足以偵測。

2. **Hand 偵測**：四種都偵測到，但低解析度的 bbox 位置可能不夠精確。
   手比人小很多，在 560 下可能只佔 2-3 個 patch。

3. **Mask 品質**：低解析度的 mask 邊緣會比較粗糙（因為 FPN 特徵圖解析度降低）。
   如果你需要精確的物件輪廓，建議保持 840 以上。

4. **誤判（False Positive）**：低解析度反而有更多偵測，
   可能是因為模型看到的細節不夠，把相似的東西誤判為目標。

### 什麼場景下品質差異最大？

```
影響小的場景（低解析度 OK）：
  - 近距離（2-5 公尺）的監控
  - 偵測大物件（人、車）
  - 只需要 bbox，不需要精確 mask
  - VisionDSL 的 zone presence 判斷

影響大的場景（需要高解析度）：
  - 遠距離（10+ 公尺）或廣角鏡頭
  - 偵測小物件（手、工具、食物）
  - 需要精確的 mask 輪廓
  - 密集場景（很多人疊在一起）
```

---

## 5. 怎麼選解析度？決策框架

### 第一步：你有多少 VRAM？跑幾路攝影機？

**單路推理**（`infer.py`，用於開發、測試、單台攝影機場景）：
```
  8 GB VRAM:
    1008 → ⚠️ 可行但極限 (7.0 GB)
     840 → ✅ (6.0 GB)
     672 → ✅ 推薦 (4.6 GB，留餘量)
     560 → ✅ 最安全 (3.9 GB)
```

**8 路多路推理**（`infer_multi.py`，正式部署場景）：
```
  8 GB VRAM:
    1008 → ❌ 超出
     840 → ❌ 需 12.9 GB，完全超出
     672 → ✅ 推薦 (4.8 GB，餘 3.2 GB 給 OS/tracker/DSL)
     560 → ✅ 最安全 (4.1 GB，餘 3.9 GB)

  12 GB VRAM:
     840 → ✅ 推薦（12.9 GB 勉強可行）
     672 → ✅ 最佳平衡

  16+ GB VRAM:
    1008 → ✅ 推薦（需實測 8 路 VRAM）
     840 → ✅
```

### 第二步：你的場景需要多少細節？

```
  SOP 驗證（人站在哪、做了什麼動作）→ 672 夠用
  人臉辨識 → 不適合 SAM3，用專門的模型
  手部操作細節 → 840 以上
  工業瑕疵檢測 → 1008
```

### 第三步：你需要多少 FPS？

```
  即時監控（需要 30 FPS）→ 560 或 672
  準即時（需要 15 FPS）→ 840 或 1008
  離線分析（不在意速度）→ 1008
```

### 推薦組合

| 部署環境 | 攝影機數 | 推薦解析度 | 理由 |
|----------|:---:|:---:|------|
| RTX 5090 (32 GB) | 8 路 | **1008** | VRAM 充裕，用最高品質 |
| RTX 4090 (24 GB) | 8 路 | **840** | 12.9 GB 可行，品質平衡 |
| RTX 4060 (8 GB) | 8 路 | **672** | 4.8 GB，餘量充足，4.0 FPS/cam |
| RTX 4060 (8 GB) | 8 路 | **560** | 4.1 GB，最大餘量，4.6 FPS/cam |
| Jetson Orin (8 GB) | 4 路 | **560** | VRAM 和算力都有限 |

> **8 GB 部署結論**：672 和 560 都可行。如果偵測小物件（手、工具），選 672；
> 只偵測大物件（人、櫃台），選 560（多 15% FPS 和 700 MB VRAM 餘量）。

---

## 6. 實作：匯出不同解析度的 ONNX + 建 engine

### 6.1 原理

解析度是在 ONNX 匯出時固定的。同一個 ONNX 不能在不同解析度下 build engine。
所以你需要：

```
匯出 672 ONNX → build 672 engine
匯出 560 ONNX → build 560 engine
```

匯出時，`export_sam3_to_onnx.py` 會自動修補 RoPE（旋轉位置編碼），
讓 global attention 層的位置資訊匹配新的 patch 數量。

### 6.2 哪些需要重新匯出？

4 個子模型中，跟解析度有關的是：

| 子模型 | 跟解析度有關？ | 原因 |
|--------|:---:|------|
| **Vision Encoder** | ✅ | RoPE 位置編碼綁定 patch 數量 |
| **Geometry Encoder** | ✅ | 輸入 FPN 特徵圖大小跟解析度有關 |
| **Text Encoder** | ❌ | 只處理文字 token，跟圖片無關 |
| **Decoder** | ✅ | FPN 輸入大小跟解析度有關 |

**Text Encoder 可以 symlink 共用，其他三個必須重新匯出。**

### 6.3 完整指令

以 672 解析度為例，在你的 PyTorch 容器（有 SAM3 model cache 的那個）中：

```bash
# === 步驟 1: 匯出 ONNX ===

# VE
docker exec <sam3 容器> python3 \
  setup/export_sam3_to_onnx.py \
  --module vision \
  --model-path facebook/sam3 \
  --image-size 672 \
  --output-dir setup/onnx_r672_q50 \
  --device cuda

# Decoder (q50)
docker exec <sam3 容器> python3 \
  setup/export_sam3_to_onnx.py \
  --module decoder \
  --model-path facebook/sam3 \
  --image-size 672 \
  --num-queries 50 \
  --output-dir setup/onnx_r672_q50 \
  --device cuda

# GE
docker exec <sam3 容器> python3 \
  setup/export_sam3_to_onnx.py \
  --module geometry \
  --model-path facebook/sam3 \
  --image-size 672 \
  --output-dir setup/onnx_r672_q50 \
  --device cuda

# TE (symlink，不需要重新匯出)
cd setup/onnx_r672_q50
ln -s ../onnx_q200/text-encoder.onnx text-encoder.onnx
```

```bash
# === 步驟 2: 建 TensorRT 引擎 ===

docker exec <tensorrt 容器> bash setup/onnx_to_tensorrt.sh \
  setup/onnx_r672_q50 \
  engines/b8_q50_r672 \
  672
```

`onnx_to_tensorrt.sh` 的第三個參數就是解析度，它會自動算出正確的 FPN 尺寸。

```bash
# === 步驟 3: 建立 config ===

# 複製一份 config，只改 engines 路徑
cp config_q50.json config_q50_r672.json
# 編輯 config_q50_r672.json，把 "engines" 改成 "engines/b8_q50_r672"
```

```bash
# === 步驟 4: 驗證 ===

docker exec <tensorrt 容器> python3 infer.py \
  --config config_q50_r672.json \
  --images Inputs/demo_3.jpg \
  --output outputs/test_672 \
  --conf 0.3
```

### 6.4 命名規則

```
ONNX 目錄:    onnx_r{解析度}_q{queries}/
              例: onnx_r672_q50/

Engine 目錄:  b{batch}_q{queries}_r{解析度}/
              例: b8_q50_r672/

Config 檔案:  config_q{queries}_r{解析度}.json
              例: config_q50_r672.json
```

### 6.5 現有引擎清單

```
engines/
├── b8_q50/          ← 1008, batch=8, queries=50（推薦預設，高品質場景）
├── b8_q200/         ← 1008, batch=8, queries=200（原始版，較大 VRAM）
├── b8_q50_r840/     ← 840（12 GB+ VRAM 場景）
├── b8_q200_r840/    ← 840, queries=200
├── b8_q50_r672/     ← 672（8 GB 部署推薦，4.0 FPS/cam @8路）
├── b8_q50_r560/     ← 560（8 GB 部署最省，4.6 FPS/cam @8路）
└── b8_q50_int8/     ← 1008, INT8 VE（實驗性，見 quantization_guide.md）
```

### 6.6 轉換時間參考（RTX 5090）

| 子模型 | 1008 | 840 | 672 | 560 |
|--------|:---:|:---:|:---:|:---:|
| Vision Encoder | ~10 min | ~8 min | ~6 min | ~5 min |
| Text Encoder | ~3 min | 同左 | 同左 | 同左 |
| Geometry Encoder | ~1 min | ~1 min | ~1 min | ~1 min |
| Decoder | ~2 min | ~2 min | ~2 min | ~2 min |
| **合計** | **~16 min** | **~14 min** | **~12 min** | **~11 min** |

> TE 跟解析度無關（只處理文字 token），可以 symlink 共用 ONNX 和 engine。

---

## 7. 常見問題

### Q: 解析度一定要是 14 的倍數嗎？

**是的。** SAM3 VE 的 patch_size = 14。圖片會被切成 `(解析度/14)²` 個 patch。
如果不是 14 的倍數，patch 邊界會對不齊，模型會報錯。

常見的有效解析度：
```
 448 = 14 × 32
 560 = 14 × 40
 672 = 14 × 48
 784 = 14 × 56
 840 = 14 × 60
 952 = 14 × 68
1008 = 14 × 72
```

### Q: 降解析度會影響座標精度嗎？

VisionDSL 用的是**歸一化座標** (0.0-1.0)，所以 bbox 的相對位置不受解析度影響。
但低解析度下，bbox 的邊緣精度會降低（因為每個 patch 代表的區域更大）。

```
1008: 每個 patch 代表 14×14 = 196 pixels → bbox 精度 ±14 pixels
 560: 每個 patch 代表 14×14 = 196 pixels → bbox 精度 ±14 pixels（一樣）
```

等等，精度不是一樣嗎？**不完全是。** 雖然 patch 大小一樣（都是 14×14），
但圖片被 resize 到不同大小後，原始圖片中每個 patch 覆蓋的區域不同：

```
原始圖片 1920×1080:
  resize 到 1008 → 每個 patch 覆蓋原圖約 27×27 pixels
  resize 到  560 → 每個 patch 覆蓋原圖約 48×48 pixels

所以 560 的 bbox 精度確實比 1008 差（大約 2 倍）
但對 VisionDSL 的 zone presence 判斷來說，通常不是問題
```

### Q: 可以在同一台機器上同時載入兩個解析度的引擎嗎？

可以，但 VRAM 會加倍。不建議。
如果需要兩階段（低解析度偵測 + 高解析度細看），建議用 crop 的方式：

```
第一步: 560 解析度跑全圖 → 找到人的位置
第二步: 把人的 bbox 區域 crop 出來 → 用 1008 解析度只跑這塊
```

但這需要修改 pipeline 架構，目前不支援。

### Q: 我能不能用更高的解析度（如 1344 = 14×96）？

技術上可以。但：
- VRAM 會暴增（FPN 大小跟 patches⁴ 成正比）
- 速度會大幅下降
- SAM3 是在 1008 解析度訓練的，超過這個解析度不保證品質更好

---

## 附錄：Amdahl's Law 與加速上限

為什麼降解析度的加速效果比 INT8 大得多？

```
INT8 只加速「計算」這一個環節:
  整體時間 = 解碼 + 前處理 + VE計算 + VE搬運 + Decoder + 後處理
  INT8 只幫到 VE計算 → 整體加速有限 (Amdahl's Law)

降解析度同時減少「計算」和「搬運」:
  VE計算量 ∝ patches² (attention)
  VE搬運量 ∝ patches² (FPN 大小)
  Decoder計算量 ∝ patches² (FPN 輸入)
  Buffer大小 ∝ patches² (FPN 複製)

  1008→672: patches 從 72→48 (↓33%)
  patches² 從 5184→2304 (↓56%)

  → 計算和搬運同時減少 56%，所以整體快了 ~78%
```

這就是為什麼**降解析度是最有效的優化手段**——它同時攻擊了 compute bound 和 memory bandwidth bound 兩個瓶頸。
