# INT8 量化完全指南

這份指南教你理解 TensorRT 的精度選擇（FP32 / FP16 / INT8），
以及如何為 SAM3 Vision Encoder 建立 INT8 引擎。

> **結論先講：** 在 RTX 5090 上，INT8 VE 沒有比 FP16 快（甚至略慢）。
> 但在某些舊卡或特殊場景下可能有用。這份指南的重點是**讓你理解原理**，
> 讓你自己判斷什麼時候該用、什麼時候不該用。

---

## 目錄

1. [數字精度是什麼？](#1-數字精度是什麼)
2. [為什麼 FP16 不需要 calibration，但 INT8 需要？](#2-為什麼-fp16-不需要-calibration但-int8-需要)
3. [Calibration 的原理](#3-calibration-的原理)
4. [實作：建立 INT8 Vision Encoder](#4-實作建立-int8-vision-encoder)
5. [實測結果與分析](#5-實測結果與分析)
6. [什麼時候 INT8 有用？什麼時候沒用？](#6-什麼時候-int8-有用什麼時候沒用)
7. [常見問題](#7-常見問題)

---

## 1. 數字精度是什麼？

電腦儲存數字的方式有很多種。精度越高，能表達的數值範圍和細節越多，但佔的空間也越大。

### 比喻

想像你在量身高：

| 精度 | 量法 | 例子 | bits |
|------|------|------|------|
| **FP32** | 用毫米尺 | 175.324 cm | 32 bits |
| **FP16** | 用公分尺 | 175.3 cm | 16 bits |
| **INT8** | 用手掌量 | "大約 11 個手掌高" | 8 bits |

- **FP32**（32-bit 浮點數）：PyTorch 訓練時的預設精度。最精確，但最慢、最佔空間。
- **FP16**（16-bit 浮點數）：精度夠用，速度快一倍。這是我們目前用的。
- **INT8**（8-bit 整數）：只有 256 個可能的值（-128 到 127）。理論上再快一倍，但需要知道怎麼「對應」。

### GPU 硬體支援

NVIDIA GPU 有專門的硬體單元（Tensor Core）來加速不同精度的運算：

```
RTX 3090 (Ampere):   FP16 = 142 TFLOPS,  INT8 = 284 TOPS  → INT8 快 2x
RTX 4090 (Ada):      FP16 = 330 TFLOPS,  INT8 = 660 TOPS  → INT8 快 2x
RTX 5090 (Blackwell): FP16 = 419 TFLOPS,  INT8 = 838 TOPS  → INT8 快 2x（理論值）
```

看起來 INT8 永遠快 2 倍？**不是。** 因為速度不只取決於算力。

---

## 2. 為什麼 FP16 不需要 calibration，但 INT8 需要？

### FP16：直接轉就好

FP32 → FP16 是「無損」的（對深度學習來說）：

```
FP32: 0.12345678      → FP16: 0.1235      （丟掉後面幾位小數）
FP32: -3.14159265     → FP16: -3.141      （一樣可以表達）
FP32: 1e-8 (0.00000001) → FP16: 1e-8       （小數字也能表達）
```

因為 FP16 是浮點數，它有「指數部分」可以自動調整範圍。就像科學記號一樣：
`3.14 × 10⁵` 和 `3.14 × 10⁻⁵` 都能用，只是精度差一點。

所以 `trtexec --fp16` 直接轉就行，TensorRT 會自動把每個 FP32 權重截斷成 FP16。

### INT8：必須知道範圍

INT8 是整數，只有 -128 到 127 這 256 個值。它**沒有指數部分**。

問題來了：如果某一層的 activation 值範圍是 [-3.5, 7.2]，你要怎麼用 -128~127 來表達？

```
方案 1: 直接四捨五入
  -3.5 → -4    ✓
  0.1  → 0     ✗ （精度全失！0.1 和 0.2 都變成 0）
  7.2  → 7     ✓

方案 2: 先縮放 (scale)，再四捨五入
  scale = 127 / 7.2 = 17.64
  -3.5 × 17.64 = -61.7 → -62  ✓
  0.1  × 17.64 = 1.76  → 2    ✓（精度 OK！）
  7.2  × 17.64 = 127.0 → 127  ✓
```

方案 2 就是 INT8 量化的核心——**你需要知道每一層的數值範圍，才能算出 scale factor。**

但問題是：每一層的 activation 範圍取決於**輸入資料**。不同的圖片會產生不同的 activation 值。

所以你需要用**真實資料**跑一遍模型，記錄每一層的 min/max，才能算出 scale。

**這個過程就叫 calibration（校準）。**

---

## 3. Calibration 的原理

### 流程

```
[1] 準備 100~500 張代表性圖片
         ↓
[2] 一張一張餵進模型，記錄每一層的 activation 值
         ↓
[3] 對每一層的值做統計分析（找出最佳的 scale）
         ↓
[4] 產出 calibration cache（每層一個 scale factor）
         ↓
[5] 用這些 scale 建立 INT8 引擎
```

### Calibration 演算法

TensorRT 提供幾種 calibrator：

| Calibrator | 方法 | 優缺點 |
|------------|------|--------|
| **MinMax** | 直接取 min/max | 簡單但容易被極端值帶偏 |
| **Entropy** | 最小化 FP32 和 INT8 的分佈差異（KL散度） | 最常用，品質最好 |
| **Percentile** | 取 99.99% 分位數 | 介於兩者之間 |

我們用的是 `IInt8EntropyCalibrator2`（Entropy 的改良版）。

用白話解釋 Entropy calibration：

```
假設某層的 FP32 activation 值分佈長這樣：

  出現次數
    │    ▓▓
    │   ▓▓▓▓
    │  ▓▓▓▓▓▓▓
    │ ▓▓▓▓▓▓▓▓▓▓
    └────────────────→ 值
   -10  -5   0   5  10  15

  大部分值集中在 -5 到 5 之間，但偶爾有到 15 的。

  MinMax 方案: scale = 127 / 15 = 8.47
    → -5 到 5 只用了 -42 到 42（浪費了一半的 INT8 精度）

  Entropy 方案: scale = 127 / 6.5 = 19.54
    → -5 到 5 用了 -98 到 98（精度最佳）
    → 超過 6.5 的值被截斷（clamp），但這些值很少，影響不大
```

Entropy 的核心思想：**寧可犧牲極端值的精度，也要保住常見值的精度。**

### Calibration 資料的選擇

關鍵原則：**calibration 資料要跟實際使用場景越接近越好。**

```
✅ 好的 calibration 資料:
  - 從你的監控影片中均勻取樣 100 張 frame
  - 包含不同光照（白天 / 晚上）
  - 包含不同人數（0 人 / 1 人 / 多人）

❌ 不好的 calibration 資料:
  - 用 ImageNet 的圖片（跟你的場景無關）
  - 只用同一個時間點的 frame（光照單一）
  - 全部空場景（模型沒看到人，activation 分佈偏移）
```

---

## 4. 實作：建立 INT8 Vision Encoder

### 前置條件

- TensorRT 容器已建立（見 README.md）
- ONNX 檔案已匯出
- 有一段代表性的影片（用來抽 calibration frames）

### 4.1 Calibration script

已寫好在 `setup/build_int8_ve.py`。核心結構：

```python
class VECalibrator(trt.IInt8EntropyCalibrator2):
    """繼承 TensorRT 的 calibrator 介面"""

    def get_batch_size(self):
        return 1  # 一次餵一張圖

    def get_batch(self, names):
        """TensorRT 每次呼叫這個函式取一張 calibration 圖片"""
        frame = self.frames[self.current_idx]
        tensor = self._preprocess(frame)      # 跟 infer.py 一模一樣的前處理
        cuda.memcpy_htod(self.d_input, tensor) # 上傳到 GPU
        return [int(self.d_input)]             # 回傳 GPU 記憶體位址

    def read_calibration_cache(self):
        """如果有 cache 就直接用，跳過 calibration"""
        return self._cache

    def write_calibration_cache(self, cache):
        """calibration 完成後存檔，下次不用重跑"""
        with open(self.cache_path, "wb") as f:
            f.write(cache)
```

**重點**：`_preprocess()` 必須跟 `infer.py` 的前處理完全一致（resize → normalize → CHW）。
如果不一致，calibration 記錄的 activation 範圍就會跟實際推論不同，精度會崩掉。

### 4.2 指令：第一次 build（含 calibration）

```bash
docker exec <你的 TensorRT 容器> python3 \
  /root/VisionDSL/models/sam3_pipeline/setup/build_int8_ve.py \
  --onnx /root/VisionDSL/models/sam3_pipeline/setup/onnx_q200/vision-encoder.onnx \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/shop.mp4 \
  --output /root/VisionDSL/models/sam3_pipeline/engines/b8_q50_int8/vision-encoder.engine \
  --num-frames 100 \
  --image-size 1008
```

**參數說明：**

| 參數 | 意義 | 怎麼選 |
|------|------|--------|
| `--onnx` | VE 的 ONNX 檔案 | 必須跟目標解析度匹配 |
| `--video` | calibration 影片 | 用你實際部署場景的影片 |
| `--num-frames` | 抽幾張 frame | 100 張通常夠，越多越準但越慢 |
| `--image-size` | 輸入解析度 | 跟 ONNX 匯出時一致 |
| `--output` | 輸出 engine 路徑 | 自訂 |

**過程中會看到：**

```
[Calibrator] Extracted 100 calibration frames    ← 抽 frame
Parsing ONNX: ...                                ← 解析模型結構
Building INT8 engine (this takes 10-30 min)...
  Starting Calibration.                          ← 開始校準
  Calibrated batch 0 in 29.0 seconds.            ← 每張 ~30 秒
  Calibrated batch 1 in 28.5 seconds.
  ...
  Calibrated batch 99 in 32.1 seconds.
  [Calibrator] Cache saved: ...                  ← 存 calibration cache
  Engine generation completed in 178.2 seconds.  ← 建引擎
Engine saved: ... (883 MB)                       ← 完成
```

**總時間：** ~60 分鐘（calibration ~50 分鐘 + engine build ~10 分鐘）

### 4.3 指令：第二次 build（用 cache，跳過 calibration）

calibration 完成後會產生 `.cache` 檔。下次 build 只需幾分鐘：

```bash
docker exec <容器> python3 \
  /root/VisionDSL/models/sam3_pipeline/setup/build_int8_ve.py \
  --onnx /root/VisionDSL/models/sam3_pipeline/setup/onnx_q200/vision-encoder.onnx \
  --output /root/VisionDSL/models/sam3_pipeline/engines/b8_q50_int8/vision-encoder.engine \
  --cache /root/VisionDSL/models/sam3_pipeline/engines/b8_q50_int8/vision-encoder_calib.cache \
  --image-size 1008
```

### 4.4 設定 config

INT8 VE + FP16 其他引擎：

```bash
# 建立引擎目錄，VE 用 INT8，其他 symlink 到 FP16
mkdir -p engines/b8_q50_int8
cp <int8 build 的 VE> engines/b8_q50_int8/vision-encoder.engine
cd engines/b8_q50_int8
ln -s ../b8_q50/decoder.engine decoder.engine
ln -s ../b8_q50/text-encoder.engine text-encoder.engine
ln -s ../b8_q50/geometry-encoder.engine geometry-encoder.engine
```

然後在 `config_q50_int8.json` 裡指定：
```json
{ "engines": "engines/b8_q50_int8", ... }
```

---

## 5. 實測結果與分析

### RTX 5090 實測（2026-02-15, shop.mp4, 4 classes, batch=8 single camera）

| 指標 | FP16 | INT8 VE | 差異 |
|------|:----:|:-------:|:----:|
| **avg_ms** | 62.5 | 64.6 | +3.4% 更慢 |
| **VRAM** | 7,064 MB | 6,896 MB | -168 MB (-2.4%) |
| **Engine 大小** | 885 MB | 883 MB | -2 MB |
| **偵測數差異** | baseline | < 0.2% | 精度幾乎無損 |

### 為什麼沒變快？三個原因

**原因 1: VE 是 memory bandwidth bound，不是 compute bound**

```
Vision Encoder 做什麼：
  32 層 Transformer → 每層都有 Self-Attention

Self-Attention 的計算：
  Q × K^T → [batch, heads, patches², patches²]
  然後乘以 V

在 1008 解析度：patches = 72, patches² = 5184
Attention 矩陣大小 = 5184 × 5184 = 26,873,856 個數字

GPU 需要做的事：
  1. 從 VRAM 讀取 Q, K, V 矩陣     ← 搬資料（memory bandwidth）
  2. 做矩陣乘法                     ← 算數學（compute）
  3. 把結果寫回 VRAM                 ← 搬資料（memory bandwidth）

在 RTX 5090 上：
  - 記憶體頻寬: 1792 GB/s
  - INT8 算力: 838 TOPS
  - FP16 算力: 419 TFLOPS

步驟 2 確實快了一倍（INT8 vs FP16）
但步驟 1 和 3 的速度完全一樣 — 記憶體頻寬不變

如果步驟 1+3 佔了 70% 的時間，步驟 2 佔 30%：
  INT8 理論加速 = 1 / (0.7 + 0.3/2) = 1 / 0.85 = 17%
  實際上因為精度轉換開銷，可能更低
```

**原因 2: 精度轉換的 overhead**

build 時看到一些警告：
```
[TRT] [W] Missing scale and zero-point for tensor layers.14.layer_norm1...
         expect fall back to non-int8 implementation
```

但實際分析 calibration cache（`vision-encoder_calib.cache`），**校準本身是成功的**：

| 指標 | 數值 |
|------|------|
| 總 tensor 數 | 3,675 |
| 成功取得 scale factor | **3,669 (99.8%)** |
| 失敗（7f800000 = ∞） | **6 (0.2%)** |

失敗的 6 個都是預期內的：`images`（輸入 tensor）、`Cast_output`（型別轉換）、`FPN ConvTranspose`（FPN 最終輸出層）。

**問題不在校準品質，而是 ViT 架構本身**：Softmax、LayerNorm 等操作必須在 FP16/FP32 執行，
TensorRT 會在 INT8 和 FP16 層之間自動插入 Quantize/Dequantize (Q/DQ) 節點：

```
FP16 input → [Quantize] → INT8 MatMul → [Dequantize] → FP16 Softmax →
[Quantize] → INT8 MatMul → [Dequantize] → FP16 LayerNorm → ...
```

每個 Q/DQ 轉換都有 overhead，在 32 層 Transformer 中反覆切換，抵消了 INT8 MatMul 的加速。

**原因 3: RTX 5090 的 FP16 已經極快**

Blackwell 架構的 FP16 Tensor Core 效率很高。在舊卡（例如 RTX 3090 Ampere）上，
FP16 的效率沒那麼高，INT8 的加速幅度可能更明顯。

---

## 6. 什麼時候 INT8 有用？什麼時候沒用？

### 判斷框架

```
你的模型是 compute bound 還是 memory bandwidth bound？

  Compute bound（適合 INT8）:
    - 模型很小，計算密集
    - 例如：ResNet-50, YOLO
    - GPU 大部分時間在「算」，不是在「搬」

  Memory bandwidth bound（INT8 幫助有限）:
    - 模型很大，attention 矩陣巨大
    - 例如：ViT-L (SAM3), GPT
    - GPU 大部分時間在「搬」資料
```

### 速查表

| 場景 | INT8 有用嗎？ | 原因 |
|------|:---:|------|
| SAM3 VE on RTX 5090 | ❌ | Memory bound + FP16 已極快 |
| SAM3 VE on RTX 3060 | ⚠️ | 可能有 10-20% 加速（FP16 效率較低） |
| YOLOv8 on 任何 GPU | ✅ | 小模型，compute bound |
| ResNet-50 on 任何 GPU | ✅ | 小模型，compute bound |
| LLM (GPT) on 任何 GPU | ❌ | Memory bound（跟 VE 同理）|

### 什麼時候值得嘗試？

1. **VRAM 極度緊張**：INT8 權重只佔 FP16 的一半空間
2. **部署在舊卡**：RTX 3060/3070 上 INT8 可能有明顯加速
3. **模型很小**：YOLO、ResNet 等，compute bound 的模型
4. **可以接受精度損失**：某些應用不需要 100% 精度

### 什麼時候不要用？

1. **模型是 memory bound**：像 SAM3 VE 這種大 ViT
2. **GPU 的 FP16 已經很快**：RTX 4090 / 5090
3. **精度至關重要**：醫療影像、自動駕駛
4. **沒有代表性的 calibration 資料**

### 跟降解析度比較

在 8 GB VRAM + 8 路攝影機的部署場景中，INT8 和降解析度的效果差異很大：

| 方案 | VRAM 節省 | 速度提升 | 偵測品質 | 工程量 |
|------|:---:|:---:|:---:|:---:|
| **INT8 量化** | -168 MB (-2.4%) | 無（反而慢 3.4%） | 無損 | 中（寫 calibrator） |
| **降解析度 1008→672** | -2,482 MB (-35%) | +78% | 大物件無損 | 低（重匯 ONNX） |
| **降解析度 1008→560** | -3,190 MB (-45%) | +120% | 大物件無損 | 低 |

**結論：降解析度是最有效的優化手段**。它同時攻擊 compute bound 和 memory bandwidth bound
兩個瓶頸（Amdahl's Law）。INT8 只攻擊 compute，而 SAM3 VE 的瓶頸在 memory。

詳見 [`resolution_guide.md`](resolution_guide.md)。

---

## 7. 常見問題

### Q: trtexec 可以直接 --int8 嗎？不用寫 Python script？

可以，但你需要提供 calibration 資料。`trtexec` 有 `--calib` 參數可以讀 calibration cache。
但要產生 cache，你還是需要寫 calibrator（或用現成的工具如 TensorRT 的 `polygraphy`）。

### Q: calibration cache 可以跨 GPU 使用嗎？

可以。cache 記錄的是每層的 scale factor，跟 GPU 無關。
但最終的 engine 仍然綁定 GPU 架構（要在目標 GPU 上重建 engine）。

### Q: 為什麼 engine 大小沒變（883 vs 885 MB）？

因為大量層 fallback 到 FP16，權重仍以 FP16 儲存。
如果 100% 的層都成功量化，engine 應該會縮小到約 ~450 MB。

### Q: 可以只對某些層做 INT8 嗎？

可以。TensorRT 支援 per-layer 精度控制（layer-wise quantization）。
但需要用 Python builder API 手動設定，比較複雜。

### Q: FP8 呢？比 INT8 好嗎？

FP8 是 Hopper/Blackwell 架構的新功能（8-bit 浮點數）。
它結合了 INT8 的速度和 FP16 的靈活性（有指數部分）。
TensorRT 10.x 有支援，但需要模型本身配合。未來可能是更好的選擇。

---

## 附錄：build_int8_ve.py 的完整參數

```bash
python3 setup/build_int8_ve.py \
  --onnx <VE ONNX 路徑>        # 必須
  --output <輸出 engine 路徑>   # 必須
  --video <calibration 影片>    # 第一次必須（沒有 cache 時）
  --num-frames <抽幾張>         # 預設 100
  --image-size <解析度>         # 預設 1008
  --cache <calibration cache>  # 有的話就跳過 calibration
```
