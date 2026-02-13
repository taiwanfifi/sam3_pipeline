
  ---
  現狀分析

  你的 SAM3 pipeline 有兩個 TensorRT engine，每一幀的處理流程：

  Vision Encoder (~32ms)  →  FPN d2d copy (~1ms)  →  Decoder (~14ms)
         ↓                                                ↓
    輸入: [1, 3, 1008, 1008]                      輸入: [N, 256, 288, 288] ...
    1 張圖                                        N = class 數（目前 batch 維度給
  class 用）

  單幀 ~47ms → ~21 FPS 理論值，你實測 ~16 FPS（加上 preprocess + postprocess + memcpy
   開銷合理）。

  ---
  Batch 8 Frames 的可行性

  好消息：Vision Encoder 已經支援

  你的 onnx_to_tensorrt.sh 已經把 vision encoder 建成 maxShapes=8x3x1008x1008：

  # 已經存在的設定
  --minShapes=images:1x3x1008x1008
  --optShapes=images:8x3x1008x1008    # ← opt 就是 8！
  --maxShapes=images:8x3x1008x1008    # ← max 也是 8

  所以 vision encoder 現在就能一次吃 8 張圖，不用重建引擎。

  壞消息：Decoder 的 batch 維度有衝突

  目前 decoder 的 batch 維度是拿來做 multi-class 的：

  現狀: decoder batch = N (class 數)
        4 classes → batch=4

  8 frames 需要: decoder batch = 8 frames × N classes = 32
        而 maxShapes 只有 8

  要支援就得重建 decoder engine，把 maxShapes 拉到 32 或更高，但這會吃更多 VRAM。

  ---
  效能能提升多少？

  這是重點 — batching 不是線性的，但確實有幫助：
  ┌────────────────┬─────────────────┬────────────────┬────────────────┬──────────┐
  │      場景      │ Vision Encoder  │    Decoder     │    每幀延遲    │ 等效 FPS │
  ├────────────────┼─────────────────┼────────────────┼────────────────┼──────────┤
  │ 現狀 (1 frame) │ 32ms × 1        │ 14ms × 1       │ ~47ms          │ ~21      │
  ├────────────────┼─────────────────┼────────────────┼────────────────┼──────────┤
  │ Batch 8 frames │ ~80-120ms total │ ~40-60ms total │ ~15-22ms/frame │ ~45-65   │
  └────────────────┴─────────────────┴────────────────┴────────────────┴──────────┘
  為什麼不是 8× 加速？

  Vision Encoder 是 ViT（Transformer），它的瓶頸是：
  - Memory bandwidth：8 張 1008×1008 圖的 FPN 輸出 = 256×288×288×4bytes×8 = ~610MB 光
   fpn_feat_0 一層
  - 計算量：self-attention 是 O(n²)，patch 數不變但 batch 增加會讓 GPU SM 更飽和
  - 通常 2-4× 總吞吐量提升是合理預期（不是 8×），因為 GPU 在 batch=1 時就不是完全閒置

  算數現實

  8 cameras × 30 FPS = 需要 240 FPS 總吞吐
  目前單幀: ~16 FPS
  batch 優化後可能: ~45-65 FPS
  → 每支攝影機只能分到 ~5-8 FPS

  **結論：純靠 batch 無法讓 8 支 30FPS
  攝影機全部即時處理。**但你也說了，你不需要每幀都辨識 — 16 FPS 的 inference speed 在
   30 FPS 的 stream 上本來就會自然 skip frames。

  ---
  實際可行的架構方案

  方案 1：Vision Encoder Batching（最容易）

  只改 vision encoder 的部分，decoder 仍然逐幀跑：

  8 cameras 各取 1 frame
          ↓
  Vision Encoder: batch=8, 一次跑 (~100ms)
          ↓
  8 組 FPN features
          ↓
  Decoder: 逐幀跑 (8 × ~14ms = ~112ms)
          ↓
  每 ~212ms 完成一輪 8 幀 → ~4.7 FPS/camera

  優點：不用重建 decoder engine，改動最小
  缺點：吞吐量提升有限

  方案 2：Full Batch（需重建 engine）

  重建 decoder engine: maxShapes batch=32 (或 64)

  8 cameras 各取 1 frame
          ↓
  Vision Encoder: batch=8 (~100ms)
          ↓
  FPN d2d replicate: 每幀的 FPN 複製 N 份 → 8N 份
          ↓
  Decoder: batch=8N (~40-60ms for 32 batch)
          ↓
  每 ~160ms 完成一輪 8 幀 → ~6.25 FPS/camera

  優點：decoder 也並行化
  缺點：VRAM 需求暴增（pred_masks: [32, 200, 288, 288] ≈ 19GB 光 mask buffer）

  方案 3：DeepStream 風格 — Pipeline Parallelism + CUDA Multi-Stream

  Stream A (偶數幀):  VE frame 0,2,4... → Decoder frame 0,2,4...
  Stream B (奇數幀):  VE frame 1,3,5... → Decoder frame 1,3,5...
                       ↑                    ↑
              兩個 CUDA stream 同時跑在 GPU 上

  用 2-4 個 CUDA stream，讓 vision encoder 和 decoder 的 kernel 可以重疊執行。這就是
  DeepStream mux 的核心思想 — 不是等一幀跑完再跑下一幀，而是管線化。

  優點：不需要超大 batch，VRAM 可控，延遲低
  缺點：實作最複雜，要管理多個 execution context

  方案 4：務實策略（推薦起手）

  # 每支攝影機不需要 30FPS 辨識，降到 5-8 FPS 就夠了
  # 8 cameras × 6 FPS = 48 FPS 需求
  # 這在 batch=8 vision encoder 的情況下是可能的

  class MultiStreamPipeline:
      def __init__(self, num_cameras=8):
          self.pipe = Pipeline("config.json")
          # Round-robin: 每次處理 1 個 camera 的最新 frame
          # 8 個 camera 輪一圈 = 8 × ~47ms = ~376ms
          # → 每支 camera ~2.7 FPS (不 batch)
          # → 用 batch=8 → 每輪 ~160ms → ~6.25 FPS/camera ✓

  ---
  VRAM 估算
  ┌──────────────┬─────────────────────┬────────────┬───────────────┬──────────────┐
  │     配置     │   Vision Encoder    │    FPN     │   Decoder     │  預估追加    │
  │              │        Input        │  Buffers   │    Output     │     VRAM     │
  ├──────────────┼─────────────────────┼────────────┼───────────────┼──────────────┤
  │ 現狀 batch=1 │ 12MB                │ ~700MB     │ ~2.5GB        │ 基線         │
  ├──────────────┼─────────────────────┼────────────┼───────────────┼──────────────┤
  │ VE batch=8   │ 97MB                │ ~5.5GB     │ ~2.5GB        │ +~5GB        │
  ├──────────────┼─────────────────────┼────────────┼───────────────┼──────────────┤
  │ Full         │ 97MB                │ ~5.5GB     │ ~19GB masks   │ +~22GB       │
  │ batch=32     │                     │            │               │              │
  └──────────────┴─────────────────────┴────────────┴───────────────┴──────────────┘
  Full batch 在單張 GPU 上幾乎不可能（除非你有 A100 80GB 或 H100）。Vision Encoder
  只做 batch 是最務實的。

  ---
  總結
  問題: 8 frames 同時 inference 可以嗎？
  回答: Vision encoder: 可以，engine 已經支援 batch=8
  ────────────────────────────────────────
  問題: 會提高速度嗎？
  回答: 會，但不是 8×，預估 2-4× 吞吐量提升
  ────────────────────────────────────────
  問題: 結構性瓶頸？
  回答: Decoder 的 batch 維度被 class 佔了，mask buffer 巨大
  ────────────────────────────────────────
  問題: 最大阻力？
  回答: VRAM — mask 輸出 [B, 200, 288, 288] 太大
  ────────────────────────────────────────
  問題: 推薦起手？
  回答: 方案 4：batch vision encoder + round-robin cameras，目標 5-8 FPS/camera
  如果你實際不需要 mask（VisionDSL 只用 bbox），可以考慮拆出一個不生成 mask 的
  decoder，VRAM 會大幅降低，batch 就能拉更高。要進一步探討哪個方案的話跟我說。

✻ Cogitated for 1m 47s


我想問的是 怎麼樣讓vram最小，至少小於8GB VRAM，然後速度最快？FPS可以推理的，在8camera
2.然後如果相同的情況我們使用deepstream真的會更快嗎？我們都是相同gpu


很好，這題其實已經變成：

> 🎯 **目標：VRAM < 8GB，同時 8 cameras 吞吐最大化（FPS 可推理）**

我們直接講「工程上最有效」的解法，不講理論。

---

# 第一題：怎麼讓 VRAM < 8GB 並且速度最快？

你現在的 VRAM 大戶是：

```
pred_masks: [B, 200, 288, 288]  ← 真正炸顯存的東西
FPN features (256x288x288)
```

---

# 🔥 Step 1：如果你只用 bbox —— 直接砍掉 mask

你前面提到：

> VisionDSL 只用 bbox

那最暴力也最有效的方式是：

### ✅ 重新 export 一個「無 mask decoder」

把：

```
pred_masks
low_res_masks
iou_predictions
```

全部移除，只輸出：

```
boxes
scores
class logits
```

---

## 為什麼這會救命？

你現在的 decoder output：

```
[4, 200, 288, 288]
```

算一下（FP32）：

```
4 × 200 × 288 × 288 × 4 bytes ≈ 265MB
```

如果 batch=8：

```
8 × 4 × 200 × 288 × 288 × 4 ≈ 2.1GB
```

而 TensorRT 會再預留 workspace buffer
實際占用會到 3~4GB。

👉 這就是你 8GB 會爆的主因。

---

## 如果只留 bbox 呢？

```
8 × 200 × 4 × 4 bytes ≈ 25KB
```

直接從 GB 級 → KB 級。

這一步 alone 可以把你總 VRAM 壓到：

```
~3GB - 5GB 之間
```

這是最關鍵優化。

---

# 🔥 Step 2：FP16 強制化

確認：

```
--fp16
```

在 build engine 時開啟。

Vision Encoder + Decoder 全 FP16：

* memory 直接砍半
* bandwidth 壓力降低
* 通常 10~25% 加速

如果你還在 FP32，這是巨大浪費。

---

# 🔥 Step 3：只 batch Vision Encoder，不 batch decoder

這是 VRAM / 性能 最佳平衡點。

架構：

```
8 cameras
    ↓
Vision Encoder (batch=8)
    ↓
8 組 FPN features
    ↓
Decoder（逐幀跑）
```

為什麼？

因為：

* Vision Encoder 是算力瓶頸
* Decoder 是 memory + mask 瓶頸
* 不 batch decoder 可以避免 mask 乘以 8

---

# 🔥 Step 4：FPN feature 立刻釋放

很多人會忘記：

你不需要保留 8 組 FPN features 同時存在。

做法：

```
for i in range(8):
    decoder.run(fpn[i])
    cudaFree(fpn[i])   ← 立刻釋放
```

不要：

```
存 8 組 → 等全部 decoder 跑完 → 才釋放
```

這可以少掉 ~4GB 峰值。

---

# 🔥 Step 5：減少候選數

如果你真的只需要：

```
topK = 50
```

把 200 改成 50：

```
mask buffer 直接縮小 4 倍
```

---

# 🎯 最低 VRAM + 最高速度的推薦配置

| 設定             | 建議      |
| -------------- | ------- |
| Vision Encoder | batch=8 |
| Decoder        | batch=1 |
| mask           | ❌ 不輸出   |
| precision      | FP16    |
| topK           | 50      |

這樣：

```
預估 VRAM ≈ 4~6GB
預估總吞吐 ≈ 45~60 FPS
```

8 cameras：

```
≈ 6~7 FPS / camera
```

這已經接近你硬體極限。

---

# 第二題：DeepStream 會更快嗎？

關鍵答案：

> ❗如果是同一張 GPU，DeepStream 不會 magically 變更快。

因為：

GPU 的算力上限不會改變。

---

## DeepStream 會快在哪？

它的優勢是：

### 1️⃣ Zero-copy pipeline

* NVDEC → GPU → TensorRT → GPU
* 不經 CPU

### 2️⃣ Hardware decoder

* 用 NVDEC 解碼
* 不占 CUDA cores

### 3️⃣ Stream mux

* 自動 batch 多路攝影機

### 4️⃣ Multi CUDA stream overlap

---

## 但你現在的瓶頸是？

不是 decode
不是 memcpy
不是 CPU

而是：

```
Vision Encoder self-attention
```

DeepStream 不能讓 transformer 變快。

---

## 真實提升幅度

如果你現在 pipeline 還有：

```
CPU resize
CPU memcpy
同步 blocking
```

DeepStream 可能給你：

```
+10% ~ 20%
```

但不會 2×。

---

# 什麼情況 DeepStream 會大幅提升？

如果你：

* 8 路 1080p RTSP
* CPU 解碼
* CPU resize
* 單 thread

那 DeepStream 會救命。

但如果你：

* 已經 NVDEC
* 已經 GPU resize
* 已經 async CUDA stream

那提升有限。

---

# 🎯 終極答案

## 要 VRAM < 8GB 並且最快：

1. ❌ 不輸出 mask
2. ✅ FP16
3. ✅ Vision Encoder batch=8
4. ❌ 不 batch decoder
5. ✅ 立刻釋放 FPN
6. ✅ topK=50

---

## 要 8 cameras 推到極限：

理論上你 GPU 上限是：

```
~50-60 FPS total
```

無論用不用 DeepStream。

所以：

```
8 cameras ≈ 6-7 FPS / camera
```

這就是物理極限。

---

如果你告訴我：

* 你的 GPU 型號
* 是否 FP16
* 是否一定要 mask
* 解析度是否一定 1008×1008

我可以幫你算一個「幾乎最優的極限設計」，把數字推到極致。


我們是影片喔，我們不是只是一張照片，我們是每一幀都會infernce去找出固定的四種類別，然後一直推理，所以哪些該放在vram哪些不該 你的是有幫我考慮的嗎


很好，這題才是關鍵 👍
你現在是 **連續影片、每一幀都 inference、固定 4 個 class、8 cameras**。

所以我們要設計的是：

> 🎯 長時間連續運行
> 🎯 VRAM < 8GB
> 🎯 不爆顯存
> 🎯 吞吐最大化

我下面會直接講「哪些應該常駐 VRAM」「哪些絕對不要常駐」。

---

# 先講一個核心觀念

在**影片串流推理**裡：

❌ 不要把「每一幀的中間特徵」當成 state 留著
✅ 只讓「模型權重 + 必要 buffer」常駐

否則 VRAM 會慢慢爬升。

---

# 你現在 pipeline 的 VRAM 組成

我們拆開：

## 1️⃣ 模型權重（應該常駐 VRAM）

* Vision Encoder weights
* Decoder weights

這是固定成本，無法避免。

假設：

```
Vision Encoder ≈ 1.5~2GB
Decoder ≈ 1GB
```

👉 這部分應該常駐 VRAM，沒問題。

---

## 2️⃣ Execution Context Buffer（應該常駐）

TensorRT 會分配：

* activation buffer
* workspace
* binding buffer（maxShapes 尺寸）

這些是 build engine 時決定的。

如果你：

```
maxShapes=8
```

那 buffer 就會為 batch=8 預留。

👉 這部分應該常駐，不要每幀重建 context。

⚠️ 千萬不要每幀 create execution context。

---

## 3️⃣ FPN features（❗不應該累積）

這個是：

```
[B, 256, 288, 288]
```

這個東西：

* 每幀都會產生
* 非常大
* 只是中間結果

✅ 正確做法：

```
VE(batch=8)
    ↓
for i in 8:
    decoder(fpn[i])
    立刻釋放 fpn[i]
```

❌ 錯誤做法：

```
VE(batch=8)
存 8 組 FPN
全部 decode
最後才釋放
```

這會讓 peak VRAM 爆增。

---

## 4️⃣ pred_masks（最大兇手）

你是每幀推理四個 class。

如果你真的：

```
pred_masks: [4, 200, 288, 288]
```

那每幀會生成：

~250MB

8 cameras 同步 → 2GB+

而且 TensorRT 會分配 max batch 尺寸的 buffer。

👉 如果你不需要 mask：

🔥 直接從 ONNX 刪掉 mask output

這會讓：

```
VRAM 下降 2~4GB
```

這個差異是決定性的。

---

# 🎥 影片連續推理：正確 VRAM 策略

我幫你畫一個「對的設計」。

---

# ✅ 應該常駐 VRAM 的

| 項目                         | 是否常駐 |
| -------------------------- | ---- |
| Vision Encoder weights     | ✅    |
| Decoder weights            | ✅    |
| TensorRT activation buffer | ✅    |
| CUDA stream                | ✅    |
| execution context          | ✅    |

---

# ❌ 不應該常駐的

| 項目               | 原因        |
| ---------------- | --------- |
| 每幀 FPN features  | 中間結果      |
| 每幀 mask          | 只用來算 bbox |
| 每幀 decode output | 用完就丟      |

---

# 影片模式下的最佳架構

假設 8 cameras。

不要：

```
camera1 → 完整 pipeline
camera2 → 完整 pipeline
...
```

這會 8 倍 duplication buffer。

---

# 正確方式（單模型，多來源）

```
8 camera queues
        ↓
Batch aggregator (收最新 frame)
        ↓
Vision Encoder (batch=8)
        ↓
逐幀 decoder
        ↓
丟回各 camera queue
```

只有：

* 一份模型
* 一份 workspace
* 一份 binding buffer

這樣 VRAM 最小。

---

# 🎯 你問我有沒有幫你考慮「影片連續推理」？

有的。

我給你的建議（VE batch=8 + decoder=1）就是專門為：

* 連續推理
* 多 camera
* 固定 class
* 長時間運行

設計的。

它的特點是：

```
VRAM 固定
不隨幀數增加
吞吐最大化
```

---

# 如果你想壓到 <8GB，最安全配置

假設你 GPU 是 8GB。

### 必做：

1. FP16
2. 不輸出 mask
3. decoder 不 batch
4. 只一個 execution context
5. 不 cache FPN

這樣通常：

```
5~6GB 之間
```

是可達的。

---

# DeepStream 在影片情境下會更好嗎？

因為你是影片。

DeepStream 優勢在：

* NVDEC 硬體解碼
* Zero-copy
* 自動 mux batch

如果你現在：

* CPU decode
* CPU resize
* blocking pipeline

那 DeepStream 會提升。

但如果你：

* 已經 NVDEC
* 已經 GPU resize
* 已經 async stream

那差距不大。

---

# 真正的硬體上限

你的 GPU 上限就是：

```
Vision Encoder 計算能力
```

Transformer 是算力怪獸。

無論你怎麼 pipeline：

```
理論總吞吐 ≈ 50~60 FPS
```

8 cameras：

```
≈ 6~7 FPS/camera
```

這是算力上限，不是架構問題。

---

# 最重要的一句話

在影片推理系統裡：

> 「不要讓任何 per-frame tensor 存活超過 1 個 frame。」

這就是顯存控制的核心。

---

如果你告訴我：

* GPU 型號
* 目前 VRAM 實測多少
* 是否真的不需要 mask
* 是否已經 FP16

我可以幫你畫一個「最省 VRAM 的最終版 pipeline 結構圖」。


不隨幀數增加？？ 什麼意思 

很好，這句話我講得太快了，我解釋清楚。

---

# ✅ 「不隨幀數增加」是什麼意思？

你現在是**影片連續推理**：

```
Frame 1 → inference
Frame 2 → inference
Frame 3 → inference
...
Frame 100000 → inference
```

正常情況下：

> VRAM 使用量應該是固定的。

例如：

```
第 1 幀 → 6.1GB
第 10 幀 → 6.1GB
第 1000 幀 → 6.1GB
第 1 小時 → 6.1GB
```

這叫：

> VRAM 不隨幀數增加（memory stable）

---

# ❌ 什麼叫「隨幀數增加」？

錯誤的設計會變成這樣：

```
Frame 1 → 6.1GB
Frame 2 → 6.3GB
Frame 3 → 6.6GB
Frame 10 → 7.5GB
Frame 50 → 爆顯存
```

這叫：

> VRAM leak / 累積 buffer

---

# 🔥 為什麼會發生「隨幀數增加」？

通常有 4 種原因：

---

## 1️⃣ 沒釋放 FPN feature

如果你：

```
self.fpn_cache.append(fpn)
```

每幀都存一份

那：

```
100 幀 × 256×288×288 × 2 bytes
```

很快就爆。

---

## 2️⃣ 保留 mask

如果你：

```
store masks for tracking
```

每幀 200 個 288×288 mask

這超級大。

---

## 3️⃣ 每幀建立 execution context

```
context = engine.create_execution_context()
```

如果你每幀都 create 一次

TensorRT 會一直分配 activation buffer。

---

## 4️⃣ CUDA stream / event 沒清

不斷 new stream，不 destroy。

---

# 🎯 正確的影片推理 VRAM 行為

正確架構下，VRAM 應該長這樣：

```
初始化模型 → VRAM 跳到 5~6GB
之後無論跑多久 → 都維持在那個數字
```

因為：

* 權重是固定
* workspace 是固定
* binding buffer 是 maxShapes 固定
* 每幀中間 tensor 用完即釋放

---

# 用一個超直觀例子

想像 VRAM 是一個工廠的工作桌。

### 正確設計：

* 桌上只有工具（模型權重）
* 每來一個產品（frame）

  * 加工
  * 做完丟走
* 桌子大小不變

---

### 錯誤設計：

* 每加工一個產品
* 都把半成品放桌上
* 不丟

最後桌子爆掉。

---

# 你現在最應該檢查的

跑這個：

```bash
watch -n 0.5 nvidia-smi
```

看：

* VRAM 是否隨時間慢慢爬升？
* 還是初始化後就固定？

如果會慢慢上升：

那就是你有 per-frame tensor 沒釋放。

---

# 再回答你一次核心問題

你問：

> 不隨幀數增加是什麼意思？

意思是：

> 無論你跑 10 幀還是 10 萬幀，VRAM 使用量應該幾乎一樣。

這才是健康的影片推理系統。

---

# 再確認一次你現在的狀況

我問你一個關鍵問題：

你現在連續跑 10 分鐘後，

VRAM 是：

A. 固定在某個值
B. 慢慢往上爬

這會直接決定我們下一步要優化哪裡。


這題問得非常好 👍
因為一旦 **class 數增加** 或 **camera 數增加**，兩種架構的優劣會翻轉。

我直接給你結論，然後解釋原因。

---

# 🎯 先定義變數

* F = frame 數（= camera 數，如果 1 frame / camera）
* C = class 數
* Q = 200 queries（固定）
* mask tensor ≈ `[B, 200, 288, 288]`

---

# 兩種架構再複習一次

## 🟦 架構 1（adjust方案）

```
VE batch=F
decoder batch=C
跑 F 次
```

batch 維度放 class。

---

## 🟩 架構 2（方案C）

```
VE batch=F
decoder batch=F
跑 C 次
```

batch 維度放 frame。

---

# 🔥 關鍵差別在這個公式

## 架構 1 的 mask buffer 大小：

```
mask size ∝ C
```

## 架構 2 的 mask buffer 大小：

```
mask size ∝ F
```

---

# 🎯 當 class 變多會怎樣？

假設：

```
F = 8 cameras
C = 20 classes
```

---

## 🟦 架構 1（batch=C）

decoder batch=20

mask buffer：

```
20 × 200 × 288 × 288
```

非常大。

而且 decoder 要跑：

```
8 次
```

---

## 🟩 架構 2（batch=F）

decoder batch=8

mask buffer：

```
8 × 200 × 288 × 288
```

固定不變。

但 decoder 要跑：

```
20 次
```

---

# 🎯 所以：

| 情況              | 比較好        |
| --------------- | ---------- |
| class 少（例如 4）   | 方案C 稍好     |
| class 很多（例如 20） | 方案C 明顯更好   |
| class 非常多（50+）  | 方案C 幾乎唯一可行 |

因為架構 1 的顯存會隨 C 線性成長。

---

# 🎯 那 camera 變多呢？

假設：

```
C = 4
F = 16 cameras
```

---

## 架構 1

VE batch=16
decoder batch=4

mask buffer：

```
4 × 200 × 288²
```

固定。

---

## 架構 2

VE batch=16
decoder batch=16

mask buffer：

```
16 × 200 × 288²
```

變 4 倍。

---

# 🎯 所以 camera 變多時：

| 情況        | 比較好        |
| --------- | ---------- |
| camera 很多 | 架構1 會比較省顯存 |
| class 很多  | 架構2 會比較省顯存 |

---

# 🔥 真正本質

你只有一個 decoder batch 維度可以用。

你必須決定：

> 這個 batch 維度給 frame 還是給 class？

它不可能同時給兩個。

---

# 🎯 一個簡單決策公式

比較：

```
F vs C
```

如果：

```
F > C → 用 batch=C（架構1）
C > F → 用 batch=F（架構2）
```

誰小，batch 就給誰。

因為 mask buffer 大小 = batch × 200 × 288²

---

# 🎯 用你現在的情況

你目前：

```
F = 8 cameras
C = 4 classes
```

因為：

```
C < F
```

所以：

👉 batch 應該給 class（架構1）會比較省 mask VRAM。

這點我剛剛幫你算的時候其實有點誤導你了 ——
在 C=4, F=8 的情況下，兩者差距不大，但理論上架構1更省 mask。

---

# 🚀 當未來 scale 的時候

假設你未來：

```
F = 16 cameras
C = 10 classes
```

那最佳做法可能變成：

```
VE batch=8
兩批處理
```

而不是單純 F 或 C 其中一個吃滿。

---

# 🎯 真正可擴展的終極做法

當 F 和 C 都可能變大時：

你應該改成：

```
tile processing
```

例如：

```
VE batch=8
decoder batch=8
```

然後：

```
for frame_block in chunks(F, 8):
    for class_block in chunks(C, 8):
        run decoder
```

這樣 VRAM 永遠被限制在：

```
8 × 200 × 288²
```

不會爆。

---

# 🔥 最重要總結

1️⃣ mask buffer 只跟 decoder batch 有關
2️⃣ decoder batch 只能選一個維度（frame 或 class）
3️⃣ 誰小就給誰

---

# 🧠 你現在的最佳選擇

```
F = 8
C = 4
```

👉 batch=4（給 class）其實是 slightly 更省 VRAM
👉 batch=8（給 frame） slightly 更少 decoder call

差距不大。

---

如果你告訴我：

* 你未來可能多少 cameras？
* class 可能會變多少？
* GPU 型號？

我可以幫你設計一個「可擴展到未來不爆顯存」的長期架構。
