# 環境建置與引擎轉換指南

這份指南教你從零開始建立 TensorRT 推論環境，並將 SAM3 模型轉換為 TensorRT 引擎。

> **如果你的 GPU 跟我們一樣是 RTX 5090**，`engines/` 裡的引擎可以直接用 — 跳到[第三階段](#第三階段建立-tensorrt-容器)裝環境就好。
>
> **如果你的 GPU 不同**，必須走完全部階段重新轉換引擎。

---

## 總覽

```
              第一階段                    第二階段                    第三階段
          ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
          │  安裝 Docker   │          │  匯出 ONNX    │          │ 建立 TensorRT │
          │  + GPU 工具包  │    →     │  (PyTorch →   │    →     │   容器 + 轉換   │
          │              │          │    ONNX)      │          │   引擎        │
          └──────────────┘          └──────────────┘          └──────────────┘
            (主機端，一次)             (任意 GPU，一次)            (你的 GPU，一次)
```

**整個流程只需做一次。** 之後就可以直接跑推論了。

---

## 這個資料夾裡有什麼

```
setup/
├── README.md                  ← 你正在看的這份指南
├── quantization_guide.md      ← INT8 量化教學（原理 + 實驗結果 + 指令）
├── resolution_guide.md        ← 解析度調整教學（VRAM 計算 + benchmark + 決策框架）
├── export_sam3_to_onnx.py     ← PyTorch → ONNX 匯出腳本
├── onnx_to_tensorrt.sh        ← ONNX → TensorRT 引擎轉換腳本
├── build_int8_ve.py           ← INT8 VE 校準 + 建置腳本
├── onnx_q50/                  ← Q50 decoder ONNX（--num-queries 50 匯出）
├── onnx_r672_q50/             ← 672 解析度 ONNX（VE + decoder + GE，TE symlink）
└── onnx_r560_q50/             ← 560 解析度 ONNX
```

轉換完成後，引擎會儲存到 `engines/` 的子資料夾。命名規則：`b{batch}_q{queries}[_r{resolution}][_int8]`。

```
engines/
├── b8_q200/                   ← 1008, batch=8, queries=200（原始版）
├── b8_q50/                    ← 1008, batch=8, queries=50（推薦預設）
├── b8_q50_r840/               ← 840, Q50
├── b8_q200_r840/              ← 840, Q200
├── b8_q50_r672/               ← 672, Q50（8 GB 部署推薦）
├── b8_q50_r560/               ← 560, Q50（8 GB 部署最省 VRAM）
├── b8_q50_int8/               ← 1008, INT8 VE（實驗性，見 quantization_guide.md）
└── tokenizer.json             ← 共用
```

---

## 第一階段：安裝 Docker + GPU 支援

> 如果你已經有 Docker 和 `nvidia-container-toolkit`，跳到第二階段。

### 1.1 安裝 nvidia-container-toolkit

這個工具讓 Docker 容器可以使用 GPU。

```bash
# 加入 NVIDIA 的套件來源
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安裝
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 設定 Docker 使用 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 1.2 驗證 GPU 可用

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

如果看到你的 GPU 資訊就成功了。

---

## 第二階段：匯出 ONNX（任意 GPU，只需做一次）

ONNX 是 GPU 無關的中間格式。你只需要匯出一次，產出的 `.onnx` 檔可以在任何 GPU 上轉成 TensorRT 引擎。

### 2.1 關於 SAM3 模型

- **模型名稱：** SAM3 (Segment Anything Model 3)，Meta AI Research 開發
- **HuggingFace：** `facebook/sam3`
- **授權：** SAM License（詳見模型頁面）

### 2.2 建立 PyTorch 容器

匯出 ONNX 需要 PyTorch + transformers，我們用一個獨立容器來做：

```bash
docker run -d \
  --name sam3_pytorch \
  --gpus all \
  --ipc=host \
  -v "$(pwd)/..:/root/sam3_pipeline" \
  nvidia/cuda:12.8.0-devel-ubuntu24.04 \
  sleep infinity
```

> `$(pwd)/..` 會把 `sam3_pipeline/` 整個資料夾掛載到容器的 `/root/sam3_pipeline`。

安裝必要套件：

```bash
docker exec sam3_pytorch bash -c "
  apt-get update && apt-get install -y python3 python3-pip git
  pip3 install --break-system-packages \
    torch torchvision torchaudio \
    'transformers>=5.0' \
    huggingface_hub
"
```

### 2.3 安裝 SAM3 套件

```bash
docker exec sam3_pytorch bash -c "
  cd /tmp && git clone https://github.com/facebookresearch/sam3.git
  cd sam3 && pip3 install --break-system-packages -e .
"
```

### 2.4 下載模型 & 匯出 ONNX

```bash
# 如果模型需要登入 HuggingFace：
# docker exec sam3_pytorch huggingface-cli login

docker exec sam3_pytorch python3 \
  /root/sam3_pipeline/setup/export_sam3_to_onnx.py \
  --all \
  --model-path facebook/sam3 \
  --output-dir /root/sam3_pipeline/setup/onnx \
  --image-size 1008 \
  --opset-version 16 \
  --device cuda
```

完成後會產生 4 個 ONNX 檔案（共約 1.6 GB）：

```
setup/onnx/
├── vision-encoder.onnx     # 878 MB
├── text-encoder.onnx       # 675 MB
├── geometry-encoder.onnx   #  16 MB
└── decoder.onnx            #  47 MB (queries=200)
```

**匯出 VRAM 優化版 decoder（queries=50）：**

如果只需要匯出 queries=50 的 decoder，不需要重新匯出其他 3 個 encoder：

```bash
docker exec sam3_pytorch python3 \
  /root/sam3_pipeline/setup/export_sam3_to_onnx.py \
  --module decoder \
  --model-path facebook/sam3 \
  --output-dir /root/sam3_pipeline/setup/onnx_q50 \
  --num-queries 50 \
  --device cuda
```

`--num-queries` 控制 decoder 輸出候選數量。DETR 內部仍用完整 200 queries 做 cross-attention，偵測品質不受影響，只有 output buffer 縮小。詳見 [`optimize.md`](../optimize.md)。

PyTorch 容器的任務到此結束，可以刪除：

```bash
docker stop sam3_pytorch && docker rm sam3_pytorch
```

---

## 第三階段：建立 TensorRT 容器 + 轉換引擎

### 3.1 建立 TensorRT 容器

```bash
docker pull nvcr.io/nvidia/tensorrt:25.03-py3

docker run -d \
  --name sam3_trt \
  --gpus all \
  --ipc=host \
  -v "$(pwd)/..:/root/sam3_pipeline" \
  nvcr.io/nvidia/tensorrt:25.03-py3 \
  sleep infinity
```

這個容器包含：Ubuntu 24.04 + CUDA 12.8 + cuDNN 9.8 + TensorRT 10.9 + Python 3.12 + trtexec。

### 3.2 安裝 Python 套件

```bash
docker exec sam3_trt pip install \
  pycuda==2025.1 \
  numpy==1.26.4 \
  opencv-python-headless==4.10.0.84 \
  pillow==11.1.0 \
  tokenizers==0.22.2
```

### 3.3 驗證環境

```bash
docker exec sam3_trt python3 -c "
import tensorrt as trt; print('TensorRT', trt.__version__)
import pycuda.driver as d; import pycuda.autoinit
dev = d.Device(0)
print('GPU:', dev.name(), '-', dev.total_memory() // 1024**2, 'MB')
print('OK')
"
```

### 3.4 轉換引擎

> **如果你的 GPU 跟我們一樣（RTX 5090），跳過這步** — `engines/` 裡的引擎可以直接用。

```bash
# 轉換標準版 (queries=200)
docker exec sam3_trt bash /root/sam3_pipeline/setup/onnx_to_tensorrt.sh \
  /root/sam3_pipeline/setup/onnx \
  /root/sam3_pipeline/engines/b8_q200

# 轉換優化版 decoder (queries=50) — 只需要 decoder
docker exec sam3_trt trtexec --fp16 \
  --onnx=/root/sam3_pipeline/setup/onnx_q50/decoder.onnx \
  --saveEngine=/root/sam3_pipeline/engines/b8_q50/decoder.engine \
  --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x1x256,prompt_mask:1x1 \
  --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x33x256,prompt_mask:1x33 \
  --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,prompt_features:8x60x256,prompt_mask:8x60
```

轉換約需 2~15 分鐘（decoder 最快 ~2 分鐘，VE 最久）。

### 3.5 驗證引擎

```bash
docker exec sam3_trt python3 -c "
import tensorrt as trt
trt.init_libnvinfer_plugins(trt.Logger(), '')
for name in ['vision-encoder', 'text-encoder', 'geometry-encoder', 'decoder']:
    path = '/root/sam3_pipeline/engines/b8_q200/' + name + '.engine'
    with open(path, 'rb') as f:
        engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
    print(name + ':', 'OK' if engine else 'FAILED')
"
```

4 個都顯示 OK 就完成了。

### 3.6 轉換不同解析度的引擎（降 VRAM / 提速）

預設 1008 解析度在 8 路攝影機時需要大量 VRAM。降低解析度可以大幅省 VRAM 和提速。
完整原理見 [`resolution_guide.md`](resolution_guide.md)。

**核心概念**：解析度是「烤」進 ONNX 的（RoPE 位置編碼綁定 patch 數量），不同解析度需要各自匯出 ONNX。

以 672 解析度為例：

```bash
# === 步驟 1: 匯出 672 的 ONNX（在 PyTorch 容器中） ===

# VE — 必須重新匯出（RoPE 綁定解析度）
docker exec <sam3 容器> python3 \
  /root/sam3_pipeline/setup/export_sam3_to_onnx.py \
  --module vision \
  --model-path facebook/sam3 \
  --image-size 672 \
  --output-dir /root/sam3_pipeline/setup/onnx_r672_q50 \
  --device cuda

# Decoder — 必須重新匯出（FPN 輸入大小綁定解析度）
docker exec <sam3 容器> python3 \
  /root/sam3_pipeline/setup/export_sam3_to_onnx.py \
  --module decoder \
  --model-path facebook/sam3 \
  --image-size 672 \
  --num-queries 50 \
  --output-dir /root/sam3_pipeline/setup/onnx_r672_q50 \
  --device cuda

# GE — 必須重新匯出（FPN 輸入大小綁定解析度）
docker exec <sam3 容器> python3 \
  /root/sam3_pipeline/setup/export_sam3_to_onnx.py \
  --module geometry \
  --model-path facebook/sam3 \
  --image-size 672 \
  --output-dir /root/sam3_pipeline/setup/onnx_r672_q50 \
  --device cuda

# TE — 跟解析度無關，symlink 共用
cd /root/sam3_pipeline/setup/onnx_r672_q50
ln -s ../onnx_q50/text-encoder.onnx text-encoder.onnx
```

> **匯出參數說明**：
> - `--module`：只匯出指定子模型（vision/decoder/geometry/text）
> - `--image-size`：輸入解析度，必須是 14 的倍數
> - `--num-queries`：decoder 輸出候選數（50 = VRAM 優化版，詳見 [`optimize.md`](../optimize.md)）
> - 匯出每個子模型約需 1-2 分鐘（加載模型 + ONNX trace）

```bash
# === 步驟 2: 建 TensorRT 引擎（在 TensorRT 容器中） ===

docker exec sam3_trt bash /root/sam3_pipeline/setup/onnx_to_tensorrt.sh \
  /root/sam3_pipeline/setup/onnx_r672_q50 \
  /root/sam3_pipeline/engines/b8_q50_r672 \
  672
```

> `onnx_to_tensorrt.sh` 的第三個參數是解析度。它會自動計算 FPN 尺寸：
> `patches = 672 / 14 = 48` → FPN = 192/96/48
>
> 轉換約需 10-15 分鐘（VE 最久 ~6 分鐘）。

```bash
# === 步驟 3: 建立 config ===

# 複製一份 config，把 engines 路徑改成新的
cp config_q50.json config_q50_r672.json
# 編輯: "engines": "engines/b8_q50_r672"
```

```bash
# === 步驟 4: 驗證 ===

docker exec sam3_trt python3 /root/sam3_pipeline/infer.py \
  --config /root/sam3_pipeline/config_q50_r672.json \
  --images /root/sam3_pipeline/Inputs/demo_3.jpg \
  --output /root/sam3_pipeline/outputs/test_672
```

560 解析度流程完全相同，把所有 `672` 換成 `560`，`r672` 換成 `r560`。

**哪些需要重新匯出？**

| 子模型 | 跟解析度有關？ | 原因 |
|--------|:---:|------|
| Vision Encoder | ✅ 必須 | RoPE 位置編碼綁定 patch 數 |
| Geometry Encoder | ✅ 必須 | FPN 輸入大小綁定解析度 |
| **Text Encoder** | **❌** | 只處理文字 token，跟解析度無關 |
| Decoder | ✅ 必須 | FPN 輸入大小綁定解析度 |

### 3.7 清理（選擇性）

ONNX 檔案不再需要，可以刪除以節省空間：

```bash
rm -rf setup/onnx/
```

---

## 完成！

環境和引擎都準備好了。回到上一層看 [README.md](../README.md) 開始跑偵測。

流程總結：

```
setup/export_sam3_to_onnx.py  →  setup/onnx/  →  setup/onnx_to_tensorrt.sh  →  engines/
     (PyTorch → ONNX)            (中間檔案)         (ONNX → TensorRT)          (最終引擎)
```

---

## 容器常用指令

```bash
docker start sam3_trt                    # 啟動容器
docker stop sam3_trt                     # 停止容器
docker exec -it sam3_trt bash            # 進入容器 shell
docker exec sam3_trt nvidia-smi          # 查看 GPU 資訊
```

---

## 最大類別數（MAX_CLASSES）與 VRAM

### 原理

SAM3 的 decoder 用 **batch 維度** 來同時處理多個類別 — batch=1 就是 1 個類別，batch=4 就是 4 個類別。TensorRT 引擎在轉換時需要指定 `maxShapes`（所有輸入的最大維度），這個值會**鎖死**引擎能處理的最大 batch。

重點：**TensorRT 會為 maxShapes 預留 activation memory**，不管你實際用幾個類別，VRAM 佔用量都差不多。所以選擇 maxShapes 本質上就是在選擇「VRAM 預算」。

### 實測 VRAM（RTX 5090, 4 classes: 3 text + 1 image）

| 引擎 maxShapes | VRAM 佔用 | 適合 GPU |
|----------------|-----------|----------|
| batch=4 | **~5.0 GB** | 8 GB 以上 |
| batch=8（預設） | **~7.5 GB** | 12 GB 以上 |
| batch=16（估計） | **~12 GB** | 16 GB 以上 |

> VRAM 主要被 decoder 的 FPN 特徵圖吃掉（`fpn_feat_0` 每個 class 佔 ~80 MB），加上 TensorRT 內部的 activation memory。
> Image prompt 的 class 比 text prompt 略大（多了 geometry 特徵），但差異在 1-2 MB 以內，可忽略。

### 如何調整

修改 `onnx_to_tensorrt.sh`，把**所有** `maxShapes` 和 `optShapes` 裡的 batch 數字統一改掉：

```bash
# 例如改成 max 4 classes：
# 把所有 8x... 改成 4x...
--optShapes=images:4x3x1008x1008
--maxShapes=images:4x3x1008x1008
# ... 對所有 4 個引擎都要改
```

**必須同時改 4 個引擎**（vision-encoder、text-encoder、geometry-encoder、decoder），batch 數字要一致。

改完後重新轉換：

```bash
docker exec sam3_trt bash /root/sam3_pipeline/setup/onnx_to_tensorrt.sh \
  /root/sam3_pipeline/setup/onnx
```

然後把 `extract.py`、`infer.py`、`config_editor.py` 裡的 `MAX_CLASSES` 常數也改成對應的值。

### 引擎目錄結構

引擎按 `b{batch}_q{queries}[_r{resolution}][_int8]` 命名，每組 4 個引擎必須匹配：

```
engines/
├── b8_q200/                      ← 1008, batch=8, queries=200
│   ├── vision-encoder.engine
│   ├── text-encoder.engine
│   ├── geometry-encoder.engine
│   └── decoder.engine
├── b8_q50/                       ← 1008, batch=8, queries=50（推薦預設）
│   ├── vision-encoder.engine     → symlink to b8_q200/
│   ├── text-encoder.engine       → symlink to b8_q200/
│   ├── geometry-encoder.engine   → symlink to b8_q200/
│   └── decoder.engine            （Q50 獨立建置）
├── b8_q50_r840/                  ← 840 解析度（12 GB+ VRAM）
│   └── ...                       （VE/GE/decoder 各自建置，TE symlink）
├── b8_q50_r672/                  ← 672 解析度（8 GB 部署推薦）
│   └── ...
├── b8_q50_r560/                  ← 560 解析度（8 GB 部署最省 VRAM）
│   └── ...
├── b8_q50_int8/                  ← 1008, INT8 VE（實驗性）
│   ├── vision-encoder.engine     （INT8 獨立建置）
│   ├── text-encoder.engine       → symlink to b8_q50/
│   ├── geometry-encoder.engine   → symlink to b8_q50/
│   └── decoder.engine            → symlink to b8_q50/
└── tokenizer.json                ← 共用
```

**Symlink 規則**：
- 只有 **decoder** 受 queries 數量影響（Q200 vs Q50）
- 只有 **VE / GE / decoder** 受解析度影響，**TE 永遠可以共用**
- 只有 **VE** 受精度影響（INT8 vs FP16），其他三個保持 FP16

切換引擎只需改 `config.json` 的 `engines` 路徑（如 `"engines": "engines/b8_q50_r672"`）。

**相關文件**：
- Q200 vs Q50 比較 → [`optimize.md`](../optimize.md)
- 解析度選擇指南 → [`resolution_guide.md`](resolution_guide.md)
- INT8 量化實驗 → [`quantization_guide.md`](quantization_guide.md)

---

## 常見問題

| 問題 | 解法 |
|------|------|
| 引擎載入失敗 / segfault | 你的 GPU 跟轉換時不同，需要重新執行 `onnx_to_tensorrt.sh` |
| ONNX 匯出失敗 | 確認 PyTorch 和 transformers 版本，需要 `transformers>=5.0` |
| `docker: permission denied` | 把使用者加入 docker 群組：`sudo usermod -aG docker $USER` |
| VRAM 不足 | 改用較小的 maxShapes batch（見上方「最大類別數」章節），或檢查是否有其他程式佔用 GPU |
| `tokenizers` import error | `pip install tokenizers`（只有 extract.py 需要） |
| `ROIAlign_TRT plugin not found` | 程式碼已處理（`trt.init_libnvinfer_plugins`），不需額外操作 |

---

## 開發環境（本引擎的建置環境）

```
GPU:        NVIDIA GeForce RTX 5090 (32 GB, compute 12.0)
Driver:     570.195.03
CUDA:       12.8
cuDNN:      9.8.0
TensorRT:   10.9.0.34
Python:     3.12.3
Container:  nvcr.io/nvidia/tensorrt:25.03-py3
```
