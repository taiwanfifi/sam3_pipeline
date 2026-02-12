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
├── export_sam3_to_onnx.py     ← PyTorch → ONNX 匯出腳本
└── onnx_to_tensorrt.sh        ← ONNX → TensorRT 引擎轉換腳本
```

轉換完成後，引擎會自動輸出到上一層的 `engines/` 資料夾。

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
└── decoder.onnx            #  47 MB
```

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
docker exec sam3_trt bash /root/sam3_pipeline/setup/onnx_to_tensorrt.sh \
  /root/sam3_pipeline/setup/onnx
```

轉換約需 5~15 分鐘。完成後引擎會出現在 `engines/` 資料夾。

### 3.5 驗證引擎

```bash
docker exec sam3_trt python3 -c "
import tensorrt as trt
trt.init_libnvinfer_plugins(trt.Logger(), '')
for name in ['vision-encoder', 'text-encoder', 'geometry-encoder', 'decoder']:
    path = '/root/sam3_pipeline/engines/' + name + '.engine'
    with open(path, 'rb') as f:
        engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
    print(name + ':', 'OK' if engine else 'FAILED')
"
```

4 個都顯示 OK 就完成了。

### 3.6 清理（選擇性）

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

## 常見問題

| 問題 | 解法 |
|------|------|
| 引擎載入失敗 / segfault | 你的 GPU 跟轉換時不同，需要重新執行 `onnx_to_tensorrt.sh` |
| ONNX 匯出失敗 | 確認 PyTorch 和 transformers 版本，需要 `transformers>=5.0` |
| `docker: permission denied` | 把使用者加入 docker 群組：`sudo usermod -aG docker $USER` |
| VRAM 不足 | 你的 GPU 記憶體可能不夠，可以試試減少 `--maxShapes` 的 batch |
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
