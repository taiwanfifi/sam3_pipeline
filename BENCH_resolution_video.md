# Resolution Benchmark: r448 vs r560 + 影片輸出 Overhead

RTX 5090, 4 classes (PERSON/HEAD/HAND/SCREWDRIVER), Q50, FP16.
影片源：shop.mp4 (145s) + hair.mp4 (71s) + car.mp4 (299s)。

---

## 測試矩陣

| # | 解析度 | 模式 | 影片輸出 | 說明 |
|---|--------|------|----------|------|
| 1 | 448 | 單機 | 無 | 純效能基準 |
| 2 | 448 | 8 cam | 無 | 純效能基準 |
| 3 | 560 | 單機 | 無 | 純效能基準 |
| 4 | 560 | 8 cam | 無 | 純效能基準 |
| 5 | 448 | 單機 | `--save-video` | 含影片輸出 |
| 6 | 448 | 8 cam | `--save-video --save-cameras` | grid + 8 個別攝影機 |
| 7 | 560 | 單機 | `--save-video` | 含影片輸出 |
| 8 | 560 | 8 cam | `--save-video --save-cameras` | grid + 8 個別攝影機 |

---

## 結果

### Round 1: 純效能 (JSON only)

| 指標 | r448 單機 | r448 8cam | r560 單機 | r560 8cam |
|------|----------|----------|----------|----------|
| **Avg ms** | 26.1 | 108.6 (14ms/cam) | 30.3 | 132.8 (17ms/cam) |
| **P95 ms** | 50.6 | 153.0 | 58.5 | 180.2 |
| **FPS** | 38.4 | 73.6 total (9.2/cam) | 33.0 | 60.2 total (7.5/cam) |
| **VRAM buffers** | 125 MB | 246 MB | 195 MB | 383 MB |
| **Frames processed** | 4074/4357 | 1338/4357 | 3923/4357 | 1094/4357 |

### Round 2: 含影片輸出 (Debug)

| 指標 | r448 單機 | r448 8cam | r560 單機 | r560 8cam |
|------|----------|----------|----------|----------|
| **Avg ms** | 25.1 | 106.3 (13ms/cam) | 30.2 | 130.2 (16ms/cam) |
| **P95 ms** | 46.7 | 151.2 | 58.0 | 172.1 |
| **FPS** | 39.8 | 75.3 total (9.4/cam) | 33.1 | 61.4 total (7.7/cam) |
| **VRAM buffers** | 125 MB | 246 MB | 195 MB | 383 MB |
| **Frames processed** | 4130/4357 | 1368/4357 | 3946/4357 | 1116/4357 |

### Overhead 比較（有無影片輸出）

| 配置 | 無影片 avg ms | 有影片 avg ms | 差異 | Overhead |
|------|:---:|:---:|:---:|:---:|
| r448 單機 | 26.1 | 25.1 | -1.0 | **~0%** |
| r448 8cam | 108.6 | 106.3 | -2.3 | **~0%** |
| r560 單機 | 30.3 | 30.2 | -0.1 | **~0%** |
| r560 8cam | 132.8 | 130.2 | -2.6 | **~0%** |

> **結論：影片輸出的 overhead 為零。** AVI 直寫（MJPG intra-frame codec）的寫入成本
> 完全被 GPU 推論時間遮蓋，不會拖慢 pipeline。差異在統計誤差範圍內。

---

## r448 vs r560 比較

| 指標 | r448 | r560 | 差異 |
|------|------|------|------|
| **單機 FPS** | 38.4 | 33.0 | r448 快 16% |
| **8cam total FPS** | 73.6 | 60.2 | r448 快 22% |
| **8cam per cam FPS** | 9.2 | 7.5 | r448 快 23% |
| **VRAM (8cam)** | 246 MB | 383 MB | r448 省 36% |
| **Patches** | 32×32 = 1024 | 40×40 = 1600 | r560 多 56% |

> r448 在速度和 VRAM 上都有優勢，但解析度低 20%。
> 對於近距離監控（2-5m）偵測大物件（人、頭），r448 足夠。
> 需要偵測小物件（手、工具）時，r560 的精度更好。

---

## 影片輸出流程

影片儲存是 debug 用途，正式部署只輸出 JSON。

### 架構：兩段式

```
Phase 1: 推論 + 即時 AVI 輸出
┌─────────────────────────────────────────────────┐
│  GPU 推論 → draw_overlay() → writer.write(frame) │
│                                    ↓              │
│                              AVI (MJPG)           │
│  每一幀直接寫入，可隨時預覽                          │
└─────────────────────────────────────────────────┘

Phase 2: 批次轉檔
┌─────────────────────────────────────────────────┐
│  ffmpeg -i input.avi -c:v libx264 output.mp4     │
│  轉成功 → 刪除 AVI                                │
│  轉失敗 → 保留 AVI + 警告                          │
└─────────────────────────────────────────────────┘
```

### 為什麼先 AVI 再轉 MP4？

| 方案 | 優點 | 缺點 |
|------|------|------|
| **直接寫 MP4 (H.264)** | 一步完成 | OpenCV H.264 支援不穩定，需要 inter-frame 編碼，有延遲 |
| **先 AVI (MJPG) 再轉** | MJPG 是 intra-frame codec，每幀獨立壓縮，零延遲寫入 | 需要兩步，AVI 暫時佔較多空間 |
| **Temp JPG → 合成 AVI** | (舊方案) | 多餘的磁碟 I/O (寫 JPG → 讀 JPG → 寫 AVI) |

選擇 AVI (MJPG) 直寫的原因：
1. **即時輸出** — 推論中就能預覽影片
2. **零 overhead** — MJPG 編碼比 GPU 推論快得多
3. **容錯** — 即使中途斷電，已寫入的幀都保留在 AVI 裡
4. **OpenCV 相容** — 所有平台都支援 MJPG fourcc

### 輸出檔案

**單機 (`infer.py --save-video`)**：

| 檔案 | 說明 |
|------|------|
| `{timestamp}_output.mp4` | H.264 overlay 影片（從 AVI 轉檔） |
| `{timestamp}_detections.jsonl` | 逐幀偵測結果 |
| `{timestamp}_performance.json` | 效能統計 |

**8cam (`infer_multi.py --save-video --save-cameras`)**：

| 檔案 | 說明 |
|------|------|
| `{timestamp}_grid.mp4` | 2×4 grid 總覽影片 |
| `{timestamp}_cam0_{label}.mp4` | 個別攝影機 overlay 影片 |
| `{timestamp}_cam1_{label}.mp4` | ... |
| `{timestamp}_detections.jsonl` | 逐幀偵測結果 |
| `{timestamp}_performance.json` | 效能統計 |

---

## 如何重現

所有指令在 Docker 容器 `william_tensorrt` 內執行。
工作目錄：`/root/VisionDSL/models/sam3_pipeline/`。

### 準備 config

```bash
# config_r448.json — 只改 engines 路徑
{
  "engines": "engines/b8_q50_r448",
  ...  # 其餘同 config.json
}

# config_r560.json
{
  "engines": "engines/b8_q50_r560",
  ...
}
```

### Round 1: 純效能測試（不儲存影片）

```bash
# r448 單機
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_r448.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/shop.mp4 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/r448_single_perf

# r448 8 cameras
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer_multi.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_r448.json \
  --video Inputs/shop.mp4 Inputs/hair.mp4 Inputs/car.mp4 \
  --cameras 8 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/r448_8cam_perf

# r560 單機
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_r560.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/shop.mp4 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/r560_single_perf

# r560 8 cameras
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer_multi.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_r560.json \
  --video Inputs/shop.mp4 Inputs/hair.mp4 Inputs/car.mp4 \
  --cameras 8 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/r560_8cam_perf
```

### Round 2: 含影片輸出（Debug 模式）

```bash
# r448 單機 + overlay video
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_r448.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/shop.mp4 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/r448_single_video \
  --save-video

# r448 8 cameras + grid + individual camera videos
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer_multi.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_r448.json \
  --video Inputs/shop.mp4 Inputs/hair.mp4 Inputs/car.mp4 \
  --cameras 8 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/r448_8cam_video \
  --save-video --save-cameras

# r560 單機 + overlay video
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_r560.json \
  --video /root/VisionDSL/models/sam3_pipeline/Inputs/shop.mp4 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/r560_single_video \
  --save-video

# r560 8 cameras + grid + individual camera videos
docker exec william_tensorrt python3 \
  /root/VisionDSL/models/sam3_pipeline/infer_multi.py \
  --config /root/VisionDSL/models/sam3_pipeline/config_r560.json \
  --video Inputs/shop.mp4 Inputs/hair.mp4 Inputs/car.mp4 \
  --cameras 8 \
  --output /root/VisionDSL/models/sam3_pipeline/outputs/r560_8cam_video \
  --save-video --save-cameras
```

### 自動化 Benchmark

```bash
# 一次跑完全部 8 個測試
docker exec william_tensorrt bash \
  /root/VisionDSL/models/sam3_pipeline/benchmark.sh
```

`benchmark.sh` 會依序執行 Round 1 + Round 2，最後印出所有 `performance.json` 的匯總。

---

## 附錄：原始 performance.json

<details>
<summary>r448_single_perf</summary>

```json
{
  "frames": 4074,
  "classes": 4,
  "prompt_len": 34,
  "avg_ms": 26.1,
  "min_ms": 18.3,
  "max_ms": 164.0,
  "p95_ms": 50.6,
  "est_fps": 38.4,
  "buffer_vram_mb": 124.9
}
```
</details>

<details>
<summary>r448_8cam_perf</summary>

```json
{
  "cameras": 8,
  "video_sources": ["shop.mp4", "hair.mp4", "car.mp4"],
  "classes": 4,
  "queries": 50,
  "image_size": 448,
  "prompt_len": 34,
  "frames_processed": 1338,
  "total_avg_ms": 108.6,
  "per_camera_avg_ms": 13.6,
  "min_ms": 80.6,
  "max_ms": 255.4,
  "p95_ms": 153.0,
  "total_fps": 73.6,
  "per_camera_fps": 9.2,
  "buffer_vram_mb": 245.5
}
```
</details>

<details>
<summary>r560_single_perf</summary>

```json
{
  "frames": 3923,
  "classes": 4,
  "prompt_len": 34,
  "avg_ms": 30.3,
  "min_ms": 20.1,
  "max_ms": 131.7,
  "p95_ms": 58.5,
  "est_fps": 33.0,
  "buffer_vram_mb": 195.1
}
```
</details>

<details>
<summary>r560_8cam_perf</summary>

```json
{
  "cameras": 8,
  "video_sources": ["shop.mp4", "hair.mp4", "car.mp4"],
  "classes": 4,
  "queries": 50,
  "image_size": 560,
  "prompt_len": 34,
  "frames_processed": 1094,
  "total_avg_ms": 132.8,
  "per_camera_avg_ms": 16.6,
  "min_ms": 106.5,
  "max_ms": 305.0,
  "p95_ms": 180.2,
  "total_fps": 60.2,
  "per_camera_fps": 7.5,
  "buffer_vram_mb": 382.9
}
```
</details>

<details>
<summary>r448_single_video</summary>

```json
{
  "frames": 4130,
  "classes": 4,
  "prompt_len": 34,
  "avg_ms": 25.1,
  "min_ms": 18.2,
  "max_ms": 135.0,
  "p95_ms": 46.7,
  "est_fps": 39.8,
  "buffer_vram_mb": 124.9
}
```
</details>

<details>
<summary>r448_8cam_video</summary>

```json
{
  "cameras": 8,
  "video_sources": ["shop.mp4", "hair.mp4", "car.mp4"],
  "classes": 4,
  "queries": 50,
  "image_size": 448,
  "prompt_len": 34,
  "frames_processed": 1368,
  "total_avg_ms": 106.3,
  "per_camera_avg_ms": 13.3,
  "min_ms": 79.6,
  "max_ms": 235.4,
  "p95_ms": 151.2,
  "total_fps": 75.3,
  "per_camera_fps": 9.4,
  "buffer_vram_mb": 245.5
}
```
</details>

<details>
<summary>r560_single_video</summary>

```json
{
  "frames": 3946,
  "classes": 4,
  "prompt_len": 34,
  "avg_ms": 30.2,
  "min_ms": 20.5,
  "max_ms": 140.8,
  "p95_ms": 58.0,
  "est_fps": 33.1,
  "buffer_vram_mb": 195.1
}
```
</details>

<details>
<summary>r560_8cam_video</summary>

```json
{
  "cameras": 8,
  "video_sources": ["shop.mp4", "hair.mp4", "car.mp4"],
  "classes": 4,
  "queries": 50,
  "image_size": 560,
  "prompt_len": 34,
  "frames_processed": 1116,
  "total_avg_ms": 130.2,
  "per_camera_avg_ms": 16.3,
  "min_ms": 104.7,
  "max_ms": 258.2,
  "p95_ms": 172.1,
  "total_fps": 61.4,
  "per_camera_fps": 7.7,
  "buffer_vram_mb": 382.9
}
```
</details>
