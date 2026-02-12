兩步驟：

  1. extract.py — 每次改 config
  跑一次，產生特徵檔到
  features/（有快取，沒改的類別自動跳過）
  2. infer.py — 吃 features/
  的特徵跑偵測，不需要 text/geo encoder


1. 結構跟流程夠好嗎？                    
                                         
  我覺得架構設計是乾淨的，兩步分離的核心好 
  處是：                                   
                                           
  - extract.py 載入 3-4 個引擎，但只跑一次
  - infer.py 只載入 2 個引擎（vision +
  decoder），推論時很輕

  但有個地方可以更好：現在 extract
  還是要載入全部 3 個引擎，即使你只用 text
  prompt。理論上可以按需載入，但考量到它只
  跑一次，影響不大，先不過度設計。

  2. Image prompt（比對圖）有支援嗎？

  程式碼有寫進去了，但還沒實測過。
  extract.py 裡面的 _geo_features()
  完整實作了：

  1. 讀參考圖片 → 跑 vision encoder
  2. 拿 fpn_feat_2 + fpn_pos_2 + boxes +
  labels → 跑 geometry encoder
  3. 得到 geo features [1, N+1, 256]

  prompt_type: "both" 的情況也有處理，會把
  text features 跟 geo features concat
  起來。

  但這部分需要你提供真正的參考圖片才能驗證
  。 目前只有 text prompt 是跑過確認 OK
  的。

  3. Multi-class one-time inference？

  是的，所有 class 在單次 decoder forward
  pass 完成。 流程是：

  Vision encoder: 1 次  (~32ms)
           ↓
  FPN features → GPU d2d copy 複製 N 份
  (~1ms)
           ↓
  Decoder: 1 次，batch=N  (~14ms)

  不是跑 N 次 decoder，是靠 batch
  維度一次搞定。所以 2 class 跟 4 class
  差距很小。

  4. N 是固定的還是浮動的？

  執行期間是固定的。 Pipeline.__init__
  的時候根據 config 裡有幾個 class
  就預先分配好所有 GPU buffer：

  - batch_fpn_gpu：N 份 FPN
  - prompt_feat_gpu：[N, P, 256]
  - out_gpu：[N, 200, 288, 288] 的 mask
  buffer 等等

  這些 buffer 大小是寫死的，改 N 要重啟。

  不過 decoder engine 本身支援動態
  batch（建引擎的時候 maxShapes 設到
  4），所以技術上可以做成動態的 —
  但要重新分配
  buffer，目前沒必要做這個複雜度。

  5. TensorRT 怎麼轉的？

  我沒有重新轉，引擎是之前就建好的。
  整個流程是：

  PyTorch 模型 (facebook/sam3)
         ↓
  export_onnx.py  →  4 個 .onnx 檔
         ↓  (定義 dynamic_axes: batch,
  prompt_len, num_boxes)
  trtexec  →  4 個 .engine 檔
         ↓  (指定 minShapes / optShapes /
  maxShapes)
  engines/ 裡的現成引擎

  export_onnx.py
  裡定義了每個引擎的動態軸，例如 decoder：

  - prompt_features: {0: "batch", 1:
  "prompt_len"}
  - prompt_mask: {0: "batch", 1:
  "prompt_len"}

  然後用 trtexec 建引擎時指定範圍，例如
  batch 最小 1、最大 4，prompt_len 最小
  1、最大 60。TensorRT
  就會在這個範圍內做最佳化。

  所以我們的 pipeline 能用不同的
  prompt_len（text=32, geo 可能=5,
  both=37），都不需要重建引擎，只要不超過
  60 就行。