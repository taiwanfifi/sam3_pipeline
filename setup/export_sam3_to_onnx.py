"""
SAM3 ONNX 匯出腳本 — 將 SAM3 模型從 PyTorch 轉為 ONNX 格式

SAM3 (Segment Anything Model 3) 是 Meta AI 開發的分割模型，由 4 個子模型組成：

  1. Vision Encoder  — 影像編碼器：用 ViT 骨幹網路從圖片萃取多尺度特徵 (FPN)
  2. Text Encoder    — 文字編碼器：用 CLIP 將文字描述轉為 256 維特徵向量
  3. Geometry Encoder — 幾何編碼器：將參考圖片中的框選區域轉為 256 維特徵
  4. Decoder          — 解碼器：融合上述特徵，輸出偵測框 (boxes)、遮罩 (masks)、信心分數

什麼是 ONNX？
  ONNX (Open Neural Network Exchange) 是開放的神經網路交換格式。
  它與 GPU 架構無關 — 同一份 .onnx 檔案可以在任何 GPU 上使用。
  匯出後再用 trtexec 轉成 TensorRT 引擎（針對你的 GPU 優化）。

使用方式：
  python3 export_sam3_to_onnx.py --all \\
    --model-path facebook/sam3 \\
    --output-dir ./onnx \\
    --image-size 1008 \\
    --opset-version 16 \\
    --device cuda

來源：
  模型：https://huggingface.co/facebook/sam3
  程式碼：https://github.com/facebookresearch/sam3
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

# transformers 是 HuggingFace 的模型庫，提供 SAM3 的 PyTorch 實作
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.sam3.modeling_sam3 import Sam3Model, Sam3ViTRotaryEmbedding


# ---------------------------------------------------------------------------
# RoPE 解析度修補 — 讓 Vision Encoder 支援非原始解析度匯出
# ---------------------------------------------------------------------------

def patch_rope_for_resolution(model: Sam3Model, target_image_size: int):
    """
    修補 ViT 全域注意力層的旋轉位置編碼 (RoPE)，使其匹配目標解析度。

    SAM3 的 ViT backbone 有 32 層：
      - 28 個 windowed 層使用固定 24×24 窗口 → RoPE = [576, 64]（不受影響）
      - 4 個 global 層 (7, 15, 23, 31) 使用完整 patch grid → RoPE = [patches², 64]

    原始模型的 global RoPE 為 72×72 = 5184 位置（對應 1008×1008 解析度）。
    如果目標解析度不同（如 840×840 → 60×60 = 3600 位置），
    直接推論會因維度不匹配而報錯。

    此函式在模型載入後、ONNX 匯出前，用目標解析度重新計算 global 層的 RoPE buffer。
    Windowed 層和所有模型權重完全不動。

    當 target_image_size 等於原始解析度時，此函式不做任何修改。
    """
    backbone = model.vision_encoder.backbone
    config = backbone.config

    original_image_size = config.image_size  # 通常是 1008
    patch_size = config.patch_size            # 14

    original_patches = original_image_size // patch_size
    target_patches = target_image_size // patch_size

    if target_patches == original_patches:
        return  # 解析度相同，不需要修補

    print(f"  RoPE 修補: {original_patches}×{original_patches} → {target_patches}×{target_patches} "
          f"(global 層: {config.global_attn_indexes})")

    window_size = config.window_size  # 24

    for layer_idx in config.global_attn_indexes:
        layer = backbone.layers[layer_idx]
        # global 層的 window_size == 0 → RoPE 覆蓋整個 patch grid
        # scale = window_size / patches（正規化位置到窗口相對座標）
        scale = window_size / target_patches
        new_rope = Sam3ViTRotaryEmbedding(
            config,
            end_x=target_patches,
            end_y=target_patches,
            scale=scale,
        )
        # 搬到與原始 RoPE 相同的裝置
        device = layer.rotary_emb.rope_embeddings_cos.device
        new_rope = new_rope.to(device)
        layer.rotary_emb = new_rope

    old_positions = original_patches * original_patches
    new_positions = target_patches * target_patches
    print(f"  RoPE 修補完成: {old_positions} → {new_positions} 位置 "
          f"(4 個 global 層已更新)")


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """座標格式轉換：中心點+寬高 [cx, cy, w, h] → 左上右下 [x1, y1, x2, y2]"""
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1
    )


def compute_sine_position_encoding(
    shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
) -> torch.Tensor:
    """
    產生正弦位置編碼 (Sine Position Encoding)。

    位置編碼讓模型知道每個像素在圖片中的空間位置，
    類似 Transformer 文字模型中的 positional encoding，但這裡是二維 (x, y)。
    使用 sin/cos 交替編碼不同頻率，讓模型能區分遠近關係。

    為什麼不用 cumsum？因為 TensorRT 對 cumsum 的支援不完整，
    改用 arange 可以達成相同效果且相容 TensorRT。
    """
    batch_size, channels, height, width = shape

    # 產生 y 軸與 x 軸的座標 (1 到 height/width，正規化到 0~2pi)
    y_embed = (
        torch.arange(1, height + 1, dtype=dtype, device=device)
        .view(1, height, 1).expand(batch_size, height, width)
    )
    x_embed = (
        torch.arange(1, width + 1, dtype=dtype, device=device)
        .view(1, 1, width).expand(batch_size, height, width)
    )

    eps = 1e-6
    y_embed = y_embed / (height + eps) * scale
    x_embed = x_embed / (width + eps) * scale

    # 不同維度使用不同頻率的 sin/cos，讓模型學到豐富的位置資訊
    dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    # sin 和 cos 交替排列，最終形狀: [batch, 256, height, width]
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)

    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# 1/4  Vision Encoder — 影像編碼器
# ---------------------------------------------------------------------------

class VisionEncoderWrapper(nn.Module):
    """
    影像編碼器：ViT 骨幹 + FPN 頸部網路

    作用：
      輸入一張 RGB 圖片，輸出 3 層不同解析度的特徵圖 (Feature Pyramid)。
      - fpn_feat_0: [B, 256, 288, 288]  高解析度，用於精細遮罩
      - fpn_feat_1: [B, 256, 144, 144]  中解析度
      - fpn_feat_2: [B, 256,  72,  72]  低解析度，用於 Geometry Encoder 和 Decoder
      - fpn_pos_2:  [B, 256,  72,  72]  fpn_feat_2 的位置編碼

    架構簡述：
      ViT (Vision Transformer) 將圖片切成 14x14 的小塊 (patch)，
      經過 32 層 Transformer 處理後，再透過 FPN 產生多尺度特徵。
      位置嵌入在初始化時預先計算好，避免推論時重複運算。
    """

    def __init__(self, sam3_model: Sam3Model, device="cpu", image_size=1008):
        super().__init__()

        backbone = sam3_model.vision_encoder.backbone

        # ViT 的核心元件
        self.patch_embeddings = backbone.embeddings.patch_embeddings  # 將圖片切塊並投影
        self.dropout = backbone.embeddings.dropout
        self.layer_norm = backbone.layer_norm
        self.layers = backbone.layers                                 # 32 層 Transformer

        # FPN (Feature Pyramid Network)：將單一尺度特徵轉換為多尺度
        self.neck = sam3_model.vision_encoder.neck

        patch_size = backbone.config.patch_size    # 14 — 每個 patch 的像素大小
        self.height_patches = image_size // patch_size  # 1008/14 = 72
        self.width_patches = image_size // patch_size   # 1008/14 = 72
        hidden_size = backbone.config.hidden_size       # 1024 — ViT 的隱藏維度

        # --- 預先計算 ViT 位置嵌入 ---
        # 原始位置嵌入是 24x24（訓練時的 patch 數），需要 tile 到 72x72
        orig_pos_embed = backbone.embeddings.position_embeddings.data  # [1, 576, 1024]
        pretrain_size = int(orig_pos_embed.shape[1] ** 0.5)           # 24

        pos_embed = orig_pos_embed.reshape(
            1, pretrain_size, pretrain_size, hidden_size
        ).permute(0, 3, 1, 2)                                        # [1, 1024, 24, 24]

        # tile: 將 24x24 重複到足夠大，再裁切到 72x72
        repeat_h = self.height_patches // pretrain_size + 1  # 4
        repeat_w = self.width_patches // pretrain_size + 1   # 4
        pos_embed = pos_embed.tile([1, 1, repeat_h, repeat_w])[
            :, :, :self.height_patches, :self.width_patches
        ]                                                    # [1, 1024, 72, 72]

        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(
            1, self.height_patches * self.width_patches, hidden_size
        )                                                    # [1, 5184, 1024]
        self.register_buffer("vit_pos_embed", pos_embed.to(device))

        # --- 預先計算 FPN 第 2 層的正弦位置編碼 ---
        # 只有第 2 層需要，因為 Geometry Encoder 和 Decoder 都用這一層
        num_pos_feats = sam3_model.vision_encoder.neck.config.fpn_hidden_size // 2  # 128
        pos_enc_2 = compute_sine_position_encoding(
            shape=(1, 256, self.height_patches, self.width_patches),
            device=device, dtype=torch.float32, num_pos_feats=num_pos_feats,
        )
        self.register_buffer("pos_enc_2", pos_enc_2)

    def forward(self, images: torch.Tensor):
        """
        輸入: images [B, 3, 1008, 1008] — 正規化後的 RGB 圖片
        輸出: 4 個張量 — 3 層 FPN 特徵 + 位置編碼
        """
        batch_size = images.shape[0]

        # Patch Embedding: 圖片 → patch 序列
        embeddings = self.patch_embeddings(images)    # [B, 5184, 1024]
        embeddings = embeddings + self.vit_pos_embed  # 加上位置資訊
        embeddings = self.dropout(embeddings)

        # ViT Transformer: 32 層自注意力處理
        hidden_states = embeddings.view(
            batch_size, self.height_patches, self.width_patches, -1
        )
        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # 轉為 [B, C, H, W] 空間格式，送入 FPN
        hidden_states = hidden_states.permute(0, 3, 1, 2)  # [B, 1024, 72, 72]

        fpn_hidden_states, _ = self.neck(hidden_states)

        return (
            fpn_hidden_states[0],                            # [B, 256, 288, 288]
            fpn_hidden_states[1],                            # [B, 256, 144, 144]
            fpn_hidden_states[2],                            # [B, 256,  72,  72]
            self.pos_enc_2.expand(batch_size, -1, -1, -1),  # [B, 256,  72,  72]
        )


# ---------------------------------------------------------------------------
# 2/4  Text Encoder — 文字編碼器
# ---------------------------------------------------------------------------

class TextEncoderWrapper(nn.Module):
    """
    文字編碼器：CLIP Text Encoder + 投影層

    作用：
      將文字 token 序列轉為 256 維特徵向量。
      例如 "person" 這個詞會被 tokenizer 轉成數字序列 (input_ids)，
      再經過 CLIP 文字編碼器產生語意特徵。

    輸入:
      input_ids      [B, 32]  — tokenizer 產生的 token ID 序列（固定長度 32）
      attention_mask [B, 32]  — 標記哪些位置是真實 token（1）、哪些是填充（0）

    輸出:
      text_features  [B, 32, 256]  — 每個 token 的 256 維特徵
      text_mask      [B, 32]       — 布林遮罩，True = 有效 token
    """

    def __init__(self, sam3_model: Sam3Model):
        super().__init__()
        self.text_encoder = sam3_model.text_encoder
        self.text_projection = sam3_model.text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        text_features = self.text_projection(text_features)  # 投影到 256 維
        text_mask = attention_mask > 0
        return text_features, text_mask


# ---------------------------------------------------------------------------
# 3/4  Geometry Encoder — 幾何編碼器
# ---------------------------------------------------------------------------

class GeometryEncoderWrapper(nn.Module):
    """
    幾何編碼器：將參考圖片中的框選區域轉為特徵向量

    作用：
      當你用「圖片 prompt」時（例如框出一張杯子的照片），
      這個編碼器會：
        1. 用 ROI Align 從 FPN 特徵中裁切出框選區域
        2. 加上位置編碼（告訴模型框在圖片中的位置）
        3. 加上標籤嵌入（正樣本 vs 負樣本）
        4. 經過 3 層 cross-attention 融合視覺特徵
        5. 最後加上一個 CLS token（代表整體語意）

    輸入:
      input_boxes        [B, num_boxes, 4]  — 正規化 cxcywh 格式的框
      input_boxes_labels [B, num_boxes]     — 標籤 (1=正, 0=負, -10=填充)
      fpn_feat_2         [B, 256, 72, 72]   — Vision Encoder 的第 2 層特徵
      fpn_pos_2          [B, 256, 72, 72]   — 對應的位置編碼

    輸出:
      geometry_features  [B, num_boxes+1, 256]  — 幾何特徵（+1 是 CLS token）
      geometry_mask      [B, num_boxes+1]       — 布林遮罩
    """

    def __init__(self, sam3_model: Sam3Model):
        super().__init__()
        self.geometry_encoder = sam3_model.geometry_encoder
        self.hidden_size = self.geometry_encoder.hidden_size  # 256
        self.roi_size = self.geometry_encoder.roi_size        # 7（ROI Align 裁切後的尺寸）

    def forward(
        self,
        input_boxes: torch.Tensor,
        input_boxes_labels: torch.Tensor,
        fpn_feat: torch.Tensor,
        fpn_pos: torch.Tensor,
    ):
        batch_size, num_boxes = input_boxes.shape[:2]
        device = input_boxes.device
        ge = self.geometry_encoder

        # --- 處理填充框 ---
        # -10 表示填充（不是真實的框），需要在 mask 中標記為 False
        box_mask = input_boxes_labels != -10
        box_labels = torch.where(
            input_boxes_labels == -10,
            torch.zeros_like(input_boxes_labels),
            input_boxes_labels,
        )

        # 將 FPN 特徵攤平成序列格式，供 cross-attention 使用
        vision_feats_flat = fpn_feat.flatten(2).transpose(1, 2)      # [B, 72*72, 256]
        vision_pos_embeds_flat = fpn_pos.flatten(2).transpose(1, 2)  # [B, 72*72, 256]

        # --- ROI Align ---
        # 從 FPN 特徵圖中，根據框的座標裁切出 7x7 的區域特徵
        boxes_embed = self._roi_align_boxes(
            ge, input_boxes, fpn_feat, batch_size, num_boxes, device,
        )

        # --- 位置編碼 ---
        # 將框的中心座標和寬高編碼成位置特徵
        center_x, center_y = input_boxes[:, :, 0], input_boxes[:, :, 1]
        box_width, box_height = input_boxes[:, :, 2], input_boxes[:, :, 3]
        pos_enc = ge._encode_box_coordinates(
            center_x.flatten(), center_y.flatten(),
            box_width.flatten(), box_height.flatten(),
        )
        pos_enc = pos_enc.view(batch_size, num_boxes, pos_enc.shape[-1])
        boxes_embed = boxes_embed + ge.boxes_pos_enc_project(pos_enc)

        # --- 標籤嵌入 + CLS token ---
        # 標籤嵌入告訴模型這個框是正樣本（要找的）還是負樣本（不要的）
        label_embed = ge.label_embed(box_labels.long())
        prompt_embeds = label_embed + boxes_embed
        prompt_mask = box_mask

        # CLS token: 代表這組框的整體語意，類似 BERT 的 [CLS]
        cls_embed = ge.cls_embed.weight.view(1, 1, self.hidden_size).expand(
            batch_size, -1, -1
        )
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        prompt_embeds = torch.cat([prompt_embeds, cls_embed], dim=1)
        prompt_mask = torch.cat([prompt_mask, cls_mask], dim=1)

        prompt_embeds = ge.prompt_layer_norm(ge.final_proj(prompt_embeds))

        # --- Cross-Attention ---
        # 3 層 cross-attention：讓框特徵「看到」FPN 視覺特徵，學習更好的表示
        prompt_attention_mask = create_bidirectional_mask(
            config=ge.config, input_embeds=prompt_embeds, attention_mask=prompt_mask
        )
        for layer in ge.layers:
            prompt_embeds = layer(
                prompt_feats=prompt_embeds,
                vision_feats=vision_feats_flat,
                vision_pos_encoding=vision_pos_embeds_flat,
                prompt_mask=prompt_attention_mask,
            )

        return ge.output_layer_norm(prompt_embeds), prompt_mask

    def _roi_align_boxes(
        self,
        ge,
        input_boxes: torch.Tensor,
        fpn_feat: torch.Tensor,
        batch_size: int,
        num_boxes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        用 ROI Align 從 FPN 特徵圖中裁切框內區域，回傳框嵌入向量。

        ROI Align 是 Mask R-CNN 提出的技術：
        根據框座標從特徵圖中精確裁切出固定大小 (7x7) 的區域，
        再經過卷積壓成一個向量。
        """
        img_feats_last = fpn_feat.permute(0, 2, 3, 1)
        normalized_img_feats = ge.vision_layer_norm(img_feats_last).permute(0, 3, 1, 2)

        height, width = normalized_img_feats.shape[-2:]

        # 框座標的直接線性投影
        boxes_embed = ge.boxes_direct_project(input_boxes)

        # cxcywh → xyxy，然後乘以特徵圖尺寸得到實際像素座標
        boxes_xyxy = box_cxcywh_to_xyxy(input_boxes)
        scale = torch.tensor(
            [width, height, width, height], dtype=boxes_xyxy.dtype, device=device
        )
        boxes_xyxy = boxes_xyxy * scale.view(1, 1, 4)

        # ROI Align 需要 [batch_index, x1, y1, x2, y2] 格式
        batch_indices = (
            torch.arange(batch_size, device=device)
            .view(-1, 1).expand(-1, num_boxes).reshape(-1, 1).float()
        )
        boxes_with_batch = torch.cat([batch_indices, boxes_xyxy.view(-1, 4)], dim=1)

        # bfloat16 不被 ROI Align 支援，需要轉成 float16
        dtype = (
            torch.float16
            if normalized_img_feats.dtype == torch.bfloat16
            else normalized_img_feats.dtype
        )
        sampled_features = torchvision.ops.roi_align(
            normalized_img_feats.to(dtype), boxes_with_batch.to(dtype), self.roi_size
        ).to(normalized_img_feats.dtype)

        # ROI 特徵經過 7x7 卷積壓成一個向量，加到框嵌入上
        pooled_projection = ge.boxes_pool_project(sampled_features).view(
            batch_size, num_boxes, self.hidden_size
        )
        return boxes_embed + pooled_projection


# ---------------------------------------------------------------------------
# 4/4  Decoder — 解碼器
# ---------------------------------------------------------------------------

class DecoderWrapper(nn.Module):
    """
    解碼器：DETR Encoder + DETR Decoder + Mask Decoder

    這是最核心的模組，負責產生最終的偵測結果。整體流程：

      1. DETR Encoder (6 層)
         - 將 FPN 視覺特徵和 prompt 特徵（文字/幾何）做 cross-attention
         - 讓視覺特徵「知道」我們要找什麼

      2. DETR Decoder (6 層)
         - 用 200 個可學習的 query（候選框）去「詢問」編碼後的特徵
         - 每個 query 會對齊到圖片中的一個區域，輸出一個偵測候選

      3. Top-K 篩選（可選）
         - 如果 num_queries < 200，從 200 個候選中依分數選出前 K 個
         - 只對前 K 個產生遮罩 → 大幅節省 VRAM 和運算量
         - DETR 內部仍用完整 200 queries，不影響偵測品質

      4. 輸出頭
         - box_head: 預測偵測框 [B, K, 4] (xyxy 格式)
         - dot_product_scoring: 計算匹配分數 [B, K]
         - presence_logits: 判斷「畫面中是否存在目標物」 [B, 1]
         - mask_decoder: 產生像素級遮罩 [B, K, H, W]

    輸入:
      fpn_feat_0       [B, 256, 288, 288]  — 高解析度 FPN 特徵
      fpn_feat_1       [B, 256, 144, 144]  — 中解析度 FPN 特徵
      fpn_feat_2       [B, 256,  72,  72]  — 低解析度 FPN 特徵
      fpn_pos_2        [B, 256,  72,  72]  — 位置編碼
      prompt_features  [B, prompt_len, 256] — 文字或幾何 prompt 特徵
      prompt_mask      [B, prompt_len]      — prompt 的有效位遮罩

    輸出:
      pred_masks       [B, K, H, W]  — K 個候選遮罩（K = num_queries）
      pred_boxes       [B, K, 4]     — K 個偵測框 (xyxy)
      pred_logits      [B, K]        — K 個匹配分數
      presence_logits  [B, 1]        — 目標是否存在的分數

    參數:
      num_queries: 輸出的候選數量（預設 200 = 全部，可設為 50/100 以節省 VRAM）
    """

    def __init__(self, sam3_model: Sam3Model, num_queries: int = 200):
        super().__init__()
        self.detr_encoder = sam3_model.detr_encoder      # 6 層 encoder
        self.detr_decoder = sam3_model.detr_decoder      # 6 層 decoder + 200 queries
        self.mask_decoder = sam3_model.mask_decoder       # 像素級遮罩生成
        self.dot_product_scoring = sam3_model.dot_product_scoring  # prompt 匹配分數
        self.box_head = sam3_model.detr_decoder.box_head  # 偵測框回歸

        # 模型內部固定 200 queries，num_queries 控制輸出數量
        self.model_queries = sam3_model.config.detr_decoder_config.num_queries  # 200
        self.num_queries = min(num_queries, self.model_queries)
        self.use_topk = self.num_queries < self.model_queries

    def forward(
        self,
        fpn_feat_0: torch.Tensor,
        fpn_feat_1: torch.Tensor,
        fpn_feat_2: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
    ):
        # --- DETR Encoder ---
        # 讓 FPN 視覺特徵和 prompt 特徵互相交換資訊
        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_feat_2],
            text_features=prompt_features,
            vision_pos_embeds=[fpn_pos_2],
            text_mask=prompt_mask,
        )

        # --- DETR Decoder ---
        # 200 個 query 去「詢問」編碼後的特徵，找出偵測目標
        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs.last_hidden_state,
            text_features=encoder_outputs.text_features,
            vision_pos_encoding=encoder_outputs.pos_embeds_flattened,
            text_mask=prompt_mask,
            spatial_shapes=encoder_outputs.spatial_shapes,
        )

        # --- 偵測框 ---
        # 每個 query 預測一個框的偏移量，加上參考點後取 sigmoid 歸一化
        all_box_offsets = self.box_head(decoder_outputs.intermediate_hidden_states)
        reference_boxes_inv_sig = self._inverse_sigmoid(decoder_outputs.reference_boxes)
        all_pred_boxes = box_cxcywh_to_xyxy(
            (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        )

        # --- 匹配分數 ---
        # 用 dot product 計算每個 query 與 prompt 的相似度
        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs.intermediate_hidden_states,
            text_features=encoder_outputs.text_features,
            text_mask=prompt_mask,
        ).squeeze(-1)

        # 取最後一層的輸出（最精確）
        pred_logits = all_pred_logits[-1]           # [B, 200]
        pred_boxes = all_pred_boxes[-1]             # [B, 200, 4]
        decoder_hidden_states = decoder_outputs.intermediate_hidden_states[-1]
        presence_logits = decoder_outputs.presence_logits[-1]  # [B, 1]

        # --- Top-K 篩選（可選） ---
        # 從 200 個候選中選出分數最高的前 K 個，只對這些產生遮罩
        # DETR 內部仍使用完整 200 queries，偵測品質不受影響
        if self.use_topk:
            K = self.num_queries
            _, topk_indices = pred_logits.topk(K, dim=-1)           # [B, K]
            topk_indices, _ = topk_indices.sort(dim=-1)             # 保持原始順序

            # 用 gather 從 200 中挑出 top-K
            idx_boxes = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
            pred_boxes = pred_boxes.gather(1, idx_boxes)            # [B, K, 4]
            pred_logits = pred_logits.gather(1, topk_indices)       # [B, K]

            idx_hidden = topk_indices.unsqueeze(-1).expand(-1, -1, decoder_hidden_states.shape[-1])
            decoder_hidden_states = decoder_hidden_states.gather(1, idx_hidden)  # [B, K, 256]

        # --- Mask Decoder ---
        # 從 decoder 的 query 特徵生成像素級遮罩（只處理 K 個）
        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=[fpn_feat_0, fpn_feat_1, fpn_feat_2],
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            prompt_features=prompt_features,
            prompt_mask=prompt_mask,
        )

        return mask_outputs.pred_masks, pred_boxes, pred_logits, presence_logits

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """sigmoid 的反函數，用於將參考點座標轉回 logit 空間"""
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


# ---------------------------------------------------------------------------
# 匯出函式
# ---------------------------------------------------------------------------

def export_vision_encoder(
    model: Sam3Model, output_dir: Path,
    device: str, opset_version: int, image_size: int,
):
    """匯出 Vision Encoder 為 ONNX（動態 batch 維度）"""
    print("匯出 Vision Encoder...")
    wrapper = VisionEncoderWrapper(model, device=device, image_size=image_size)
    wrapper = wrapper.to(device).eval()

    # 用假資料跑一次 forward，torch.onnx.export 會追蹤計算圖
    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, image_size, image_size, device=device),),
        str(output_dir / "vision-encoder.onnx"),
        input_names=["images"],
        output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
        # dynamic_axes: 讓 batch 維度可以變動（1 到 4 張圖同時處理）
        dynamic_axes={
            "images": {0: "batch"},
            "fpn_feat_0": {0: "batch"},
            "fpn_feat_1": {0: "batch"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
        },
    )
    print(f"  OK: {output_dir / 'vision-encoder.onnx'}")


def export_text_encoder(
    model: Sam3Model, output_dir: Path,
    device: str, opset_version: int,
):
    """匯出 Text Encoder 為 ONNX（動態 batch 維度）"""
    print("匯出 Text Encoder...")
    wrapper = TextEncoderWrapper(model).to(device).eval()

    # 32 是 CLIP tokenizer 的最大 token 長度（固定值，不可變動）
    seq_len = 32
    torch.onnx.export(
        wrapper,
        (
            torch.randint(0, 49408, (1, seq_len), device=device),
            torch.ones(1, seq_len, dtype=torch.long, device=device),
        ),
        str(output_dir / "text-encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features", "text_mask"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_features": {0: "batch"},
            "text_mask": {0: "batch"},
        },
    )
    print(f"  OK: {output_dir / 'text-encoder.onnx'}")


def export_geometry_encoder(
    model: Sam3Model, output_dir: Path,
    device: str, opset_version: int, image_size: int = 1008,
):
    """匯出 Geometry Encoder 為 ONNX（動態 batch + 動態框數量）"""
    patches = image_size // 14  # ViT patch size
    print(f"匯出 Geometry Encoder (fpn={patches}x{patches})...")
    wrapper = GeometryEncoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.rand(1, 5, 4, device=device),                          # 5 個框（示範用）
            torch.ones(1, 5, dtype=torch.long, device=device),           # 全部標記為正樣本
            torch.randn(1, 256, patches, patches, device=device),        # FPN 特徵
            torch.randn(1, 256, patches, patches, device=device),        # 位置編碼
        ),
        str(output_dir / "geometry-encoder.onnx"),
        input_names=["input_boxes", "input_boxes_labels", "fpn_feat_2", "fpn_pos_2"],
        output_names=["geometry_features", "geometry_mask"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
        # batch 和 num_boxes 都是動態的
        dynamic_axes={
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
            "geometry_features": {0: "batch", 1: "num_prompts"},
            "geometry_mask": {0: "batch", 1: "num_prompts"},
        },
    )
    print(f"  OK: {output_dir / 'geometry-encoder.onnx'}")


def export_decoder(
    model: Sam3Model, output_dir: Path,
    device: str, opset_version: int,
    num_queries: int = 200, image_size: int = 1008,
):
    """匯出 Decoder 為 ONNX（動態 batch + 動態 prompt 長度）

    Args:
        num_queries: 輸出候選數量。預設 200（全部）。
            設為較小的值（如 50）可以大幅節省 VRAM：
            - DETR 內部仍使用全部 200 queries（不影響偵測品質）
            - 在 mask decoder 之前做 top-K 篩選
            - mask decoder 只處理 K 個候選 → output [B, K, fpn0, fpn0]
            - VRAM 節省與 K 成正比：200→50 = mask buffer 減 75%
        image_size: 輸入圖片解析度（影響 FPN 特徵圖大小）。
            patches = image_size / 14，FPN 尺寸 = patches*4, patches*2, patches。
    """
    Q = num_queries
    patches = image_size // 14
    fpn0 = patches * 4   # e.g. 288 for 1008, 240 for 840
    fpn1 = patches * 2   # e.g. 144 for 1008, 120 for 840
    fpn2 = patches        # e.g. 72 for 1008, 60 for 840
    print(f"匯出 Decoder (num_queries={Q}, fpn={fpn0}/{fpn1}/{fpn2})...")
    wrapper = DecoderWrapper(model, num_queries=Q).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.randn(1, 256, fpn0, fpn0, device=device),  # fpn_feat_0
            torch.randn(1, 256, fpn1, fpn1, device=device),  # fpn_feat_1
            torch.randn(1, 256, fpn2, fpn2, device=device),  # fpn_feat_2
            torch.randn(1, 256, fpn2, fpn2, device=device),  # fpn_pos_2
            torch.randn(1,  32, 256, device=device),        # prompt_features
            torch.ones(1, 32, dtype=torch.bool, device=device),  # prompt_mask
        ),
        str(output_dir / "decoder.onnx"),
        input_names=[
            "fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2",
            "prompt_features", "prompt_mask",
        ],
        output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
        # prompt_len 是動態的（文字 prompt = 32，幾何 prompt = 框數*2+1，組合 = 兩者相加）
        dynamic_axes={
            **{f"fpn_feat_{i}": {0: "batch"} for i in range(3)},
            "fpn_pos_2": {0: "batch"},
            "prompt_features": {0: "batch", 1: "prompt_len"},
            "prompt_mask": {0: "batch", 1: "prompt_len"},
            "pred_masks": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "pred_logits": {0: "batch"},
            "presence_logits": {0: "batch"},
        },
    )
    print(f"  OK: {output_dir / 'decoder.onnx'} (queries={Q})")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="將 SAM3 模型從 PyTorch 匯出為 ONNX 格式",
    )
    parser.add_argument(
        "--module", choices=["vision", "text", "geometry", "decoder"],
        help="只匯出指定的子模型 (vision / text / geometry / decoder)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="匯出全部 4 個子模型",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="SAM3 模型路徑（HuggingFace ID 或本機路徑，例如 facebook/sam3）",
    )
    parser.add_argument("--image-size", type=int, default=1008, help="輸入圖片尺寸")
    parser.add_argument("--output-dir", default="onnx", help="ONNX 輸出資料夾")
    parser.add_argument("--device", default="cpu", help="運算裝置 (cpu / cuda)")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument(
        "--num-queries", type=int, default=200,
        help="Decoder 輸出候選數量（預設 200）。"
             "設為 50 或 100 可大幅節省 VRAM，"
             "DETR 內部仍用 200 queries 不影響偵測品質",
    )
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.error("請指定 --module 或 --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"載入 SAM3 模型: {args.model_path} ...")
    model = Sam3Model.from_pretrained(args.model_path).to(args.device).eval()
    print("  OK")

    # 如果目標解析度與模型預設不同，修補 global attention 層的 RoPE
    patch_rope_for_resolution(model, args.image_size)
    print()

    # 每個匯出函式只接收自己需要的參數
    common = dict(model=model, output_dir=output_dir,
                  device=args.device, opset_version=args.opset_version)

    exporters = {
        "vision":   lambda: export_vision_encoder(**common, image_size=args.image_size),
        "text":     lambda: export_text_encoder(**common),
        "geometry": lambda: export_geometry_encoder(**common, image_size=args.image_size),
        "decoder":  lambda: export_decoder(**common, num_queries=args.num_queries, image_size=args.image_size),
    }

    modules = list(exporters.keys()) if args.all else [args.module]

    with torch.no_grad():
        for m in modules:
            exporters[m]()

    print(f"\n匯出完成！ONNX 模型儲存於: {output_dir}")


if __name__ == "__main__":
    main()
