import torch
import torch.nn as nn
from transformers import BertModel
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ClassroomInteractionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # === 1. 左塔: Brain Transformer ===
        # 先把 44 维特征映射到 Transformer 喜欢的维度 (如 128)
        self.brain_embedding = nn.Linear(config.N_CHANNELS, config.BRAIN_HIDDEN_DIM)
        self.pos_encoder = PositionalEncoding(config.BRAIN_HIDDEN_DIM, config.MAX_SEQ_LEN_BRAIN)
        
        # Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.BRAIN_HIDDEN_DIM, 
            nhead=4, 
            dim_feedforward=512,
            batch_first=True
        )
        self.brain_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # === 2. 右塔: Text Transformer (BERT) ===
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        # === 3. 融合分类层 ===
        fusion_input_dim = config.BRAIN_HIDDEN_DIM + config.TEXT_HIDDEN_DIM
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.FUSION_HIDDEN_DIM, config.NUM_CLASSES)
            # 注意：不加 Sigmoid，Loss 函数里会加
        )

    def forward(self, brain_x, input_ids, attention_mask):
        # --- Brain Path ---
        b_emb = self.brain_embedding(brain_x)     # [B, T, 44] -> [B, T, 128]
        b_emb = self.pos_encoder(b_emb)           # 加上位置信息
        b_out = self.brain_transformer(b_emb)     # [B, T, 128]
        b_feat = b_out.mean(dim=1)                # Global Pooling -> [B, 128]

        # --- Text Path ---
        t_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t_feat = t_out.pooler_output              # [B, 768]

        # --- Fusion ---
        combined = torch.cat((b_feat, t_feat), dim=1) # [B, 128+768]
        logits = self.fusion_head(combined)           # [B, 12]

        return logits