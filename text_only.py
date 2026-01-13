import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import argparse
import glob
import os
from transformers import AutoModel  # 新增：用于加载BERT

from src.config import Config
from src.dataset import BrainTextDataset
# from src.model import ClassroomInteractionNet # 注释掉原来的模型，或者留着不调用

# ==========================================
# 1. 新增：定义一个纯文本模型 (TextOnlyNet)
# ==========================================
# ... (前面的引用保持不变)

class TextOnlyNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # === 修改开始 ===
        # 尝试从 cfg 获取路径，如果没有，就默认用在线的 'bert-base-chinese'
        model_path = getattr(cfg, 'BERT_PATH', 'bert-base-chinese')
        print(f"Loading BERT from: {model_path}")
        
        self.bert = AutoModel.from_pretrained(model_path)
        # === 修改结束 ===
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, cfg.NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# ... (后面的代码保持不变)

# --- 参数解析函数 ---
def get_args():
    parser = argparse.ArgumentParser(description="Classroom NeuroAI Text-Only Training")
    parser.add_argument('--data_file', type=str, default=None, 
                        help='指定单个xlsx文件的路径')
    return parser.parse_args()

# ==========================================
# 2. 修改：evaluate 函数 (去掉 b_x)
# ==========================================
def evaluate(model, dataloader, cfg):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # b_x = batch['brain_x'].to(cfg.DEVICE) # 不需要脑电数据了
            i_ids = batch['input_ids'].to(cfg.DEVICE)
            mask = batch['attention_mask'].to(cfg.DEVICE)
            labels = batch['labels'].cpu().numpy()

            # 修改：只传文本参数
            logits = model(i_ids, mask) 
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # 阈值判定 (可以根据上一次实验调整，比如 0.5)
            preds = (probs > 0.5).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
    else:
        return 0.0, 0.0, 0.0, 0.0
    
    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    prec = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return f1, acc, rec, prec

def train():
    args = get_args()
    cfg = Config()
    
    # === 确定要读取的文件列表 (保持不变) ===
    target_files = []
    if args.data_file:
        if os.path.exists(args.data_file):
            target_files = [args.data_file]
        else:
            raise FileNotFoundError(f"找不到文件: {args.data_file}")
    else:
        search_path = os.path.join('./data/', '*.xlsx')
        target_files = glob.glob(search_path)

    if len(target_files) == 0:
        print("错误: 没有找到任何 xlsx 文件！")
        return

    # === 预扫描总长度 (保持不变) ===
    print(f"\n正在预读取所有文件以生成索引...")
    total_len = 0
    for f_path in target_files:
        try:
            df = pd.read_excel(f_path, sheet_name=cfg.SHEET_UTT)
            total_len += len(df)
        except Exception:
            pass
            
    all_indices = np.arange(total_len)
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    train_size = int(0.8 * total_len)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    
    # === 数据集 (保持不变，Dataset 虽然会返回 brain_x，但我们在下面不用就行了) ===
    train_dataset = BrainTextDataset(cfg, file_list=target_files, split='train', indices=train_indices)
    val_dataset = BrainTextDataset(cfg, file_list=target_files, split='val', indices=val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # ==========================================
    # 3. 修改：实例化纯文本模型
    # ==========================================
    print("正在初始化纯文本模型 (TextOnlyNet)...")
    model = TextOnlyNet(cfg).to(cfg.DEVICE) # 使用新定义的类
    
    pos_weight = torch.ones([cfg.NUM_CLASSES]).to(cfg.DEVICE) * 8.0 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 注意：纯文本模型通常比多模态更容易过拟合，可以考虑稍微调小 learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    print(f"\n开始纯文本基准测试 (Device: {cfg.DEVICE})...")
    best_f1 = 0.0
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for batch in loop:
            # b_x = batch['brain_x'].to(cfg.DEVICE) # 不需要
            i_ids = batch['input_ids'].to(cfg.DEVICE)
            mask = batch['attention_mask'].to(cfg.DEVICE)
            labels = batch['labels'].to(cfg.DEVICE)

            optimizer.zero_grad()
            
            # 修改：只传文本参数
            logits = model(i_ids, mask)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = train_loss / len(train_loader)
        val_f1, val_acc, val_rec, val_prec = evaluate(model, val_loader, cfg)
        
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Acc: {val_acc:.4f} | Rec: {val_rec:.4f} | Prec: {val_prec:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # 建议换个名字保存，别覆盖了多模态的模型
            torch.save(model.state_dict(), "best_text_only_model.pth")
            print(f">>> 发现新高 F1: {best_f1:.4f}, 模型已保存为 best_text_only_model.pth")

if __name__ == '__main__':
    train()