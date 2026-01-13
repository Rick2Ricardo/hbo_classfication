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

from src.config import Config
from src.dataset import BrainTextDataset
from src.model import ClassroomInteractionNet

# --- 参数解析函数 ---
def get_args():
    parser = argparse.ArgumentParser(description="Classroom NeuroAI Training")
    
    # 1. 数据路径参数：可以是文件夹，也可以是具体文件
    parser.add_argument('--data_path', type=str, default='./data/child', 
                        help='数据路径：可以是包含xlsx的文件夹路径，也可以是单个xlsx文件路径')
    
    # 2. 模型保存目录
    parser.add_argument('--save_dir', type=str, default='./model_save', 
                        help='模型保存的文件夹')
    
    # 3. 模型保存文件名
    parser.add_argument('--model_name', type=str, default='best_model.pth', 
                        help='保存的模型文件名')
    
    return parser.parse_args()

def evaluate(model, dataloader, cfg):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            b_x = batch['brain_x'].to(cfg.DEVICE)
            i_ids = batch['input_ids'].to(cfg.DEVICE)
            mask = batch['attention_mask'].to(cfg.DEVICE)
            labels = batch['labels'].cpu().numpy()

            logits = model(b_x, i_ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # 阈值判定
            preds = (probs > 0.5).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
    else:
        return 0.0, 0.0, 0.0, 0.0
    
    # === 计算各项指标 ===
    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    prec = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return f1, acc, rec, prec

def train():
    # 1. 获取参数和配置
    args = get_args()
    cfg = Config()
    
    # ====================================================
    # 2. 确定要读取的文件列表 (核心修改逻辑)
    # ====================================================
    target_files = []
    input_path = args.data_path

    if os.path.isfile(input_path):
        # 情况A: 用户指定的是一个具体的文件
        print(f"检测到输入为单个文件: {input_path}")
        target_files = [input_path]
    elif os.path.isdir(input_path):
        # 情况B: 用户指定的是一个文件夹，读取下面所有 xlsx
        print(f"检测到输入为文件夹: {input_path}")
        search_pattern = os.path.join(input_path, '*.xlsx')
        target_files = glob.glob(search_pattern)
        print(f"在该目录下找到 {len(target_files)} 个 xlsx 文件")
    else:
        raise FileNotFoundError(f"路径不存在或无效: {input_path}")

    if len(target_files) == 0:
        print(f"错误: 在路径 {input_path} 下未找到任何 .xlsx 文件！")
        return

    # ====================================================
    # 3. 确定模型保存路径 (核心修改逻辑)
    # ====================================================
    # 确保保存目录存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"创建模型保存目录: {args.save_dir}")
    
    # 组合完整的保存路径
    final_save_path = os.path.join(args.save_dir, args.model_name)
    print(f"训练最优模型将保存在: {final_save_path}")

    # ====================================================
    # 4. 预扫描总长度 (为了防泄漏切分)
    # ====================================================
    print(f"\n正在预读取所有文件以生成索引...")
    total_len = 0
    
    for f_path in target_files:
        try:
            df = pd.read_excel(f_path, sheet_name=cfg.SHEET_UTT)
            total_len += len(df)
        except Exception as e:
            print(f"警告: 跳过损坏文件 {f_path}: {e}")
            
    print(f"合并后样本总数: {total_len}")
    
    if total_len == 0:
        print("错误: 数据集总长度为0，请检查数据文件。")
        return

    # 生成索引并打乱
    all_indices = np.arange(total_len)
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    # 切分索引
    train_size = int(0.8 * total_len)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    
    print(f"物理隔离切分 -> 训练集: {len(train_indices)}, 验证集: {len(val_indices)}")
    
    # ====================================================
    # 5. 实例化数据集
    # ====================================================
    train_dataset = BrainTextDataset(cfg, file_list=target_files, split='train', indices=train_indices)
    val_dataset = BrainTextDataset(cfg, file_list=target_files, split='val', indices=val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # ====================================================
    # 6. 训练循环
    # ====================================================
    model = ClassroomInteractionNet(cfg).to(cfg.DEVICE)
    
    pos_weight = torch.ones([cfg.NUM_CLASSES]).to(cfg.DEVICE) * 8.0 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    print(f"\n开始训练 (Device: {cfg.DEVICE})...")
    best_f1 = 0.0
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for batch in loop:
            b_x = batch['brain_x'].to(cfg.DEVICE)
            i_ids = batch['input_ids'].to(cfg.DEVICE)
            mask = batch['attention_mask'].to(cfg.DEVICE)
            labels = batch['labels'].to(cfg.DEVICE)

            optimizer.zero_grad()
            logits = model(b_x, i_ids, mask)
            
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
            # 使用上面生成的 final_save_path 进行保存
            torch.save(model.state_dict(), final_save_path)
            print(f">>> 发现新高 F1: {best_f1:.4f}, 模型已保存至: {final_save_path}")

if __name__ == '__main__':
    train()