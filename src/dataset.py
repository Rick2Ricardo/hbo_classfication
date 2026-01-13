import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer

class BrainTextDataset(Dataset):
    def __init__(self, config, file_list, split='train', indices=None):
        """
        file_list: list, 包含所有要读取的 xlsx 文件路径
        indices: list, 指定使用合并后的数据的哪些行 (用于防泄漏切分)
        """
        self.cfg = config
        self.hbo_list = [] # 存放每个文件的脑电矩阵 [Array1, Array2, ...]
        
        all_utt_dfs = []   # 存放每个文件的文本DataFrame
        
        print(f"[{split.upper()}] 正在加载 {len(file_list)} 个文件...")
        
        # === 1. 循环读取所有文件 ===
        for f_idx, f_path in enumerate(file_list):
            try:
                xls = pd.ExcelFile(f_path)
                # 读取 文本 和 脑电
                utt_df = pd.read_excel(xls, sheet_name=config.SHEET_UTT)
                hbo_df = pd.read_excel(xls, sheet_name=config.SHEET_HBO)
                
                # 处理脑电 (存入列表，不合并)
                hbo_data = hbo_df.iloc[:, :config.N_CHANNELS].values.astype(np.float32)
                self.hbo_list.append(hbo_data)
                
                # 处理文本 (给每一行加一个标记：它来自第几个文件)
                # 这样我们在 __getitem__ 时就知道去 hbo_list 的第几个位置找脑电了
                utt_df['file_source_idx'] = f_idx
                all_utt_dfs.append(utt_df)
                
            except Exception as e:
                print(f"警告: 文件 {f_path} 读取失败，已跳过。错误: {e}")

        # 合并所有对话数据
        if len(all_utt_dfs) == 0:
            raise ValueError("没有读取到任何有效数据！请检查路径。")
            
        self.combined_utt_df = pd.concat(all_utt_dfs, ignore_index=True)

        # === 2. 物理隔离切分 (根据传入的 indices 过滤) ===
        if indices is not None:
            self.combined_utt_df = self.combined_utt_df.iloc[indices].reset_index(drop=True)
            print(f"[{split.upper()}] 切分后保留样本数: {len(self.combined_utt_df)}")

        # === 3. 稀有样本增强 (只在训练集) ===
        if split == 'train':
            print(f"检测稀有样本并增强 (Oversampling)...")
            rare_keywords = ['迁移', '拓展', '分析', '比较']
            rare_indices = []

            for idx, row in self.combined_utt_df.iterrows():
                label_str = str(row['label'])
                if any(k in label_str for k in rare_keywords):
                    rare_indices.append(idx)
            
            if len(rare_indices) > 0:
                duplicate_rows = self.combined_utt_df.iloc[rare_indices]
                # 复制 5 倍
                for _ in range(5): 
                    self.combined_utt_df = pd.concat([self.combined_utt_df, duplicate_rows], ignore_index=True)
                print(f"-> 增强后训练集大小: {len(self.combined_utt_df)}")

        # 4. 初始化 Tokenizer 和 标签表
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.label_map = {
            "基础知识类提问": 0, "基础知识类回应": 1,
            "个人观点类提问": 2, "个人观点类回应": 3,
            "比较归纳式提问": 4, "比较归纳式回应": 5,
            "分析阐释式提问": 6, "分析阐释式回应": 7,
            "迁移创新式提问": 8, "迁移创新式回应": 9,
            "拓展建构式提问": 10, "拓展建构式回应": 11
        }

    def __len__(self):
        return len(self.combined_utt_df)

    # def __getitem__(self, idx):
    #     row = self.combined_utt_df.iloc[idx]
        
    #     # === A. 脑电数据处理 (支持多文件) ===
    #     # 1. 先看这条数据来自哪个文件
    #     file_idx = int(row['file_source_idx'])
        
    #     # 2. 取出对应文件的脑电大矩阵
    #     target_hbo_matrix = self.hbo_list[file_idx] 
    #     total_rows = target_hbo_matrix.shape[0]
        
    #     # 3. 再根据行号切片
    #     s_row = max(0, int(row['startRow']) - 1)
    #     e_row = min(total_rows, int(row['endRow']))
        
    #     if s_row >= e_row:
    #         brain_seq = np.zeros((self.cfg.MAX_SEQ_LEN_BRAIN, self.cfg.N_CHANNELS), dtype=np.float32)
    #     else:
    #         brain_seq = target_hbo_matrix[s_row:e_row, :]
            
    #         # Padding / Truncation
    #         curr_len = brain_seq.shape[0]
    #         target_len = self.cfg.MAX_SEQ_LEN_BRAIN
    #         if curr_len > target_len:
    #             brain_seq = brain_seq[:target_len, :]
    #         elif curr_len < target_len:
    #             padding = np.zeros((target_len - curr_len, self.cfg.N_CHANNELS), dtype=np.float32)
    #             brain_seq = np.vstack([brain_seq, padding])

    #     # === B. 文本处理 ===
    #     text = str(row['content'])
    #     encoding = self.tokenizer(text, max_length=self.cfg.MAX_SEQ_LEN_TEXT, padding='max_length', truncation=True, return_tensors='pt')
        
    #     # === C. 标签处理 ===
    #     label_vec = torch.zeros(self.cfg.NUM_CLASSES)
    #     label_str = str(row['label'])
    #     if label_str != 'nan':
    #         clean_str = label_str.replace('"', '').replace('，', ',')
    #         tags = clean_str.split(',')
    #         for tag in tags:
    #             tag = tag.strip()
    #             if tag in self.label_map:
    #                 label_vec[self.label_map[tag]] = 1.0
                    
    #     return {
    #         'brain_x': torch.FloatTensor(brain_seq),
    #         'input_ids': encoding['input_ids'].flatten(),
    #         'attention_mask': encoding['attention_mask'].flatten(),
    #         'labels': label_vec
    #     }
    
def __getitem__(self, idx):
        row = self.combined_utt_df.iloc[idx]
        
        # === A. 脑电数据处理 (加上关键的 Time Shift) ===
        file_idx = int(row['file_source_idx'])
        target_hbo_matrix = self.hbo_list[file_idx] 
        total_rows = target_hbo_matrix.shape[0]
        
        # --- 核心修改 1：加上延迟偏移 (Hemodynamic Lag) ---
        # 假设采样率约 10Hz，一般延迟 4-6秒，我们取 40 行的偏移量
        HRF_DELAY = 40 
        
        # 原来的索引加上偏移量
        s_row = max(0, int(row['startRow']) - 1 + HRF_DELAY)
        e_row = min(total_rows, int(row['endRow']) + HRF_DELAY)
        
        if s_row >= e_row:
            brain_seq = np.zeros((self.cfg.MAX_SEQ_LEN_BRAIN, self.cfg.N_CHANNELS), dtype=np.float32)
        else:
            brain_seq = target_hbo_matrix[s_row:e_row, :]
            
            # Padding / Truncation (保持不变)
            curr_len = brain_seq.shape[0]
            target_len = self.cfg.MAX_SEQ_LEN_BRAIN
            if curr_len > target_len:
                brain_seq = brain_seq[:target_len, :]
            elif curr_len < target_len:
                padding = np.zeros((target_len - curr_len, self.cfg.N_CHANNELS), dtype=np.float32)
                brain_seq = np.vstack([brain_seq, padding])

        # === B. 文本处理 (加上上下文 Context) ===
        current_text = str(row['content'])
        full_text = current_text
        
        # --- 核心修改 2：拼接上一句 (如果是同一个文件且不是第一行) ---
        # 我们可以简单的判断：如果 index > 0 且 file_source_idx 相同，就把上一行拼进来
        if idx > 0:
            prev_row = self.combined_utt_df.iloc[idx-1]
            if prev_row['file_source_idx'] == row['file_source_idx']:
                # 格式：[上句角色] 上句内容 [SEP] [本句角色] 本句内容
                # 例如："[助教] 为什么? [SEP] [学生] 因为..."
                full_text = f"[{prev_row['role']}] {prev_row['content']} [SEP] [{row['role']}] {current_text}"
            else:
                # 如果是文件开头第一句，只加自己的角色
                full_text = f"[{row['role']}] {current_text}"
        else:
            full_text = f"[{row['role']}] {current_text}"

        # 这里的 max_length 可能要稍微调大一点，比如从 64 改到 128，因为拼了两次话
        encoding = self.tokenizer(full_text, max_length=self.cfg.MAX_SEQ_LEN_TEXT, padding='max_length', truncation=True, return_tensors='pt')
        
        # ... (标签处理保持不变) ...
                
        # === C. 标签处理 ===
        label_vec = torch.zeros(self.cfg.NUM_CLASSES)
        label_str = str(row['label'])
        if label_str != 'nan':
            clean_str = label_str.replace('"', '').replace('，', ',')
            tags = clean_str.split(',')
            for tag in tags:
                tag = tag.strip()
                if tag in self.label_map:
                    label_vec[self.label_map[tag]] = 1.0
                    
        return {
            'brain_x': torch.FloatTensor(brain_seq),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vec
        }