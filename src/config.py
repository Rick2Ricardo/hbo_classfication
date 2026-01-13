import torch

class Config:
    # === 路径配置 ===
    DATA_PATH = './data/S12_C31_黄子硕.xlsx'
    SHEET_HBO = 'hbo'
    SHEET_UTT = 'utterance'
    MODEL_SAVE_PATH = './best_model.pth'

    # === 数据参数 ===
    N_CHANNELS = 44
    MAX_SEQ_LEN_BRAIN = 100
    MAX_SEQ_LEN_TEXT = 64
    NUM_CLASSES = 12

    # === 模型参数 ===
    BRAIN_HIDDEN_DIM = 128
    TEXT_HIDDEN_DIM = 768
    FUSION_HIDDEN_DIM = 256

    # === 训练参数 (刚才缺失的部分) ===
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10

    # === 设备选择逻辑 (Mac MPS 支持) ===
    # 优先检查 CUDA (NVIDIA)，其次检查 MPS (Mac)，最后用 CPU
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'  # Mac M1/M2/M3 芯片加速
    else:
        DEVICE = 'cpu'