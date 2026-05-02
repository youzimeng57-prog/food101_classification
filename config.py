# config.py
import os

class Config:
    # 数据相关
    data_root = './data'          # Food-101数据保存根目录
    num_classes = 101
    img_size = 224                 # ResNet标准输入尺寸
    
    # 训练超参数
    batch_size = 64                # 根据8GB显存调整，若OOM可降至32或16
    epochs = 30                    # 建议30轮，可根据收敛情况早停
    learning_rate = 1e-3           # 分类头初始学习率
    backbone_lr = 1e-5             # 骨干网络学习率（分层）
    weight_decay = 1e-4
    label_smoothing = 0.1
    
    # 梯度累积（模拟更大batch）
    gradient_accumulation_steps = 2  # 实际batch = batch_size * accum_steps
    
    # 混合精度训练
    use_amp = True                  # 开启自动混合精度
    
    # 模型保存与日志
    save_dir = './checkpoints'
    log_interval = 20               # 每20个batch打印一次loss
    eval_interval = 1               # 每1个epoch验证一次
    
    # 数据加载
    num_workers = 4                 # 根据CPU核心数调整
    pin_memory = True
    
    # 随机种子
    seed = 42
    
    @staticmethod
    def setup_dirs():
        os.makedirs(Config.save_dir, exist_ok=True)
        os.makedirs('./logs', exist_ok=True)