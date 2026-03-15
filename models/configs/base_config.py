# 基于 nnU-Net V2 得到的最优参数配置，同时包含各模型原始论文中的通道数

class Config:
    # 随机种子，用于确保实验的可重复性
    SEED = 42
    
    # 数据参数
    BATCH_SIZE = 16  # 批次大小
    PATCH_SIZE = (448, 384)  # 补丁大小
    IMAGE_SIZE = (448, 384)  # 图像大小
    
    # 训练参数
    EPOCHS = 50  # 训练轮数
    INITIAL_LR = 0.0001  # 初始学习率
    
    # 网络参数
    NUM_CLASSES = 2  # 类别数（背景和前景）
    IN_CHANNELS = 1  # 输入通道数（灰度图像）
    
    # 数据预处理参数
    NORMALIZATION = 'ZScore'  # 归一化方法
    
    # 损失函数
    BATCH_DICE = True  # 是否在批次级别计算Dice损失
    
    # 路径
    DATA_DIR = 'd:/jiazhuangxian_us/tn3k'  # 数据目录
    TEST_IMAGE_DIR = 'test-image'  # 测试图像目录
    TEST_MASK_DIR = 'test-mask'  # 测试掩码目录
    RESULTS_DIR = 'results'  # 结果目录
    
    # 各模型原始论文中的通道数配置
    MODEL_FEATURES = {
        'MKUNet': [32, 64, 128, 256, 512],
        'nnUNet': [32, 64, 128, 256, 512, 512, 512],
    }