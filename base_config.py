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
        # UNet: Original paper uses [64, 128, 256, 512]
        'UNet': [64, 128, 256, 512],  # UNet的特征通道配置
        # Attention UNet: Based on original UNet with attention gates, uses [64, 128, 256, 512]
        'AttentionUNet': [64, 128, 256, 512],  # Attention UNet的特征通道配置
        # UNet++: Original paper uses [32, 64, 128, 256, 512]
        'UNetPlusPlus': [32, 64, 128, 256, 512],  # UNet++的特征通道配置
        # Inception UNet: Based on UNet with inception modules, uses [32, 64, 128, 256, 512]
        'InceptionUNet': [32, 64, 128, 256, 512],  # Inception UNet的特征通道配置
        # TND SEG Net: Based on the paper, uses [32, 64, 128, 256, 512]
        'TNDSEGNet': [32, 64, 128, 256, 512],  # TND SEG Net的特征通道配置
        # MK UNet: Based on the paper, uses [32, 64, 128, 256, 512]
        'MKUNet': [32, 64, 128, 256, 512],  # MK UNet的特征通道配置
        # TransUNet: Reduced for memory efficiency [32, 64, 128, 256]
        'TransUNet': [32, 64, 128, 256],  # TransUNet的特征通道配置（为了内存效率而减少）
        # nnUNet: Based on nnU-Net V2 default, uses [32, 64, 128, 256, 512, 512, 512]
        'nnUNet': [32, 64, 128, 256, 512, 512, 512],  # nnUNet的特征通道配置
        # Swing UNet: Based on UNet with swing connections, uses [64, 128, 256, 512]
        'SwingUNet': [64, 128, 256, 512]  # Swing UNet的特征通道配置
    }