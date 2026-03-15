import os  # 导入os模块，用于文件和目录操作
import numpy as np  # 导入NumPy库，用于数值计算
import cv2  # 导入OpenCV库，用于图像处理
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器
import torch  # 导入PyTorch库

class TN3KDataset(Dataset):  # 定义TN3K数据集类，继承自Dataset
    def __init__(self, image_dir, mask_dir, transform=None):
        # 初始化函数，设置图像目录、掩码目录和变换
        self.image_dir = image_dir  # 图像目录
        self.mask_dir = mask_dir  # 掩码目录
        self.transform = transform  # 数据变换
        # 获取所有图像文件和掩码文件
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])  # 按名称排序
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg')])  # 按名称排序
    
    def __len__(self):
        # 返回数据集的大小
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 读取图像和掩码
        img_path = os.path.join(self.image_dir, self.image_files[idx])  # 图像路径
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])  # 掩码路径
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度掩码
        
        # 确保图像和掩码大小一致
        if img.shape != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # 调整掩码大小
        
        # 调整大小到统一尺寸
        img = cv2.resize(img, (448, 384))  # 调整图像大小
        mask = cv2.resize(mask, (448, 384))  # 调整掩码大小
        
        # 二值化掩码
        mask = (mask > 128).astype(np.uint8)  # 大于128的像素设为1
        mask = np.where(mask == 1, 1, 0).astype(np.uint8)  # 确保只有0和1两个值
        
        # 转换为张量
        img = torch.from_numpy(img).unsqueeze(0).float()  # 添加通道维度并转换为浮点数
        mask = torch.from_numpy(mask).long()  # 转换为长整型
        
        # 数据归一化（ZScore）
        mean = img.mean()  # 计算均值
        std = img.std()  # 计算标准差
        img = (img - mean) / (std + 1e-8)  # 归一化，添加1e-8防止除零
        
        if self.transform:
            img, mask = self.transform(img, mask)  # 应用数据变换
        
        return img, mask  # 返回图像和掩码

def get_data_loaders(image_dir, mask_dir, batch_size=19, train_ratio=0.8):
    # 创建数据集实例
    dataset = TN3KDataset(image_dir, mask_dir)
    
    # 分割训练集和验证集
    train_size = int(len(dataset) * train_ratio)  # 训练集大小
    val_size = len(dataset) - train_size  # 验证集大小
    
    # 固定随机种子确保数据分割一致
    generator = torch.Generator().manual_seed(42)  # 设置随机种子
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator  # 随机分割数据集
    )
    
    # 优化数据加载器配置
    num_workers = min(8, os.cpu_count())  # 使用最多8个worker
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 打乱数据
        num_workers=num_workers,  # 并行加载的worker数
        pin_memory=True,  # 启用内存锁定，加速数据传输
        persistent_workers=True  # 保持workers活跃，减少启动开销
    )
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  # 批次大小
        shuffle=False,  # 不打乱数据
        num_workers=num_workers,  # 并行加载的worker数
        pin_memory=True,  # 启用内存锁定，加速数据传输
        persistent_workers=True  # 保持workers活跃，减少启动开销
    )
    
    return train_loader, val_loader  # 返回训练和验证数据加载器