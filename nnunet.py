import torch  # 导入PyTorch库，用于深度学习模型构建
import torch.nn as nn  # 导入PyTorch的神经网络模块

class PlainUNet(nn.Module):  # 定义PlainUNet模型类，继承自nn.Module
    """普通的 nnU-Net 模型实现"""
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256, 512, 512, 512]):
        # 初始化函数，设置输入通道数、输出通道数和特征图大小列表
        super(PlainUNet, self).__init__()  # 调用父类的初始化函数
        
        # 编码器部分，用于特征提取
        self.encoder = nn.ModuleList()  # 创建一个模块列表，用于存储编码器的卷积块
        for i, feature in enumerate(features):  # 遍历特征图大小列表
            if i == 0:  # 第一个卷积块，输入通道数为in_channels
                self.encoder.append(self._conv_block(in_channels, feature))
            else:  # 后续卷积块，输入通道数为前一个特征图大小
                self.encoder.append(self._conv_block(features[i-1], feature))
        
        # 池化层，用于下采样
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，池化核大小为2x2，步长为2
        
        # 解码器部分，用于上采样和特征融合
        self.decoder = nn.ModuleList()  # 创建一个模块列表，用于存储解码器的转置卷积和卷积块
        for i in range(len(features)-2, -1, -1):  # 从倒数第二个特征图大小开始，倒序遍历
            # 添加转置卷积层，用于上采样
            self.decoder.append(
                nn.ConvTranspose2d(
                    features[i+1], features[i], kernel_size=2, stride=2  # 输入通道数为features[i+1]，输出通道数为features[i]
                )
            )
            # 添加卷积块，输入通道数是features[i] * 2（因为要拼接跳跃连接）
            self.decoder.append(self._conv_block(features[i] * 2, features[i]))
        
        # 输出层，将特征图映射到目标通道数
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  # 1x1卷积，不改变特征图大小
    
    def _conv_block(self, in_channels, out_channels):
        # 定义卷积块，包含两个卷积层和激活函数
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3卷积，保持特征图大小不变
            nn.InstanceNorm2d(out_channels),  # 实例归一化，加速训练
            nn.LeakyReLU(inplace=True),  # LeakyReLU激活函数，避免梯度消失
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 第二个3x3卷积
            nn.InstanceNorm2d(out_channels),  # 再次实例归一化
            nn.LeakyReLU(inplace=True)  # 再次使用LeakyReLU激活函数
        )
    
    def forward(self, x):  # 前向传播函数
        skip_connections = []  # 存储跳跃连接的特征图
        
        # 编码器部分
        for i, down in enumerate(self.encoder):  # 遍历编码器的卷积块
            x = down(x)  # 通过卷积块
            if i != len(self.encoder) - 1:  # 除了最后一个卷积块外，其他都添加到跳跃连接
                skip_connections.append(x)  # 保存当前特征图用于跳跃连接
                x = self.pool(x)  # 池化下采样
        
        # 解码器部分
        for i, up in enumerate(self.decoder):  # 遍历解码器的模块
            if i % 2 == 0:  # 偶数索引是转置卷积层
                x = up(x)  # 上采样
                skip = skip_connections.pop()  # 取出对应编码器的特征图
                # 确保尺寸匹配
                if x.shape != skip.shape:
                    x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                # 拼接跳跃连接的特征图
                x = torch.cat((skip, x), dim=1)  # 在通道维度上拼接
            else:  # 奇数索引是卷积块
                x = up(x)  # 通过卷积块
        
        return self.final_conv(x)  # 通过输出层，返回最终结果