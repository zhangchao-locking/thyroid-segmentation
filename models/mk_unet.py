import torch  # 导入PyTorch库，用于深度学习模型构建
import torch.nn as nn  # 导入PyTorch的神经网络模块

class MKConvBlock(nn.Module):  # 定义多尺度卷积块，用于提取不同尺度的特征
    def __init__(self, in_channels, out_channels):
        # 初始化函数，设置输入通道数和输出通道数
        super(MKConvBlock, self).__init__()  # 调用父类的初始化函数
        # 计算每个分支的输出通道数，确保总和为out_channels
        self.branch_channels = out_channels // 3  # 将输出通道数平均分配给3个分支
        remainder = out_channels % 3  # 计算余数，确保通道数总和正确
        
        # 多尺度卷积分支，使用不同大小的卷积核
        # 第一个分支：3x3卷积，提取小尺度特征
        self.branch1 = nn.Conv2d(in_channels, self.branch_channels + (1 if remainder >= 1 else 0), kernel_size=3, padding=1)
        # 第二个分支：5x5卷积，提取中等尺度特征
        self.branch2 = nn.Conv2d(in_channels, self.branch_channels + (1 if remainder >= 2 else 0), kernel_size=5, padding=2)
        # 第三个分支：7x7卷积，提取大尺度特征
        self.branch3 = nn.Conv2d(in_channels, self.branch_channels, kernel_size=7, padding=3)
        
        # 融合特征
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # 1x1卷积，融合多尺度特征
        self.norm1 = nn.InstanceNorm2d(out_channels)  # 实例归一化，加速训练
        self.relu1 = nn.LeakyReLU(inplace=True)  # LeakyReLU激活函数，避免梯度消失
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # 3x3卷积，进一步处理特征
        self.norm2 = nn.InstanceNorm2d(out_channels)  # 再次实例归一化
        self.relu2 = nn.LeakyReLU(inplace=True)  # 再次使用LeakyReLU激活函数
    
    def forward(self, x):  # 前向传播函数
        # 并行计算多个分支
        b1 = self.branch1(x)  # 通过3x3卷积分支
        b2 = self.branch2(x)  # 通过5x5卷积分支
        b3 = self.branch3(x)  # 通过7x7卷积分支
        # 拼接特征
        out = torch.cat([b1, b2, b3], dim=1)  # 在通道维度上拼接三个分支的输出
        # 融合和进一步处理
        out = self.fusion(out)  # 融合多尺度特征
        out = self.norm1(out)  # 归一化
        out = self.relu1(out)  # 激活
        out = self.conv(out)  # 进一步处理特征
        out = self.norm2(out)  # 再次归一化
        out = self.relu2(out)  # 再次激活
        return out

class MKUNet(nn.Module):  # 定义MKUNet模型类，继承自nn.Module
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256, 512, 512, 512]):
        # 初始化函数，设置输入通道数、输出通道数和特征图大小列表
        super(MKUNet, self).__init__()  # 调用父类的初始化函数
        
        # 编码器部分，用于特征提取
        self.encoder = nn.ModuleList()  # 创建一个模块列表，用于存储编码器的MKConvBlock
        for i, feature in enumerate(features):  # 遍历特征图大小列表
            if i == 0:  # 第一个MKConvBlock，输入通道数为in_channels
                self.encoder.append(MKConvBlock(in_channels, feature))
            else:  # 后续MKConvBlock，输入通道数为前一个特征图大小
                self.encoder.append(MKConvBlock(features[i-1], feature))
        
        # 池化层，用于下采样
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，池化核大小为2x2，步长为2
        
        # 解码器部分，用于上采样和特征融合
        self.decoder = nn.ModuleList()  # 创建一个模块列表，用于存储解码器的转置卷积和MKConvBlock
        for i in range(len(features)-2, -1, -1):  # 从倒数第二个特征图大小开始，倒序遍历
            # 添加转置卷积层，用于上采样
            self.decoder.append(
                nn.ConvTranspose2d(
                    features[i+1], features[i], kernel_size=2, stride=2  # 输入通道数为features[i+1]，输出通道数为features[i]
                )
            )
            # 添加MKConvBlock，输入通道数是features[i] * 2（因为要拼接跳跃连接）
            self.decoder.append(MKConvBlock(features[i] * 2, features[i]))
        
        # 输出层，将特征图映射到目标通道数
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  # 1x1卷积，不改变特征图大小
    
    def forward(self, x):  # 前向传播函数
        skip_connections = []  # 存储跳跃连接的特征图
        
        # 编码器部分
        for i, down in enumerate(self.encoder):  # 遍历编码器的MKConvBlock
            x = down(x)  # 通过MKConvBlock
            if i != len(self.encoder) - 1:  # 除了最后一个MKConvBlock外，其他都添加到跳跃连接
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
            else:  # 奇数索引是MKConvBlock
                x = up(x)  # 通过MKConvBlock
        
        return self.final_conv(x)  # 通过输出层，返回最终结果