import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.configs.base_config import Config
from models.data_loader import get_data_loaders
from models.nnunet import PlainUNet

# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self, batch_dice=False):
        super(DiceLoss, self).__init__()
        self.batch_dice = batch_dice
    
    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = nn.functional.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets).sum(dim=(2, 3))
        if self.batch_dice:
            intersection = intersection.sum(dim=0)
            union = inputs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        else:
            union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = 2 * intersection / (union + 1e-8)
        loss = 1 - dice.mean()
        return loss

# DICE+BCE组合损失
class DiceBCELoss(nn.Module):
    def __init__(self, batch_dice=False, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(batch_dice=batch_dice)
        self.bce_loss = nn.CrossEntropyLoss()
        self.bce_weight = bce_weight
    
    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        return dice_loss + self.bce_weight * bce_loss

# 计算评估指标
def calculate_metrics(preds, targets):
    preds = torch.argmax(preds, dim=1)
    # 只考虑前景区域（值为1的像素）
    preds_foreground = (preds == 1)
    targets_foreground = (targets == 1)
    
    # 计算TP、FP、FN
    TP = (preds_foreground & targets_foreground).sum().item()
    FP = (preds_foreground & ~targets_foreground).sum().item()
    FN = (~preds_foreground & targets_foreground).sum().item()
    
    # 计算Dice系数
    if TP + FP + FN == 0:
        dice = 1.0
    else:
        dice = 2 * TP / (2 * TP + FP + FN)
    
    # 计算IOU (交并比)
    if TP + FP + FN == 0:
        iou = 1.0
    else:
        iou = TP / (TP + FP + FN)
    
    # 计算召回率 (Recall)
    if TP + FN == 0:
        recall = 1.0
    else:
        recall = TP / (TP + FN)
    
    # 计算精确率 (Precision)
    if TP + FP == 0:
        precision = 1.0
    else:
        precision = TP / (TP + FP)
    
    return dice, iou, recall, precision

# 训练模型
def train_model(model, model_name, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = DiceBCELoss(batch_dice=config.BATCH_DICE)
    optimizer = optim.Adam(model.parameters(), lr=config.INITIAL_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # 初始化混合精度训练
    scaler = GradScaler()
    
    best_dice = 0.0
    train_times = []
    val_dices = []
    
    print(f"开始训练 {model_name}...")
    
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 缩放损失以防止梯度下溢
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        epoch_time = time.time() - start_time
        train_times.append(epoch_time)
        
        # 每两轮进行一次验证
        if (epoch + 1) % 2 == 0:
            # 验证
            model.eval()
            val_dice = 0.0
            val_iou = 0.0
            val_recall = 0.0
            val_precision = 0.0
            with torch.no_grad():
                with autocast():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        dice, iou, recall, precision = calculate_metrics(outputs, targets)
                        val_dice += dice
                        val_iou += iou
                        val_recall += recall
                        val_precision += precision
            
            val_dice /= len(val_loader)
            val_iou /= len(val_loader)
            val_recall /= len(val_loader)
            val_precision /= len(val_loader)
            val_dices.append(val_dice)
            
            print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Val Dice: {val_dice:.4f}, Val IOU: {val_iou:.4f}, Val Recall: {val_recall:.4f}, Val Precision: {val_precision:.4f}, Time: {epoch_time:.2f}s")
            
            if val_dice > best_dice:
                best_dice = val_dice
                # 保存最佳模型
                os.makedirs(os.path.join(config.RESULTS_DIR, model_name), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(config.RESULTS_DIR, model_name, 'best_model.pth'))
        else:
            # 非验证轮次只打印训练损失
            print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")
        
        scheduler.step()
    
    # 保存训练结果
    results = {
        'model_name': model_name,
        'best_dice': best_dice,
        'val_dices': val_dices,
        'train_times': train_times,
        'total_training_time': sum(train_times),
        'average_epoch_time': np.mean(train_times)
    }
    
    with open(os.path.join(config.RESULTS_DIR, model_name, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"{model_name} 训练完成！最佳验证Dice: {best_dice:.4f}")
    return results

# 主函数
def main():
    config = Config()
    set_seed(config.SEED)
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(
        os.path.join(config.DATA_DIR, config.TEST_IMAGE_DIR),
        os.path.join(config.DATA_DIR, config.TEST_MASK_DIR),
        batch_size=config.BATCH_SIZE
    )
    
    # 定义模型
    model_name = 'nnUNet'
    model = PlainUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.NUM_CLASSES,
        features=config.MODEL_FEATURES[model_name]
    )
    
    # 训练模型
    results = train_model(model, model_name, train_loader, val_loader, config)
    
    # 打印结果
    print("\n=== 训练结果 ===")
    print(f"模型: {model_name}")
    print(f"最佳Dice: {results['best_dice']:.4f}")
    print(f"总训练时间: {results['total_training_time']:.2f}s")

if __name__ == '__main__':
    main()