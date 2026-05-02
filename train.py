# train.py
import torch
import torch.nn as nn
import torch.optim as optim
# 修改：使用新的 torch.amp API 替代 torch.cuda.amp（消除弃用警告）
from torch.amp import GradScaler, autocast
import time
import os
import sys
from tqdm import tqdm
from config import Config
from dataset import load_food101
from model import create_model, get_parameter_groups
from utils import setup_seed, AverageMeter, save_checkpoint, plot_curves, Logger


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, logger):
    """训练一个 epoch，支持混合精度和梯度累积"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{Config.epochs} [Train]')
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(pbar):
        # 将数据异步传输到 GPU
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # 混合精度前向传播（指定设备类型为 cuda）
        with autocast(device_type='cuda', enabled=Config.use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # 梯度缩放与反向传播
        scaler.scale(loss).backward()

        # 梯度累积：每 accumulation_steps 个 batch 更新一次参数
        if (batch_idx + 1) % Config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 计算 top-1 准确率
        acc = (outputs.argmax(1) == targets).float().mean()
        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))

        # 更新进度条信息
        pbar.set_postfix({'Loss': losses.avg, 'Acc': top1.avg})

        # 定期打印日志
        if batch_idx % Config.log_interval == 0:
            logger.info(f'Train Epoch {epoch + 1} [{batch_idx}/{len(loader)}] '
                        f'Loss: {losses.avg:.4f} Acc: {top1.avg:.4f}')

    return losses.avg, top1.avg


@torch.no_grad()
def validate(model, loader, criterion, epoch, logger):
    """验证集评估，不计算梯度"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{Config.epochs} [Val]')
    for images, targets in pbar:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        acc = (outputs.argmax(1) == targets).float().mean()

        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))
        pbar.set_postfix({'Loss': losses.avg, 'Acc': top1.avg})

    logger.info(
        f'Validation Epoch {epoch + 1} | Loss: {losses.avg:.4f} | Acc: {top1.avg:.4f}')
    return losses.avg, top1.avg


def main():
    # 初始化：创建保存目录和日志文件，设置随机种子
    Config.setup_dirs()
    logger = Logger('./logs/train.log')
    setup_seed(Config.seed)

    # 加载 Food-101 数据
    logger.info("Loading Food-101 dataset...")
    train_loader, val_loader, classes = load_food101()
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 创建模型并移至 GPU
    model = create_model(num_classes=Config.num_classes, pretrained=True)
    model = model.cuda()

    # 带标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)

    # 分层优化器：backbone 使用较小学习率，分类头使用较大学习率
    param_groups = get_parameter_groups(
        model,
        backbone_lr=Config.backbone_lr,
        head_lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    optimizer = optim.AdamW(param_groups, lr=Config.backbone_lr,
                            weight_decay=Config.weight_decay)

    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=1e-6)

    # 混合精度梯度缩放器（修改：指定设备 'cuda'，消除 FutureWarning）
    scaler = GradScaler('cuda', enabled=Config.use_amp)

    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(Config.epochs):
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, logger)
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, epoch, logger)

        # 记录曲线数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1} LR: {current_lr:.6f}")

        # 保存最佳模型（基于验证准确率）
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(Config.save_dir, 'best_model.pth'))
            logger.info(f"New best model saved with acc {best_acc:.4f}")

        # 每 5 个 epoch 保存一个检查点
        if (epoch + 1) % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, filename=os.path.join(Config.save_dir, f'checkpoint_epoch{epoch + 1}.pth'))

    # 绘制训练曲线并保存
    plot_curves(train_losses, val_losses, train_accs, val_accs,
                save_path='./logs/training_curves.png')
    logger.info(f"Training finished. Best validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()