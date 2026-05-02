# dataset.py
import torch
from torchvision import datasets, transforms
from config import Config

def get_data_transforms():
    """定义训练集和验证集的数据增强与归一化"""
    # ImageNet的均值和标准差（用于预训练模型归一化）
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return train_transform, test_transform

def load_food101():
    """加载Food-101数据集"""
    train_transform, test_transform = get_data_transforms()
    
    # torchvision的Food-101会自动下载到指定根目录
    train_dataset = datasets.Food101(
        root=Config.data_root,
        split='train',
        transform=train_transform,
        download=False
    )
    
    test_dataset = datasets.Food101(
        root=Config.data_root,
        split='test',
        transform=test_transform,
        download=True
    )
    
    # 类别名称
    classes = train_dataset.classes  # 长度为101的列表
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory
    )
    
    return train_loader, test_loader, classes