# model.py
import torch.nn as nn
from torchvision import models

def create_model(num_classes=101, pretrained=True):
    """
    创建预训练的ResNet-50模型，替换最后的全连接层
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)
    
    # 获取最后一个全连接层的输入特征维度
    in_features = model.fc.in_features
    
    # 替换为新的分类头（101类）
    model.fc = nn.Linear(in_features, num_classes)
    
    # 初始化
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0)
    
    return model

def get_parameter_groups(model, backbone_lr, head_lr, weight_decay):
    """
    将骨干网络和分类头分到不同的参数组，便于分层学习率优化
    """
    # 骨干网络（除fc层以外的所有参数）
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
    ]
    return param_groups