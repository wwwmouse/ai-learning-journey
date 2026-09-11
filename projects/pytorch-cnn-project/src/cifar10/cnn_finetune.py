# CIFAR-10 · 迁移学习（二）：预训练 ResNet-18 + 全量微调
#
# 一句话：不只借现成的特征，还让特征本身适配 CIFAR-10 —— 所以精度应该比冻结版更高。
#
# 对照对象：src/cifar10/cnn.py（从零训练 5 层 CNN，80.03%）
# 与 cnn_pretrained.py 的唯一区别：本文件不冻结（全部可训练），且用更小的 lr 训更久。
#
# 和"冻结版"的关系：
#   冻结版 = 特征提取器不变，只学最后的决策层   → 快，但特征是为 ImageNet 优化的
#   微调版 = 特征也跟着调，专门适配 CIFAR-10    → 慢，但通常更好
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils import set_seed, get_device, train_one_epoch, evaluate, plot_curves, save_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model():
    # 与 cnn_pretrained.py 完全相同的三步改造，保证两个脚本只差"冻不冻"
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 改 stem：7×7 stride 2 → 3×3 stride 1，并去掉最大池化（32×32 经不起 1/4 压缩）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 换分类头：1000 类 → 10 类
    model.fc = nn.Linear(512, 10)
    return model


def main():
    set_seed(42)
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')

    device = get_device()

    # ===== 数据（与 cnn_pretrained.py 完全相同）=====
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_data = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_data  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)

    # ===== 模型 =====
    model = build_model().to(device)

    # ★ 本脚本与 cnn_pretrained.py 的唯一区别在这里：
    #   不冻结任何东西 —— 全部参数都可训练。这就是"微调"和"特征提取"的分界。
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,}  ({trainable / total * 100:.2f}%)")

    loss_fn = nn.CrossEntropyLoss()

    epochs = 20
    # 为什么 lr 要比从零训练小：预训练权重本身已经是一个很不错的解，
    # 步长太大会把它"踩坏"（灾难性遗忘）——微调的 lr 通常比从零训练小半到一个量级。
    # 从零训练那版（cnn.py）用的是 SGD lr=0.01 + StepLR；这里用 0.005 + 余弦退火。
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ===== 训练 =====
    train_losses, test_accs = [], []

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(avg_loss)
        test_accs.append(acc)
        print(f"epoch {epoch+1:2d}: train_loss={avg_loss:.4f}, test_acc={acc:.2f}%")

    # ===== 保存 =====
    IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
    plot_curves(train_losses, test_accs,
                os.path.join(IMAGES_DIR, 'cifar10_finetune.png'),
                'CIFAR-10 ResNet-18 (fine-tuned)')
    save_model(model, os.path.join(MODELS_DIR, 'cifar10_finetune.pth'))

    print(f"\n最终测试准确率: {test_accs[-1]:.2f}%   （对照：从零 5 层 CNN = 80.03%）")
    # 最终测试准确率: 93.98%   （对照：从零 5 层 CNN = 80.03%）

if __name__ == '__main__':
    main()
