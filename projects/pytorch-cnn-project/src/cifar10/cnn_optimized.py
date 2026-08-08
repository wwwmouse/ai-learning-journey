"""CIFAR-10 CNN + 防过拟合套餐（数据增强 + Dropout + 学习率调度）。

注意：当前这个两层网络在 CIFAR-10 上还处于欠拟合状态（10 epoch 训不够），
优化技巧的效果有限。真正的提升方向是加深网络或增加训练轮数。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import set_seed, get_device, train_one_epoch, evaluate, plot_curves, save_model


def main():
    set_seed(42)
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')

    device = get_device()

    # ===== 数据（训练集加增强，测试集不加）=====
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()

    train_data = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_data  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

    # ===== 模型（CNN + Dropout）=====
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),     # 3 通道（RGB）
        nn.ReLU(),
        nn.MaxPool2d(2),                                 # 32×32 → 16×16

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),                                 # 16×16 → 8×8

        nn.Flatten(),
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Dropout(0.25),                                # ← 随机关 25% 神经元
        nn.Linear(128, 10),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ===== 训练 =====
    epochs = 15
    train_losses, test_accs = [], []

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(avg_loss)
        test_accs.append(acc)
        print(f"epoch {epoch+1:2d}: train_loss={avg_loss:.4f}, test_acc={acc:.2f}%")
        # epoch 15: train_loss=1.5052, test_acc=49.99%
        
    # ===== 保存 =====
    IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
    plot_curves(train_losses, test_accs,
                os.path.join(IMAGES_DIR, 'cifar10_cnn_optimized.png'),
                'CIFAR-10 CNN+ (optimized)')
    save_model(model, os.path.join(MODELS_DIR, 'cifar10_cnn_optimized.pth'))


if __name__ == '__main__':
    main()
