# CIFAR-10 CNN + 数据增强 —— 和 cnn 用同一套模型，变量仅为"是否数据增强"
# 5 层卷积 + BatchNorm，50 epoch 训练。
# BatchNorm 是深度网络能稳定训练的关键——不加 BN 时 5 层卷积直接崩到 20%，
# 加了 BN 后准确率提升到 ~80%。
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

    # ===== 模型（5 层 CNN + BatchNorm + Dropout）=====
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),         # 标准化——没有它深层网络训不起来
        nn.ReLU(),
        nn.MaxPool2d(2),            # 32×32 → 16×16

        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),            # 16×16 → 8×8

        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),            # 8×8 → 4×4

        nn.Flatten(),               # 128 × 4 × 4 = 2048
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 10),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ===== 训练 =====
    epochs = 50
    train_losses, test_accs = [], []

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device) # 接受平均损失率
        acc = evaluate(model, test_loader, device) # 接受测试集上准确率
        scheduler.step()

        train_losses.append(avg_loss)
        test_accs.append(acc)
        print(f"epoch {epoch+1:2d}: train_loss={avg_loss:.4f}, test_acc={acc:.2f}%")
        # epoch 50: train_loss=0.6011, test_acc=80.03%
        
    # ===== 保存 =====
    IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
    plot_curves(train_losses, test_accs,
                os.path.join(IMAGES_DIR, 'cifar10_cnn_optimized.png'),
                'CIFAR-10 CNN+ (optimized)')
    save_model(model, os.path.join(MODELS_DIR, 'cifar10_cnn_optimized.pth'))


if __name__ == '__main__':
    main()
