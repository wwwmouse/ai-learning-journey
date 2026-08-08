"""MNIST CNN — 两层卷积 + 分类头，保留图片二维结构。"""
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

    # ===== 数据 =====
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

    # ===== 模型：Conv→Pool→Conv→Pool→Flatten→Linear→Linear =====
    model = nn.Sequential(
        # 第 1 块：提取简单特征（边缘、颜色变化）
        nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 1→16 通道
        nn.ReLU(),
        nn.MaxPool2d(2),                               # 28×28 → 14×14

        # 第 2 块：提取复杂特征（形状组合）
        nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16→32 通道
        nn.ReLU(),
        nn.MaxPool2d(2),                               # 14×14 → 7×7

        # 分类头
        nn.Flatten(),                                  # [batch, 32, 7, 7] → [batch, 1568]
        nn.Linear(1568, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # ===== 训练 =====
    epochs = 10
    train_losses, test_accs = [], []

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        acc = evaluate(model, test_loader, device)

        train_losses.append(avg_loss)
        test_accs.append(acc)
        print(f"epoch {epoch+1:2d}: train_loss={avg_loss:.4f}, test_acc={acc:.2f}%")
        # epoch 10: train_loss=0.0729, test_acc=97.90%

    # ===== 保存 =====
    IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
    plot_curves(train_losses, test_accs,
                os.path.join(IMAGES_DIR, 'mnist_cnn.png'),
                'MNIST CNN')
    save_model(model, os.path.join(MODELS_DIR, 'mnist_cnn.pth'))

if __name__ == '__main__':
    main()
