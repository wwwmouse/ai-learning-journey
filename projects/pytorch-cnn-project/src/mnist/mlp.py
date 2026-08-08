"""MNIST MLP — 多层感知机，作为 CNN 对比的 baseline。"""
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

    # ===== 模型：784 → 128 → 64 → 10 =====
    model = nn.Sequential(
        nn.Flatten(),               # [64, 1, 28, 28] → [64, 784]
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),          # 0~9 共 10 类
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
        # epoch 10: train_loss=0.2289, test_acc=93.74%
        
    # ===== 保存 =====
    IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
    plot_curves(train_losses, test_accs,
                os.path.join(IMAGES_DIR, 'mnist_mlp.png'),
                'MNIST MLP')
    save_model(model, os.path.join(MODELS_DIR, 'mnist_mlp.pth'))


if __name__ == '__main__':
    main()
