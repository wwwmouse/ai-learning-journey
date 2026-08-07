import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# ===== 数据（改成 CIFAR-10）=====
transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ===== 模型（和 MNIST CNN 一样结构，只改了通道数和分类头尺寸）=====
model = nn.Sequential(
    # 第 1 块
    nn.Conv2d(3, 16, kernel_size=3, padding=1),    # ← 输入 3 通道（RGB），MNIST 是 1
    nn.ReLU(),
    nn.MaxPool2d(2),                                 # 32×32 → 16×16

    # 第 2 块
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),                                 # 16×16 → 8×8

    # 分类头
    nn.Flatten(),                                    # [batch, 32, 8, 8] → [batch, 2048]
    nn.Linear(2048, 128),                            # ← 2048，MNIST 是 1568
    nn.ReLU(),
    nn.Linear(128, 10),                              # 10 类：飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ===== 训练（一个字没改）=====
epochs = 10
train_losses, test_accs = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    train_losses.append(avg_loss)
    test_accs.append(acc)
    print(f"epoch {epoch+1:2d}: train_loss={avg_loss:.4f}, test_acc={acc:.2f}%")
    # epoch 10: train_loss=1.2863, test_acc=52.01%

# ===== 画图 =====
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, epochs+1), train_losses, marker='o', color='blue')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('CIFAR-10 Training Loss')
ax2.plot(range(1, epochs+1), test_accs, marker='o', color='green')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('CIFAR-10 Test Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'cifar10_cnn.png'), dpi=150)
plt.show()
print(f"训练曲线已保存到 {IMAGES_DIR}/cifar10_cnn.png")

# ===== 保存模型 =====
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'cifar10_cnn.pth'))
print(f"模型已保存到 {MODELS_DIR}/cifar10_cnn.pth")
