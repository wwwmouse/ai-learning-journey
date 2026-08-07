import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===== 路径设置 =====
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# ===== 1. 数据增强（只改这里）=====
# 原来的 transform 只做了 ToTensor()，图片端端正正
# 现在训练集加随机干扰——旋转、平移，让模型看到"捣乱版"的图片
# 测试集不变——考试时还是正常图片
train_transform = transforms.Compose([
    transforms.RandomRotation(10),        # 随机旋转 ±10 度
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移 10%
    transforms.ToTensor(),                # 转 tensor（和之前一样）
])

test_transform = transforms.ToTensor()    # 测试集不增强，和之前一样

train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=train_transform)
test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ===== 2. 模型定义（加了 Dropout）=====
model = nn.Sequential(
    # 卷积部分不变
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    # 分类头
    nn.Flatten(),
    nn.Linear(1568, 128),
    nn.ReLU(),
    nn.Dropout(0.25),              # ← 新增：随机关掉 25% 神经元，防过拟合
    nn.Linear(128, 10),
)

# ===== 3. 损失函数和优化器 =====
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ===== 4. 学习率调度（新增）=====
# 每 5 个 epoch 把学习率乘 0.5（减半）
# epoch 1-5: lr=0.01, epoch 6-10: lr=0.005, epoch 11-15: lr=0.0025 ...
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ===== 训练（和之前完全一样）=====
epochs = 10

train_losses = []
test_accs = []

for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 测试
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    scheduler.step()  # ← 新增：每个 epoch 末更新学习率

    avg_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    train_losses.append(avg_loss)
    test_accs.append(acc)
    print(f"epoch {epoch+1:2d}: train_loss={avg_loss:.4f}, test_acc={acc:.2f}%")

# ===== 画图 =====
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, epochs+1), train_losses, marker='o', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (optimized)')

ax2.plot(range(1, epochs+1), test_accs, marker='o', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Test Accuracy (optimized)')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'mnist_cnn_optimized.png'), dpi=150)
plt.show()
print(f"训练曲线已保存到 {IMAGES_DIR}/mnist_cnn_optimized.png")

# ===== 保存模型 =====
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'mnist_cnn_optimized.pth'))
print(f"模型已保存到 {MODELS_DIR}/mnist_cnn_optimized.pth")
