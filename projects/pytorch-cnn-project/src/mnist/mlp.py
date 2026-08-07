import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===== 0. 路径设置（和 explore_mnist.py 一样）=====
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# ===== 加载数据 =====
# 把 28×28 图片转成 tensor，像素缩放到 0~1
transform = transforms.ToTensor()

train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
#把六万张图切成每 64 张一包

# 定义模型 
# 28*28=784 个像素 → 10 个输出（0~9 哪个概率最大）
model = nn.Sequential(
    nn.Flatten(),              # 把 [64, 1, 28, 28] 摊平成 [64, 784]
    nn.Linear(784, 128),       # 784 个像素 → 128 个神经元
    nn.ReLU(),                 # 激活函数——负数变成 0，正数保持不变
    nn.Linear(128, 64),        # 128 → 64
    nn.ReLU(),
    nn.Linear(64, 10),         # 64 → 10（对应数字 0~9）
)
# Linear 负责"加权求和"，ReLU 负责"掰弯"

# ===== 损失函数和优化器 =====
# CrossEntropyLoss（十分类）—— 内部自动做 softmax
loss_fn = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01) # 优化器

# =====  训练 =====
# zero_grad → forward → loss → backward → step
# 之前一口气算完，现在用一个一个 batch 来算

epochs = 10  # 整个训练集过 10 遍

train_losses = []   # 记录每个 epoch 的平均 loss
test_accs = []      # 记录每个 epoch 的测试准确率

# 遍历数据
for epoch in range(epochs):
    # ===== 训练阶段 =====
    model.train()  # 告诉模型"现在是训练"
    train_loss = 0

    for images, labels in train_loader:          # 每次取 64 张图
        optimizer.zero_grad()                    #  1.清梯度
        outputs = model(images)                  #  2.前向传播
        loss = loss_fn(outputs, labels)          #  3.算 loss
        loss.backward()                          #  4.反向传播
        optimizer.step()                         #  5.更新参数
        train_loss += loss.item()

    # ===== 测试阶段 =====
    model.eval()  # 告诉模型"现在是考试"
    correct = 0
    total = 0

    with torch.no_grad():                        # 考试时不记录梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 取 10 个输出中最大值及其位置，但不关心值，所以用_表示
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    train_losses.append(avg_loss)   # 记下来，待会画图用
    test_accs.append(acc)
    print(f"epoch {epoch+1:2d}: train_loss={avg_loss:.4f}, test_acc={acc:.2f}%")
    # epoch 10: train_loss=0.2301, test_acc=93.62%

# ===== 画训练曲线 =====
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 左图：loss 曲线
ax1.plot(range(1, epochs+1), train_losses, marker='o', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')

# 右图：准确率曲线
ax2.plot(range(1, epochs+1), test_accs, marker='o', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Test Accuracy')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'mnist_mlp.png'), dpi=150)
plt.show()
print(f"训练曲线已保存到 {IMAGES_DIR}/mnist_mlp.png")

# ===== 保存模型 =====
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'mnist_mlp.pth'))
print(f"模型已保存到 {MODELS_DIR}/mnist_mlp.pth")
