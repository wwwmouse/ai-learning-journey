import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 项目根目录（脚本在 src/ 里，往上一级就是项目根）
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# ===== 第 1 步：定义数据转换 =====
# MNIST 是 28×28 的灰度图片，每张图是一个手写数字(0-9)
# ToTensor() 做两件事：
#   1. 把像素值从 0~255 缩放到 0~1
#   2. 把 numpy/PIL 图片转成 torch 的 tensor 格式
transform = transforms.ToTensor()

# ===== 第 2 步：下载并加载训练集 =====
# 第一次运行会自动下载到 data/ 目录
train_data = datasets.MNIST(
    root=DATA_DIR,         # 用绝对路径，不管从哪运行都对
    train=True,            # True=训练集，False=测试集
    download=True,         # 没下就自动下
    transform=transform    # 对每张图执行第 1 步的转换
)

# ===== 第 3 步：看看数据长什么样 =====
print(f"训练集大小: {len(train_data)} 张图片")
print(f"每张图的形状: {train_data[0][0].shape}")  # torch.Size([1, 28, 28])
print(f"标签: {train_data[0][1]}")                 # 一个数字，比如 5

# 解释一下形状 [1, 28, 28]：
#   1 = 通道数（灰度图只有 1 个通道，彩色图有 RGB 3 个通道）
#   28 = 高度（像素）
#   28 = 宽度（像素）

# ===== 第 4 步：创建 DataLoader =====
# DataLoader 负责分批：把 60000 张图切成一个一个 batch
train_loader = DataLoader(
    train_data,
    batch_size=64,   # 每次取 64 张
    shuffle=True     # 每个 epoch 打乱顺序
)

# ===== 第 5 步：取一个 batch 看看 =====
images, labels = next(iter(train_loader))  # iter 创建一个迭代器，next 取下一个 batch

print(f"\n一个 batch 的形状: {images.shape}")  # torch.Size([64, 1, 28, 28])
print(f"对应标签的形状: {labels.shape}")       # torch.Size([64])
print(f"前 10 个标签: {labels[:10].tolist()}")
