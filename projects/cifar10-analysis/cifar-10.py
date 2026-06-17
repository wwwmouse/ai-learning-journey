import os
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family']= 'SimHei'

# 图片保存目录（在脚本所在目录下）
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results-pictures')

#   __file__:Python 内置变量，值是当前脚本文件的路径
#  os.path.abspath(__file__):把 __file__ 转成绝对路径
#  os.path.dirname(...):取路径中的目录部分，把文件名砍掉
#  os.path.join(..., 'results-pictures'):把两部分拼接成一个合法路径

os.makedirs(SAVE_DIR, exist_ok=True)
#   创建文件夹
#   exist_ok=True的意思是：如果文件夹已经存在，别报错，跳过

def save_plot(filename):
    """保存当前图表到 results-pictures 文件夹"""
    filepath = os.path.join(SAVE_DIR, filename)
    plt.savefig(filepath, dpi=300)
    print(f"已保存: {filename}")
    
# 自动下载 CIFAR-10 数据集（第一次会下载，约 170MB）
print("正在加载 CIFAR-10 数据集...")
dataset = datasets.CIFAR10(root='./data', train=True, download=True)

# 数据集属性
data = dataset.data          # (50000, 32, 32, 3)  numpy数组
targets = dataset.targets    # 50000个标签，0-9

# 类别名称
classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

print(f"数据集大小: {len(data)} 张图片")
print(f"图片尺寸: {data.shape[1:]}")  # (32, 32, 3)
print(f"类别数: {len(classes)}")

# ========================================
# 1. 类别分布统计（柱状图）
# ========================================
print("\n【1. 类别分布统计】")

# 统计每个类别的数量
class_counts = np.bincount(targets)
print("各类别数量:")
for i, (name, count) in enumerate(zip(classes, class_counts)):
    print(f"  {i}.{name}: {count}张")

# 画柱状图
plt.figure(figsize=(10, 5))#创建新画布，宽10英寸，高5英寸
plt.bar(classes, class_counts, color='steelblue')#
plt.xlabel('类别')#横轴标签
plt.ylabel('图片数量')#纵轴标签
plt.title('CIFAR-10 训练集类别分布')#整张图的标题
plt.tight_layout()#自动调整子图间距
save_plot('类别分布柱状图')
plt.show()

# ========================================
# 2. RGB 通道统计（均值、标准差）
# ========================================
print("\n【2. RGB 通道统计】")

# 分离三个通道
r_channel = data[:, :, :, 0]  # 所有图片的红色通道
g_channel = data[:, :, :, 1]  # 绿色通道
b_channel = data[:, :, :, 2]  # 蓝色通道
# 语法中:为取该维度所有，所以这里意思是所有图片，所有行，所有列的第0、1、2个通道

# 计算均值和标准差
r_mean, r_std = r_channel.mean(), r_channel.std()
g_mean, g_std = g_channel.mean(), g_channel.std()
b_mean, b_std = b_channel.mean(), b_channel.std()
# .mean()所有元素的算术平均
# .std()标准差，衡量数据分散程度

print(f"R通道: 均值={r_mean:.2f}, 标准差={r_std:.2f}")
print(f"G通道: 均值={g_mean:.2f}, 标准差={g_std:.2f}")
print(f"B通道: 均值={b_mean:.2f}, 标准差={b_std:.2f}")

# 画 RGB 直方图（三个通道叠在一起）
plt.figure(figsize=(10, 5))#创建新画布，宽10英寸，高5英寸
plt.hist(r_channel.flatten(), bins=50, color='red', alpha=0.5, label='R')
plt.hist(g_channel.flatten(), bins=50, color='green', alpha=0.5, label='G')
plt.hist(b_channel.flatten(), bins=50, color='blue', alpha=0.5, label='B')
# .flatten()把(50000,32,32)展平成(51200000)
# bins=50 把0-255分成50个区间统计
# alpha=0.5 透明度50%
# label 图例标签

plt.xlabel('像素值 (0-255)')
plt.ylabel('频数')
plt.title('CIFAR-10 RGB 通道像素分布')
plt.legend()
plt.tight_layout()
save_plot('RGB像素分布直方图.png')
plt.show()

# ========================================
# 3. 随机展示 16 张图片（4x4 拼图）
# ========================================
print("\n【3. 随机展示16张图片】")

# 随机选16个索引
np.random.seed(42)  # 固定随机种子，结果可复现
indices = np.random.choice(len(data), 16, replace=False)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle('CIFAR-10 随机样本展示', fontsize=16)

for idx, ax in zip(indices, axes.flat):
    img = data[idx]                    # 取出图片 (32, 32, 3)
    label = targets[idx]              # 标签 0-9
    ax.imshow(img)                    # 显示图片
    ax.set_title(classes[label])      # 标题是类别名
    ax.axis('off')                    # 不显示坐标轴

plt.tight_layout()
save_plot('随机样本拼图.png')
plt.show()

print("\n生成3张分析结果图片：")
print("1.类别分布柱状图")
print("2.RGB像素分布直方图")
print("3.随机样本拼图")