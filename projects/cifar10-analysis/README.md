# CIFAR-10 数据集分析

一个用于快速查看 CIFAR-10 数据集基本特征的 Python 脚本。

## 功能

| 输出文件 | 说明 |
|---------|------|
| `cifar10_class_distribution.png` | 10 个类别的样本数量柱状图 |
| `cifar10_rgb_histogram.png` | R/G/B 三通道像素值分布直方图 |
| `cifar10_sample_grid.png` | 随机 16 张样本图片（4×4 拼图） |

## 第三方库
| 库 | 作用|
|----|-----|
| torch | PyTorch 深度学习框架 |
| torchvision | PyTorch 的图像工具包，包含 CIFAR-10 数据集下载接口 |
| matplotlib | 画图库，用来生成柱状图、直方图、拼图 |
| numpy | 数值计算库，处理数组和统计运算 |

## 运行方法
### 安装依赖
```bash
pip install -r requirements.txt
```
- 安装上述的所有第三方库

### 运行脚本
   
```bash
python cifar10_analysis.py
```

注意: 第一次运行会**自动下载 CIFAR-10 数据集**（约 170MB，保存到 `./data` 目录）
下载路径: `./data`

### 运行结果

| 输出文件 | 预览 |
|---------|------|
| 类别分布 | ![类别分布](results-pictures/cifar10_class_distribution.png) |
| RGB直方图 | ![RGB直方图](results-pictures/cifar10_rgb_histogram.png) |
| 样本拼图 | ![样本拼图](results-pictures/cifar10_sample_grid.png) |

## 数据集概况

- **样本数**：50,000（训练集）
- **图片尺寸**：32 × 32 × 3（RGB）
- **类别**：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
