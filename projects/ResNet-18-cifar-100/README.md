# CIFAR-100 · ResNet-18 从零训练

在 CIFAR-100 上**从随机初始化**训练 ResNet-18（不使用任何预训练权重），
建立可复现的基线：**测试准确率 80.10%**。

## 项目结构

```
ResNet-18-cifar-100/
├── data/cifar-100-python/             # 数据集（手动放置，已 gitignore）
├── src/
│   ├── explore_data.py                # 数据探索：类别分布 / 粗类对应表 / 100 类总览 / 易混淆类别
│   └── train.py                       # 主训练脚本（自包含单文件）
├── models/
│   ├── resnet18_cifar100_best.pth     # 最佳权重
│   └── checkpoint.pth                 # 断点续训用（含优化器和调度器状态）
├── images/                            # 训练曲线、每类准确率、数据探索图（共 5 张）
├── logs/train_log.csv                 # 逐 epoch 日志
├── requirements.txt
└── README.md
```

## 运行方法

```bash
pip install -r requirements.txt

# 1. 准备数据（手动放一次，见下）
# 2. 数据探索（约 1 分钟）
python src/explore_data.py

# 3. 冒烟测试：全量数据只跑 3 轮，约 45 秒
python src/train.py --epochs 3

# 4. 完整训练：200 epoch，实测不到 50 分钟
python src/train.py

# 中途断了就续训
python src/train.py --resume
```

### 数据准备

手动放置，目录结构必须是：

```
data/cifar-100-python/{train, test, meta}
```

| 项 | 值 |
|---|---|
| 下载地址 | `https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz` |
| 压缩包 md5 | `eb9058c3a382ffc7106e4002c42a8d85` |

验证是否到位：

```bash
python -c "from torchvision import datasets; d=datasets.CIFAR100(root='data',train=True,download=False); print(len(d))"
# 期望输出： 50000
```

> 数据到位后，脚本里的 `download=True` 会通过完整性校验并**自动跳过下载**，无需改代码。
> 不能直接用 torchvision 自带下载——它硬编码的地址已随官网搬家失效（原因见笔记）。

## 数据集

| | CIFAR-10 | CIFAR-100 |
|---|---|---|
| 总图片 | 60,000 | 60,000 |
| 类别数 | 10 | **100** |
| 每类训练样本 | 5,000 | **500** |
| 随机猜 | 10% | **1%** |

总图片数相同，区别只在分摊方式——**CIFAR-100 每类的训练证据只有 CIFAR-10 的 1/10**。
100 个细类可归成 20 个粗类（superclass），数据集自带 `fine_labels` / `coarse_labels` 两套标签。

> ⚠ 粗类编号**不连续**（粗类 0 的成员是 4/30/55/72/95），不能用 `fine_label // 5` 推算。

## 实验结果

| 指标 | 结果 |
|---|---|
| **最佳测试准确率** | **80.10%**（epoch 196） |
| 随机猜基线 | 1.00% |
| 训练耗时 | 不到 50 分钟（200 epoch，≈15 秒/epoch） |
| 最终 train_loss | 0.8252 |
| 复现方式 | `--seed 42` + `cudnn.deterministic=True` |

测试准确率在 epoch 150 后基本走平（epoch 100 → 68.06%，150 → 74.78%，200 → 80.09%），
**epoch 预算刚好用完**，再加 epoch 的收益估计在 1 个点以内。

### 每类准确率：失败集中在"类内相似"的组

| 最差 5 类 | 准确率 | 最好 5 类 | 准确率 |
|---|---|---|---|
| man | 53% | road | 97% |
| bowl | 57% | wardrobe | 97% |
| boy | 57% | skunk | 96% |
| girl | 58% | motorcycle | 96% |
| otter | 60% | orange | 96% |

两条规律：

1. **`people` 粗类几乎全军覆没**——`man`(53%)、`boy`(57%)、`girl`(58%) 拿下最差榜第 1、3、4 名。
   32×32 像素下，男人/女人/男孩/女孩的视觉差异本来就极小。
2. **同一个粗类内部可以差 31 个点**——`household furniture` 里的 `wardrobe` 97%、`couch` 66%。

> **决定难度的是"类内视觉相似度"，不是"粗类归属"。**
> 粗类只是数据集的组织方式，模型眼里没有这么整齐的层级。

## 训练曲线

![训练曲线](images/resnet18_cifar100_curves.png)

左：训练 loss（4.24 → 0.83）；右：测试准确率（10.52% → 80.09%）。

## 每类准确率

![每类准确率](images/resnet18_cifar100_per_class.png)

100 个类按准确率降序排列，红线是全类平均（80.1%）。**这条曲线的坡度就是 CIFAR-100 难度的分布**：最好 97%，最差 53%，跨度 **44 个点**。

## 数据探索

| | |
|---|---|
| ![类别分布](images/cifar100_class_balance.png) | ![100 类总览](images/cifar100_sample_grid.png) |

![易混淆类别](images/cifar100_confusing_classes.png)

`cifar100_confusing_classes.png` 展示的 `people` 粗类（baby / boy / girl / man / woman），
**训练完回看，正是最差榜的主力**。

## 技术栈

Python 3.14 / PyTorch 2.13.0+cu130 / torchvision 0.28.0 / matplotlib / numpy

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop（sm_120，Blackwell） |
| 参数量 | 11,220,132（CIFAR 版 ResNet-18，改了 stem） |
| batch=128 前向显存 | 约 193 MB |
| 训练速度 | ≈ 15 秒/epoch |

## 相关笔记

- **概念与原理**（预训练 vs 从零、CNN 基础、感受野）→ `notes/deep-learning/sklearn-to-nn.md`
- **工程手段**（argparse / 学习率调度 / 混合精度 / 断点续训）→ `notes/deep-learning/pytorch-engineering.md`
