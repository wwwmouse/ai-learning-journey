# PyTorch CNN 图像分类项目

第三阶段学习项目，在 MNIST 和 CIFAR-10 上对比 MLP、CNN、优化版 CNN 与迁移学习。

## 项目结构

```
├── data/              # 数据集（MNIST + CIFAR-10）
├── src/
│   ├── utils.py               # 公共函数（训练/评估/画图/保存/设备探测）
│   ├── mnist/
│   │   ├── explore.py          # 查看 MNIST 数据格式
│   │   ├── mlp.py              # 多层感知机
│   │   ├── cnn.py              # 卷积神经网络（2 层卷积 + 池化）
│   │   └── cnn_optimized.py    # CNN + 数据增强/Dropout/学习率调度
│   └── cifar10/
│       ├── mlp.py
│       ├── cnn.py                    # 5 层卷积 + BatchNorm
│       ├── cnn_optimized.py          # 同上 + 数据增强
│       ├── cnn_pretrained.py         # 迁移学习1：冻结 backbone，只训分类头（3 epoch）
│       └── cnn_finetune.py           # 迁移学习2：预训练 ResNet-18 全量微调（20 epoch）
├── images/            # 训练曲线图（8 张）
├── models/            # 保存的模型 .pth 文件（8 个）
├── README.md
└── requirements.txt
```

> 说明：`mnist/` 4 个脚本 + `cifar10/` 5 个脚本 = **9 个脚本共用 `utils.py`**。

## 运行方法

```bash
pip install -r requirements.txt

# MNIST
python src/mnist/mlp.py
python src/mnist/cnn.py
python src/mnist/cnn_optimized.py

# CIFAR-10（按这个顺序看，是一条改进链）
python src/cifar10/mlp.py             # 51.94%  全连接看不到空间结构
python src/cifar10/cnn.py             # 76.88%  换卷积，+24.94
python src/cifar10/cnn_optimized.py   # 80.03%  加数据增强，+3.15
python src/cifar10/cnn_pretrained.py  # 63.63%  迁移学习1：冻结预训练
python src/cifar10/cnn_finetune.py    # 93.98%  迁移学习2：全量微调
```


## 实验结果

| | MLP | CNN | CNN + 数据增强 |
|---|---|---|---|
| MNIST | 93.74% | 97.90% | 97.26% |
| CIFAR-10 | 51.94% | 76.88% | 80.03% |

> CIFAR-10 使用 5 层卷积 + BatchNorm + momentum SGD，统一训练 50 epoch。
> MNIST 使用 2 层卷积 + 10 epoch。
> （迁移学习那两个脚本是另一套配置，见下文「两个迁移学习实验怎么读」。）

**CNN（无增强）vs CNN+数据增强**：前者训练 loss 0.07 但测试 76.88%，后者训练 loss 0.60 但测试 80.03%。
数据增强让模型更难"背答案"，训练集上表现变差，但测试集泛化更好——这正是防过拟合手段应该起到的效果。

**MLP vs CNN**：CIFAR-10 上 CNN 比 MLP 高 **+24.94** 个百分点。纯全连接看不到像素间的空间关系，在真实图片上天然吃亏。

- 所有结果已用 `set_seed(42)` 固定随机种子，可复现
- 优化器统一使用 SGD + momentum=0.9 + StepLR

## 训练曲线

**MNIST（10 epoch）**

| MLP | CNN | CNN + 防过拟合 |
|---|---|---|
| ![MNIST MLP](images/mnist_mlp.png) | ![MNIST CNN](images/mnist_cnn.png) | ![MNIST CNN+](images/mnist_cnn_optimized.png) |

**CIFAR-10（50 epoch）**

| MLP | CNN | CNN + 数据增强 |
|---|---|---|
| ![CIFAR-10 MLP](images/cifar10_mlp.png) | ![CIFAR-10 CNN](images/cifar10_cnn.png) | ![CIFAR-10 CNN+](images/cifar10_cnn_optimized.png) |

**迁移学习（预训练 ResNet-18，为 32×32 改造 stem）**

| 冻结 backbone（只训分类头） | 全量微调 |
|---|---|
| ![冻结 backbone](images/cifar10_pretrained.png) | ![全量微调](images/cifar10_finetune.png) |

> 每个模型都是 loss + 准确率双图。两条曲线的差距就是判断欠拟合 / 过拟合的依据。

## 两个迁移学习实验怎么读

这两个脚本**不是"哪个更强"的对照组**，它们回答的是两个不同的问题。
先记住它们**唯一的区别**：

| | `cnn_pretrained.py`（冻结） | `cnn_finetune.py`（微调） |
|---|---|---|
| 可训练参数 | 只训分类头（5,130 / 11,173,962 ≈ **0.05%**） | **全部**（11,173,962，100%） |
| 优化器 | Adam, lr=1e-3 | SGD, lr=0.005 + 余弦退火 + weight_decay |
| epoch | **3** | **20** |
| 结果 | **63.63%** | **93.98%** |
| 定位 | **诊断工具**：验证预训练权重有没有被装上 | **能力上限**：这套方法真正能到哪 |

### ① 冻结版是"诊断"，不是"选手"

**它 63.63% 低于从零训练的 80.03%** —— 但这不构成"迁移学习没用"的结论。原因是训练预算根本不同
（3 epoch vs 50 epoch），而且冻结版的特征是**为 ImageNet 优化的**，不是为 CIFAR-10 优化的
—— 它本来就该比微调版差。**这个比较不能当成绩看，它的用途是诊断。**

**它的真正价值在于回答："我搬过来的预训练权重，到底生效了没有？"**

判断依据是一条完整的诊断链：

```text
第一版（有 bug）：直接 nn.Conv2d(...) 新建 stem 层     → 46.99%
   ↓ 病因
新建的那层是【随机初始化】的，而 backbone 又被冻结
   → 这层永远得不到梯度，等于让一个随机层守在特征提取器入口
   ↓ 证据
实测 conv1 与初始值的平均绝对差 = 0.00000000（精确为零，证明它一动没动）
   ↓ 解法
用 F.interpolate 把原来的 7×7 预训练卷积核插值成 3×3 装回去
   ↓ 结果
63.63%   ← 权重确实被装上了
```

**这条链的价值在于演示了"看到一个坏数字时，怎么从机制上找病因"，而不是猜参数。**

### ② 微调版才是"换更好的起点"

微调版的 93.98% 相对从零训练的 80.03% 是 **+13.95**，是本项目最大的一次提升。

**为什么有效**：预训练权重本身就是"一个已经很不错的解"，微调是站在它肩膀上继续走，
而不是从随机初始化开始爬。

**提醒**：+13.95 这一步**同时变了两件事** ——
1.架构从 5 层自建 CNN（108 万参数）换成 ResNet-18（1117 万参数）
2.起点从随机初始化换成 ImageNet 预训练。
所以 **+13.95 是"更强网络 + 预训练"的合计收益，不能全部归因于迁移学习本身**。
要单独测出预训练的贡献，需要补一个**从零训练的 ResNet-18** 作对照（本项目尚未做）。

**微调要成功，有三个前提（缺一个就失效）**：

- **lr 必须调小**（0.005，从零训练那版是 0.01）：预训练权重已经是个好解，
  步长太大会把它"踩坏"（灾难性遗忘）
- **stem 要改造**：ResNet-18 原本第一层是 7×7 stride 2 + 3×3 最大池化，
  一上来就把图缩到 1/4。224×224 的 ImageNet 受得了，32×32 的 CIFAR 经不起
  —— 换成 3×3 stride 1，池化换 `nn.Identity()`
- **输入必须用 ImageNet 的归一化统计量**（`mean=[0.485,0.456,0.406]`）：
  预训练权重就是在这个输入分布上学出来的，喂别的分布会让前几层"常识"失配

### ③ 冻结版里一个容易漏的细节

`requires_grad=False` 只停**梯度**，不停 **BatchNorm**。
`utils.train_one_epoch()` 开头会调 `model.train()`，此时冻结层里的 BN 仍然会用
新数据更新 `running_mean/running_var` —— 相当于让统计量适配 CIFAR 的分布。
通常是好事，所以本项目不做额外处理。（想让 BN 也完全冻结，得给冻结层单独调 `.eval()`。）

## 技术栈

Python 3 / PyTorch / torchvision / matplotlib / numpy
