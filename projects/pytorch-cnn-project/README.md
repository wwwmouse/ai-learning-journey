# PyTorch CNN 图像分类项目

第三阶段学习项目，在 MNIST 和 CIFAR-10 上对比 MLP、CNN、优化版 CNN。

## 项目结构

```
├── data/              # 数据集（MNIST + CIFAR-10）
├── src/
│   ├── utils.py               # 公共函数（训练/评估/画图/保存/设备探测）
│   ├── mnist/
│   │   ├── explore.py          # 查看 MNIST 数据格式
│   │   ├── mlp.py              # 多层感知机
│   │   ├── cnn.py              # 卷积神经网络
│   │   └── cnn_optimized.py    # CNN + 数据增强/Dropout/学习率调度
│   └── cifar10/
│       ├── mlp.py
│       ├── cnn.py
│       └── cnn_optimized.py
├── images/            # 训练曲线图
├── models/            # 保存的模型 .pth 文件
├── README.md
└── requirements.txt
```

## 运行方法

```bash
pip install -r requirements.txt

# MNIST
python src/mnist/mlp.py
python src/mnist/cnn.py
python src/mnist/cnn_optimized.py

# CIFAR-10
python src/cifar10/mlp.py
python src/cifar10/cnn.py
python src/cifar10/cnn_optimized.py
```

## 实验结果

| | MLP | CNN | CNN + 优化 |
|---|---|---|---|
| MNIST | 93.74% | 97.90% | 97.26% |
| CIFAR-10 | 41.18% | 53.20% | 49.99% |

- CNN 对比 MLP：MNIST 提升 4.2%，CIFAR-10 提升 12%——数据越复杂，卷积优势越大
- 优化版在浅层网络上收益有限（甚至略降），说明模型还处于欠拟合状态，应先加深网络或增加 epoch
- 所有结果已用 `set_seed(42)` 固定随机种子，可复现

## 技术栈

Python 3 / PyTorch / torchvision / matplotlib / numpy
