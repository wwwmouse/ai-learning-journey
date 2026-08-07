# PyTorch CNN 图像分类项目

第三阶段学习项目，在 MNIST 和 CIFAR-10 上对比 MLP、CNN、优化版 CNN。

## 项目结构

```
├── data/              # 数据集（MNIST + CIFAR-10）
├── src/
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
| MNIST | 93.63% | 98.00% | 97.51% |
| CIFAR-10 | 42.06% | 53.08% | 50.54% |

- CNN 对比 MLP：MNIST 提升 4%，CIFAR-10 提升 11%——数据越复杂，卷积优势越大
- 优化版在简单模型上收益有限，后续应尝试更深网络或迁移学习

## 技术栈

Python 3 / PyTorch / torchvision / matplotlib / numpy
