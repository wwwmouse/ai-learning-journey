# 第三阶段学习报告

## 一、本周期学习内容

- PyTorch 基础：Tensor 操作、自动求导（autograd）
- 数据加载：torchvision 数据集下载、DataLoader 分批加载
- 神经网络搭建：nn.Linear（全连接层）、nn.Conv2d（卷积层）、ReLU 激活函数、MaxPool2d 池化
- 训练流程：前向传播 → 损失函数 → 反向传播 → 优化器更新参数
- 模型评估：model.eval()、torch.no_grad()、准确率计算
- 可视化：matplotlib 绘制 loss/acc 训练曲线
- 模型保存与加载：torch.save / load_state_dict
- 优化技巧：数据增强（RandomRotation、RandomHorizontalFlip）、Dropout、学习率调度（StepLR）
- 在 MNIST 和 CIFAR-10 两个数据集上对比了 MLP、CNN、CNN+优化三种方案

## 二、本周期项目成果

**GitHub 仓库**：[待补充]

**项目结构**：
```
pytorch-cnn-project/
├── src/mnist/    （MLP / CNN / CNN+优化）
├── src/cifar10/  （同上）
├── images/       （6 张训练曲线图）
└── models/       （6 个训练好的模型）
```

**实验结果**：

| | MLP | CNN | CNN + 优化 |
|---|---|---|---|
| MNIST | 93.63% | 98.00% | 97.51% |
| CIFAR-10 | 42.06% | 53.08% | 50.54% |

**运行方式**：
```bash
pip install -r requirements.txt
python src/mnist/cnn.py        # 或其他脚本
```

**截图**：见 images/ 目录下 6 张训练曲线图。

## 三、遇到的问题

1. **路径问题**：`root` 参数写相对路径，从不同目录运行时找不到数据。解决：用 `os.path.dirname(__file__)` 定位项目根再拼绝对路径。

2. **训练循环顺序记不住**：经常忘写 `optimizer.zero_grad()`，导致梯度累加、loss 不下降。解决：对照 numpy 手写版理解每步的因果关系，形成肌肉记忆。

3. **CNN 优化后准确率反而下降**：MNIST 从 98.00% 降到 97.51%，CIFAR-10 从 53.08% 降到 50.54%。分析后确认：两层小网络本身容量有限，尚未过拟合，Dropout 关掉神经元反而削弱了表达能力；数据增强增加了学习难度但 epoch 不够。

## 四、解决过程

以"优化版反而分数更低"为例：
1. 先怀疑代码有 bug，逐段对比优化版和原版的训练循环，确认逻辑一致。
2. 查阅 PyTorch 文档，理解 Dropout 只在训练时生效、测试时自动关闭，排除实现错误。
3. 回顾第二阶段学的"过拟合 vs 欠拟合"概念，意识到优化技巧是防过拟合的，模型还在欠拟合阶段不需要。
4. 得出报告中的分析结论：小模型 + 简单数据 → 优化技巧不是必需的。

## 五、下周期计划

- [ ] 学习更深网络架构（ResNet），在 CIFAR-10 上突破 80%
- [ ] 学习迁移学习：加载预训练模型微调
- [ ] 完成 1 个第四阶段方向探索项目（计算机视觉 / NLP 二选一）
- [ ] 每周至少提交 3 次代码到 GitHub