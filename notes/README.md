# 笔记索引

跨项目的知识放这里；项目产物在 [`../projects/`](../projects/)。

> **收录判据**：删掉某个项目，这条内容还有价值吗？**有** → 这里；**没有** → 项目里。

---

## python-foundation —— 工具与练习

| 文件 | 规模 | 讲什么 |
|---|---|---|
| [numpy.md](python-foundation/numpy.md) | 446 行 | ndarray 特性 / 索引切片 / 广播 / 常用函数 |
| [pandas.md](python-foundation/pandas.md) | 907 行 | Series / DataFrame / 数据处理 |
| [matplotlib.md](python-foundation/matplotlib.md) | 129 行 | 图表选择 / 图配置 / 折线图 / 条形图 |
| [numpy-exercises.md](python-foundation/numpy-exercises.md) | 184 行 | 9 道 numpy 练习 |
| [pandas-exercises.md](python-foundation/pandas-exercises.md) | 103 行 | Series 3 题 + DataFrame 2 题 |

## machine-learning —— 传统 ML

| 文件 | 规模 | 讲什么 | 依据项目 |
|---|---|---|---|
| [ml-scikitlearn.md](machine-learning/ml-scikitlearn.md) | 1026 行 | 完整 7 阶段流程 / 重点函数详解 / 主流模型对比 | titanic_project |

## deep-learning —— 神经网络

| 文件 | 规模 | 讲什么 | 依据项目 |
|---|---|---|---|
| [sklearn-to-nn.md](deep-learning/sklearn-to-nn.md) | 1390 行 | 概念与原理 + PyTorch API（数据 / 网络层 / 损失 / 优化器 / 预训练 / 感受野） | titanic + pytorch-cnn-project |
| [pytorch-engineering.md](deep-learning/pytorch-engineering.md) | 1028 行 | 长训练工程化（配置 / 调度 / 混合精度 / 正则化 / 续训 / 多进程 / 日志） | pytorch-cnn-project + ResNet-18-cifar-100 |
| [receptive_field.py](deep-learning/receptive_field.py) | 332 行 | **配套脚本**：梯度回传实测感受野 + 两种 stem 的 RF/分辨率对照 | — |

> `receptive_field.py` 与 `sklearn-to-nn.md` 的 **2.2.3 感受野**是一对：
> 笔记讲"为什么"，脚本负责"量给你看"。运行：
> `python notes/deep-learning/receptive_field.py`

## others

| 文件 | 规模 | 讲什么 |
|---|---|---|
| [claude-code-deepseek-setup.md](others/claude-code-deepseek-setup.md) | 129 行 | Claude Code + DeepSeek V4 部署（Windows） |
| [vscode-git-github-setup-guide.md](others/vscode-git-github-setup-guide.md) | 310 行 | 本地仓库关联 GitHub 全流程 |

---

## 看哪篇？

| 你的问题 | 看哪篇 |
|---|---|
| sklearn 怎么 `fit` / `predict`？模型怎么选？ | `machine-learning/ml-scikitlearn.md` |
| 卷积是什么？为什么要 BN？交叉熵在算什么？感受野多大？ | `deep-learning/sklearn-to-nn.md` |
| 训练脚本怎么扛住 1 小时、断电不白跑、能复现？ | `deep-learning/pytorch-engineering.md` |

**推荐顺序**：`python-foundation` → `machine-learning` → `deep-learning`

---

## 写作约定

新笔记照此写，保持一致：

1. **H1 标题 + 一句话说明覆盖范围**，然后 `---`
2. **`## 0.` 开场，`## I.` `## II.` 用大写罗马数字分章**，章内用 `### 1.1` 编号
3. **对比表格 + `>` 引用块放关键结论 + 代码里逐行中文注释**
4. **以「常见坑和排错清单」收尾**（表格：坑 / 现象 / 原因 / 解决）
5. **第一人称学习口吻**，主动标注不确定处
6. **标注依据**：写清结论来自哪个项目的哪次实测，不写"据说"
