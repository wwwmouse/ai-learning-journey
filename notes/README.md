# 笔记索引

这里放**跨项目的知识**。项目的代码、数据、结果在 [`../projects/`](../projects/)。

---

## 收录判据

> **删掉某个项目，这条内容还有没有价值？**
>
> - **有** → 放这里（它是跨项目的知识）
> - **没有** → 放项目里（它是那个项目的产物）

举例：`80.10%` 这个分数属于项目；"label smoothing 会抬高 loss 下界"属于这里。

---

## 一、工具参考（查语法，不用从头读）

| 笔记 | 规模 | 内容 | 什么时候看 |
|---|---|---|---|
| [numpy.md](python-foundation/numpy.md) | 446 行 | ndarray 特性 / 创建 / 索引切片 / 广播 / 常用函数 | 写数组运算忘了 API |
| [pandas.md](python-foundation/pandas.md) | 907 行 | Series / DataFrame / 数据处理 | 洗表格数据 |
| [matplotlib.md](python-foundation/matplotlib.md) | 129 行 | 图表选择 / 图配置 / 折线图 / 条形图 | 画图调样式 |
| [numpy-exercises.md](python-foundation/numpy-exercises.md) | 184 行 | 9 道练习（温度 / 成绩 / 矩阵 / 广播 / 布尔索引…） | 学完 numpy 想动手 |
| [pandas-exercises.md](python-foundation/pandas-exercises.md) | 103 行 | Series 3 题 + DataFrame 2 题 | 学完 pandas 想动手 |

## 二、机器学习

| 笔记 | 规模 | 内容 | 依据项目 | 什么时候看 |
|---|---|---|---|---|
| [ml-scikitlearn.md](python-foundation/ml-scikitlearn.md) | 1026 行 | ML 完整 7 阶段流程 / 重点函数详解 / 主流模型对比 | titanic_project | 第一次做传统 ML 分类任务 |

## 三、深度学习

| 笔记 | 规模 | 内容 | 依据项目 | 什么时候看 |
|---|---|---|---|---|
| [sklearn-to-nn.md](python-foundation/sklearn-to-nn.md) | 1272 行 | 神经网络概念与原理 + PyTorch API 用法（数据 / 网络层 / 损失 / 优化器 / 预训练） | titanic + pytorch-cnn-project | 从传统 ML 转到神经网络 |
| [pytorch-engineering.md](python-foundation/pytorch-engineering.md) | 1028 行 | 长训练工程化：配置 / 调度 / 混合精度 / 正则化 / 断点续训 / 多进程 / 评估 / 日志 | pytorch-cnn-project + ResNet-18-cifar-100 | 训练要跑半小时以上 |

## 四、环境与工具链

| 笔记 | 规模 | 内容 |
|---|---|---|
| [claude-code-deepseek-setup.md](others/claude-code-deepseek-setup.md) | 129 行 | Claude Code + DeepSeek V4 部署（Windows） |
| [vscode-git-github-setup-guide.md](others/vscode-git-github-setup-guide.md) | 310 行 | 本地仓库关联 GitHub 全流程 |

---

## 推荐阅读顺序

```
numpy → pandas → matplotlib
   ↓
ml-scikitlearn          传统 ML，理解 fit / predict 的边界
   ↓
sklearn-to-nn           神经网络是什么、为什么这样设计
   ↓
pytorch-engineering     怎么让它可靠地跑一小时
```

---

## 写作约定

保持一致的 6 条（从已有笔记里提取，新笔记照此写）：

1. **H1 标题 + 一句话说明覆盖范围**，然后 `---`
2. **`## 0.` 开场，`## I.` `## II.` 大写罗马数字分章**，章内用 `### 1.1` 编号
3. **三件套**：对比表格 + `>` 引用块放关键结论 + 代码里逐行中文注释
4. **以「常见坑和排错清单」收尾**（表格：坑 / 现象 / 原因 / 解决）
5. **第一人称学习口吻**，主动标注不确定处（例："这个争论我还没深读，先按原论文记"）
6. **标注依据**：写清这条结论来自哪个项目的哪次实测，不写"据说"

---

## 与其他目录的关系

| 目录 | 放什么 | 判据 |
|---|---|---|
| `notes/` | 跨项目的知识 | 删掉某个项目，这条内容还有价值 |
| `projects/` | 代码、数据、结果、项目级结论 | 删掉这个项目，这条内容就没意义了 |
| `others/` | 个人杂项（已在 `.gitignore` 中） | — |

**链接约定**：

- 项目 README 末尾的「相关笔记」→ 相对路径指向这里
- 笔记末尾的「相关项目」→ 相对路径指回 `projects/`
- **正文里不互相复述内容**，只在实际需要时给一行指针
