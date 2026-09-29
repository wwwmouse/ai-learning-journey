# PyTorch 训练工程化

把一个训练脚本从「能跑」写成「能长时间跑、能中断续训、能复现、能换配置重跑」所需的全部工程手段。

> **前置**：先看 **sklearn-to-nn.md**。那篇讲"神经网络是什么、怎么训"（概念与 API 用法）；
> 本篇只讲"怎么让训练可靠地跑"，涉及基础概念时只给指针、不复述。
>
> **素材来源**：`projects/ResNet-18-cifar-100/src/train.py`，
> 实测数据来自该项目的真实运行：CIFAR-100 从零训练 ResNet-18，200 epoch，**测试准确率 80.10%**。

---

## 0. 为什么要"工程化"

### 0.1 小项目和大项目的分界线

同样是"训练一个 CNN"，下面这些差别决定了脚本要不要"工程化"：

| | 之前的 CNN 小项目 | 这个项目 |
|---|---|---|
| 脚本数 | 9 个共用 `utils.py` | 单文件自包含 |
| 单次耗时 | 十分钟以内 | **40 分钟** |
| 中断的代价 | 随手重跑 | **丢掉一小时 GPU 时间** |
| 改超参 | 直接改代码 | 命令行传参 |
| 结果可复现 | 短实验容易碰巧一致 | **200 epoch 必须主动固定** |
| 评估粒度 | 总体准确率够用 | 总体数字会掩盖问题 |

**转折点不是"模型变复杂了"，而是"一次运行变得昂贵了"。** 只要一次训练要花上一小时，下面这些就都从"可选"变成"必须"：

- 能不能中途停下来、之后再接着跑？（断点续训）
- 跑一半断电，成果会不会全没？（checkpoint 策略）
- 同样的代码再跑一遍，分数会不会变？（复现性）
- 想试另一个 lr，要不要改代码？（配置管理）
- 跑完之后，我还能知道中间发生了什么吗？（日志）

### 0.2 本文要讲的 8 件事

每一件都对应上面某一条需求：

| # | 手段 | 解决什么 | 章节 |
|---|---|---|---|
| 1 | argparse | 不改代码就能换配置 | I |
| 2 | warmup + cosine | 长训练的开局稳定与后期精调 | II |
| 3 | bf16 混合精度 | 显存与速度 | III |
| 4 | label smoothing / weight decay / 增强 | 过拟合 | IV |
| 5 | checkpoint + resume | 中断不白跑 | V |
| 6 | DataLoader 多进程 | CPU 别拖 GPU 后腿 | VI |
| 7 | 按类评估 | 总体准确率的盲区 | VII |
| 8 | 日志 + 曲线 | 事后能复盘 | VIII |

---

## I. 配置管理：argparse

### 1.1 三种写法，一个需求

需求是"想试不同的 lr / epoch，但不想每次改代码"。三种写法：

```python
# 写法一：硬编码 —— 最差，改一次要改代码
for epoch in range(200):
    ...

# 写法二：模块级大写常量 —— 好一些，改一处生效
EPOCHS = 200
LR = 0.1
for epoch in range(EPOCHS):
    ...

# 写法三：argparse —— 不改代码，命令行决定
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1)
```

写法二和写法三的**默认值完全等价**——`default=200` 就是"你不传时的值"。
区别只有一个：**写法三能从命令行临时改。**

> **常见误解**："`--epochs` 的默认值写在 `main()` 里，不够显眼，是不是不规范？"
> 不是。`default=200` 就是显式的 epoch 设定。判断"是否显式"的标准是
> **这个数字有没有一个名字**，而不是它写在哪一行。
> 真正的反面写法是 `for epoch in range(200)`——`200` 是个没有名字的魔法数字。

### 1.2 三行代码在做什么

```python
parser = argparse.ArgumentParser(description='CIFAR-100 ResNet-18 从零训练')
#        ↑ 建一个"命令行参数解析器"

parser.add_argument('--epochs', type=int, default=200)
#                    ↑名字        ↑类型     ↑不传时的默认值

args = parser.parse_args()
#      ↑ 读取你敲的命令，解析成一个对象
```

一次 `add_argument` 注册一个选项，做三件事：

1. 注册名字 `--epochs`
2. 用户**不传**时，值 = `default=200`
3. 用户传了 `--epochs 3` 时，用 `type=int` 转成 `3`，存进 `args.epochs`

```python
python src/train.py                  # args.epochs == 200
python src/train.py --epochs 3       # args.epochs == 3
```

### 1.3 本项目用到的 10 个参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--epochs` | 200 | 训练轮数 |
| `--batch-size` | 128 | 每批样本数 |
| `--lr` | 0.1 | 初始学习率（cosine 的起点） |
| `--warmup` | 5 | 线性 warmup 的 epoch 数 |
| `--weight-decay` | 5e-4 | L2 正则强度 |
| `--label-smoothing` | 0.1 | 标签平滑系数 |
| `--num-workers` | 4 | 数据加载的子进程数 |
| `--seed` | 42 | 随机种子 |
| `--resume` | （开关） | 从 checkpoint 续训 |
| `--show` | （开关） | 训练完弹出图表窗口 |

后两个是**开关**，不是"有值的参数"，所以用 `action='store_true'`：出现即为 `True`，不出现即 `False`，没有 `type=` 和 `default=`。

```python
parser.add_argument('--resume', action='store_true', help='从 checkpoint.pth 续训')
```

### 1.4 怎么查、怎么改

**查全部**：

```bash
python src/train.py --help
```

**三种常用组合**：

```bash
python src/train.py --epochs 3            # 冒烟测试：全量数据只跑 3 轮
python src/train.py                       # 正式训练，全用默认值
python src/train.py --resume              # 从断点续训
```

**启动时脚本会自报配置**，不用翻代码确认：

```
开始训练：200 epoch，lr=0.1，weight_decay=0.0005，label_smoothing=0.1
```

### 1.5 什么时候该用大写常量

判断标准：**这个值被几处引用？**

| 情况 | 用什么 | 本项目的例子 |
|---|---|---|
| 用户可能想临时改的值 | argparse | `--lr`、`--epochs` |
| "项目级事实"，被多处引用 | 大写常量 | `NUM_CLASSES = 100` |
| 用于自检的期望值 | 大写常量 | `EXPECTED_PARAMS = 11_220_132` |

`EXPECTED_PARAMS` 是个好例子——它不被"使用"，而是用来**验证**：

```python
n_params = sum(p.numel() for p in model.parameters())
if n_params != EXPECTED_PARAMS:
    print(f"  ⚠ 预期 {EXPECTED_PARAMS:,}，对不上说明结构改错了（先查 stem 和 fc）")
```

**3 行代码，抓住本项目最可能的 bug**（忘记改 stem）。忘了改的话，参数量会差 7,680，准确率掉 8~12 个点，而且**不报任何错**。

---

## II. 学习率调度：warmup + cosine

### 2.1 为什么不能用一个固定 lr

固定 lr 有个两难：

```
lr 太大  →  前期学得快，但后期在最优解附近来回跳，收敛不到底
lr 太小  →  稳定，但前期爬得慢，200 epoch 都不一定爬到位
```

**调度器的思路**：让 lr 随训练进程变化——**前期大、后期小**。

> 回顾 **sklearn-to-nn.md 的 2.4.2**：那里只讲了 `StepLR`（每 N 轮砍一半）。
> 本篇讲长训练更常用的组合。

### 2.2 三种调度器对比

| 调度器 | lr 形状 | 特点 | 适用 |
|---|---|---|---|
| `StepLR` | 阶梯下降 | 简单，但"何时砍、砍多少"要手调 | 短实验 |
| `CosineAnnealingLR` | 余弦平滑下降到底 | 无超参、下降平滑、末端极慢 | **长训练** |
| `LinearLR`（warmup） | 线性上升 | 开局用 | **配合 cosine** |

**本项目不用"二选一"，而是"先升后降"**：

```
lr
0.10 ┤              ╭──────────╮
     │            ╱              ╲
0.05 ┤          ╱                  ╲
     │        ╱                      ╲___
0.01 ┤  ╱────╯                            ╲___
     └──┬────┬─────────────────────────────────→ epoch
        1    5                                200
        warmup          cosine 退火
        线性上升         平滑下降
```

### 2.3 warmup 解决什么问题

**问题**：训练开头几个 epoch，BatchNorm 的 `running_mean/var` 还是初始值（0 和 1），统计量完全不准。
此时如果 lr 已经是最大值 0.1，**梯度方向基本是噪声，一步就能把参数踹飞**——表现是 loss 爆掉或震荡不收敛。

**warmup 的做法**：先让 lr 从 0.01 慢慢升到 0.1，给 BN 几个 epoch 的时间把统计量校准，同时也让权重先走到一个大致合理的位置。

**本项目的实测证据**——看 `logs/train_log.csv` 前 6 行：

```
epoch,lr,train_loss,test_acc
1,0.010000,4.2447,10.52
2,0.028000,3.9828,14.02
3,0.046000,3.7500,19.54
4,0.064000,3.4322,26.51
5,0.082000,3.1099,34.35
6,0.100000,2.8454,37.82
```

`0.01 → 0.028 → 0.046 → 0.064 → 0.082`——**每轮加 0.018**，正好是 `(0.1 − 0.01) / 5`。
而 `train_loss` 从 4.24 平滑降到 2.85，**没有任何爆掉或震荡**。warmup 生效了。

### 2.4 warmup 和 cosine 怎么接起来：`SequentialLR`

要两个调度器接力，代码要建三个对象：

```python
warmup_epochs = 5

# ① 前 5 轮：lr 从 0.1×0.1=0.01 线性升到 0.1
warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

# ② 第 6 轮起：从 0.1 余弦退火到 ~0
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

# ③ 把两个按"第 5 轮"这个界拼起来
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
)
```

`start_factor=0.1` 的含义是"起始 lr = 基准 lr × 0.1"，所以是 `0.1 × 0.1 = 0.01`。

> ★ **关键：训练循环里仍然只调一次 `scheduler.step()`。**
> `SequentialLR` 对外表现得像一个调度器，内部自己决定第几轮该交给谁。
> 不需要在循环里判断"现在是 warmup 还是 cosine"。

**验证 cosine 段**：日志里 epoch 200 的 lr = `0.000006`。
用公式核对：`T_max = 195`，第 194 步时
`lr = 0.1 × (1 + cos(π×194/195)) / 2 ≈ 0.1 × 0.00013 / 2 ≈ 6.5e-6` ✓ 对得上。

### 2.5 一个边界情况

如果 `--epochs` 比 `--warmup` 还小（比如冒烟测试 `--epochs 3 --warmup 5`），
`CosineAnnealingLR(T_max=负数)` 会直接报错。所以本项目先做了一次收缩：

```python
warmup_epochs = min(warmup, max(0, epochs - 1))
```

`max(0, epochs - 1)` 保证至少给 cosine 留 1 个 epoch；外层 `min` 保证 warmup 不超预算。
`epochs=3` 时结果是 `warmup_epochs = 2`，余弦跑 1 轮。**参数极端时脚本不崩，就是这条的价值。**

---

## III. 混合精度：bf16 autocast

### 3.1 为什么需要

默认情况下所有张量都是 `float32`（32 位）。但深度学习的**前向/反向对精度其实不敏感**——用 16 位算，结果几乎一样，但：

- 显存减半（激活值、临时张量）
- 速度提升（GPU 的 16 位算力通常是 32 位的数倍）

本项目的实测：

| 项 | 数值 |
|---|---|
| batch=128 前向显存 | 约 **193 MB** |
| GPU | RTX 5060 Laptop（sm_120，Blackwell） |
| bf16 支持 | `torch.cuda.is_bf16_supported() == True` |
| 整轮训练速度 | ≈ **15 秒/epoch** |

### 3.2 `autocast` 在做什么

它**不是一个转换开关**，而是"自动为每个算子挑合适的精度"：

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
    outputs = model(images)           # 卷积/矩阵乘 → 用 bf16
    loss = loss_fn(outputs, labels)   # 精度敏感的算子 → 自动保留 fp32
```

- **哪些算子降精度**：卷积、矩阵乘这类"算力大头"
- **哪些保持 fp32**：归约、softmax、loss、BN 统计——这些对精度敏感

**为什么只包 forward + loss，不包 backward？**

```python
with torch.autocast(...):        # ← 只包这里
    outputs = model(images)
    loss = loss_fn(outputs, labels)
loss.backward()                  # ← 在外面
optimizer.step()
```

因为 `autocast` 是**给前向计算打标记**的（记录每个输出该用什么精度）。反向传播会沿着这张计算图自动使用匹配的精度，不需要再包一层。

### 3.3 ★ 为什么这里不需要 `GradScaler`

如果你看过别的混合精度教程，一定会见到这段：

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(...):
    loss = ...
scaler.scale(loss).backward()    # 放大 loss
scaler.step(optimizer)           # 反缩放后再更新
scaler.update()
```

**本项目没有用 `GradScaler`，不是因为偷懒，而是因为用的是 bf16 而不是 fp16。** 区别在**指数位数**：

| | 指数位 | 能表示的最小正数 | 梯度下溢风险 |
|---|---|---|---|
| fp16（半精度） | 5 位 | ≈ 6e-5 | **高**——小梯度直接变 0 |
| **bf16**（bfloat16） | **8 位（和 fp32 相同）** | ≈ 1e-38 | **无** |
| fp32 | 8 位 | ≈ 1e-38 | 无 |

fp16 的指数位太少，**很小的梯度会被压成 0**，训练直接停止。`GradScaler` 就是干这个的：先把 loss 放大若干倍，让梯度落在 fp16 能表示的范围内，更新前再缩回去。

**bf16 的指数位和 fp32 一样（8 位），动态范围完全不缺**，代价只是尾数位少（精度略低）。所以：

> **bf16 不需要 `GradScaler`。少了三行代码，也少了一类 bug。**

怎么判断该用哪个？

```python
use_amp = (device.type == 'cuda')     # 本项目：只在 CUDA 上开
```

CUDA 上开 bf16 之前先确认硬件支持：

```python
torch.cuda.is_bf16_supported()        # Ampere（sm_80）以后都支持
```

### 3.4 `enabled=use_amp` 的作用

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
```

`enabled=False` 时 `autocast` 直接变成空操作，代码原样跑 fp32。
**这样同一份代码在 CPU 和 GPU 上都能跑**，不用写两套分支——CPU 上 `use_amp` 为 `False`，自动退回 fp32。

### 3.5 什么时候不该用

| 场景 | 原因 |
|---|---|
| 老显卡（sm_75 及以前，如 GTX 16 系） | 不支持 bf16；要用 fp16 + `GradScaler` |
| 需要严格数值复现 | 降精度会让结果和 fp32 有微小差异 |
| 训练本身不稳定（loss 频繁爆炸） | 先排除精度因素再开 |
| 模型极小（如 MNIST 上的 MLP） | 提速有限，不值得增加复杂度 |

---

## IV. 正则化三件套：各自防什么

### 4.1 ★ label smoothing：本项目最关键的一项

**机制**：普通交叉熵的目标是 one-hot（正确类 = 1，其余 = 0）。`label_smoothing=0.1` 把它软化：

```
100 类、ε = 0.1 时：
  正确类   = 1 − 0.1 + 0.1/100 = 0.901
  其余 99 类 = 0.1/100          = 0.001
```

**为什么这能防过拟合**：one-hot 目标要求模型对训练样本输出**绝对自信**（概率 1.0）。而训练集里必然有噪声和歧义样本——模型为了逼近 1.0，只能去记"这张图的具体像素"，这就是背答案。软化后模型只需输出"约 0.9 的自信"，**不必把参数推向极端**。

#### ★★ 一个必须记住的推论：loss 的下界不再是 0

这是最容易踩的认知陷阱。`label_smoothing` 之后，**即使模型完美预测，loss 也降不到 0**：

```
理论最小 loss = 目标分布的熵
             = −Σ qᵢ·ln(qᵢ)
             = −(0.901·ln0.901 + 99 × 0.001·ln0.001)
             ≈ 0.778
```

**本项目的实测证据**：

| 量 | 数值 |
|---|---|
| label smoothing 的 loss 下界 | **0.778** |
| 训练 200 epoch 后的 train_loss | **0.8252** |
| 差距 | **0.047** |

差 0.047 说明模型**已经把增强后的训练集几乎完全背下来了**——拟合已经饱和，所以再加 epoch 收益很小（实测后 50 轮只涨约 5 个点且趋缓）。

> ⚠ **由此得到一条重要结论：`train_loss` 的绝对值不能跨配置比较。**
>
> 我在 CIFAR-10 项目里见过 `train_loss = 0.0749`，看起来"低得多"——但那是因为**没加 label smoothing，下界是 0**。
> 0.0749（下界 0）和 0.8252（下界 0.778）**不在同一把尺子上**，谁"训得更好"完全比不出来。
>
> **要比较，必须用"离下界多远"，或者干脆只比测试准确率。**

### 4.2 weight decay

**是什么**：在参数更新时额外减去 `lr × wd × w`，让权重倾向于变小。

```
无 wd：  w -= lr × grad
有 wd：  w -= lr × (grad + wd × w)
```

**防的是什么**：权重过大会让输出对输入极其敏感——输入变一点点，预测就翻盘。这既是过拟合的表现，也让模型脆弱。把权重压小相当于**限制了模型的"爆发力"**，逼它用温和的函数拟合。

**本项目 `wd = 5e-4` 的依据**：1117 万参数 vs 5 万张训练图，参数量是数据的 200 多倍——必须约束。

### 4.3 数据增强

**是什么**：训练时对每张图做随机变换，让模型每轮看到的都不一样。

本项目用了三种，各自编码一条**不变性假设**：

| 变换 | 假设 | 为什么对 CIFAR 成立 |
|---|---|---|
| `RandomCrop(32, padding=4)` | 物体的**位置**不重要 | 猫在图片中间还是偏左，都还是猫 |
| `RandomHorizontalFlip()` | **左右镜像**不改变类别 | 朝左的猫和朝右的猫都是猫 |
| `RandomErasing(p=0.25)` | 局部被遮挡不影响判断 | 挡住猫的耳朵，还是能认出是猫 |

**注意 `RandomHorizontalFlip` 的不变性不是普适的**：交通标志、文字、左右手这类任务**不能翻转**——翻转后类别就变了。它成立是因为 CIFAR 的 100 个类恰好都满足。

> **为什么增强只加训练集？**
> 增强的作用是"人为制造训练难度"。测试集是考卷，考卷不能提前改。
> 测试集如果也做翻转平移，测出来的分数会**失真且和其它实验不可比**。
> 本项目只在 `train_transform` 里加增强，`test_transform` 只有 `ToTensor + Normalize`。

### 4.4 三者和"过拟合/欠拟合"的对应

> 承接 **sklearn-to-nn.md 的 3.3**：那里给出的诊断表是——
> - 训练 loss 高 + 测试 acc 低 → **欠拟合** → 加容量，**别加正则化**
> - 训练 loss 低 + 测试 acc 明显更低 → **过拟合** → 加正则化

三件套全部属于"过拟合时才该用"的**正则化手段**：

| 手段 | 防过拟合的机制 | 什么时候别用 |
|---|---|---|
| label smoothing | 不让模型对训练样本追求绝对自信 | 模型还在欠拟合时（会压制学习） |
| weight decay | 限制权重幅度，抑制极端函数 | 同上；太小没效果，太大欠拟合 |
| 数据增强 | 让模型无法靠记忆像素偷懒 | 同上（给还没学会走的人加负重） |

**本项目的诊断**：训练集≈满分、测试集 80.10%，差约 19 个点 → **确实过拟合，所以三件套都用**。
但准确率到最后一轮（epoch 200）还在微涨，**没出现"train 继续降、test 掉头向下"的恶化形态**——说明剂量合适，没有压制过度。

---

## V. 断点续训：checkpoint 里该存什么

### 5.1 只存 `state_dict` 不够

> 回顾 **sklearn-to-nn.md 的 1.5**：`torch.save(model.state_dict(), ...)` 只保存参数。
> 那是"训练完把结果存起来"的用法。**断点续训是另一回事。**

**区别**：训练结束后的模型只需要"参数"；但**想接着训**，就必须恢复到中断那一刻的**完整训练状态**：

```python
torch.save({
    'epoch':        epoch + 1,                 # ① 从哪继续
    'model':        model.state_dict(),        # ② 参数
    'optimizer':    optimizer.state_dict(),    # ③ 动量缓冲
    'scheduler':    scheduler.state_dict(),    # ④ lr 退火进度
    'best_acc':     best_acc,                  # ⑤ 历史最佳
    'train_losses': train_losses,              # ⑥ 曲线历史
    'test_accs':    test_accs,
}, ckpt_path)
```

**每一项缺了会怎样**：

| 缺了 | 后果 |
|---|---|
| `epoch` | 不知道从哪继续，只能从 0 重跑 |
| `model` | 参数回到随机初始化 |
| `optimizer` | **SGD 的动量缓冲丢失 → 优化器"失忆"**，需要几十轮才能重新积累动量；Adam 更严重（一阶/二阶矩全丢） |
| `scheduler` | lr 从 0.1 重新开始，cosine 进度归零 → 相当于把退火重置 |
| `best_acc` | 后续刷新最佳时会用 0 作比较，可能**用更差的结果覆盖掉 best.pth** |
| `train_losses` / `test_accs` | 续训后曲线从中间断开，最终图只剩后半段 |

> ★ **最容易忽略的是 `optimizer`。** 很多人以为"参数对了就能接着训"，
> 但 SGD + momentum 的 `v` 缓冲是**独立于参数**的一份状态，丢了就是丢了。

### 5.2 存 / 读的完整代码

**存**（每个 epoch 末尾覆盖）：

```python
torch.save({...}, ckpt_path)
```

**读**（启动时，如果 `--resume`）：

```python
if args.resume and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch']
    best_acc = ckpt['best_acc']
    train_losses = ckpt['train_losses']
    test_accs = ckpt['test_accs']
```

然后训练循环从 `start_epoch` 开始：

```python
for epoch in range(start_epoch, args.epochs):
```

### 5.3 ★ `weights_only` 的坑

`torch.load` 有个安全参数 `weights_only`。它控制"允不允许加载任意 Python 对象"：

| 值 | 能读什么 | 风险 |
|---|---|---|
| `True`（**PyTorch 2.6+ 的默认值**） | 只允许张量和基本容器 | 安全 |
| `False` | 任意 pickle 对象 | 理论上可执行恶意代码（加载不可信文件时危险） |

**坑在哪**：我们的 checkpoint 里存的是**字典**，里面有 `int`、`list`、`dict`。用 `weights_only=True` 读取时可能被拒绝。

**本机实测**（torch 2.13.0）：

```
weights_only=False 读取完整 checkpoint OK  epoch = 1  best_acc = 3.5
默认 weights_only=True 也能读（torch 版本较宽松）
纯 state_dict 用 weights_only=True 读取 OK，共 122 个张量
```

所以在新版本上默认值恰好也能读。但**行为随版本变化**，所以：

> **规则：读"纯参数"用 `weights_only=True`；读"完整训练状态"显式写 `weights_only=False`。**

本项目两处分别对应：

```python
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)   # 完整状态 → False
model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))  # 纯参数 → True
```

### 5.4 为什么 `best.pth` 和 `checkpoint.pth` 要分开存

两个文件服务于**两个不同的目的**：

| 文件 | 存什么 | 什么时候写 | 用途 |
|---|---|---|---|
| `resnet18_cifar100_best.pth` | **只有参数** | 每当测试准确率刷新纪录 | 最终交付、复现结果 |
| `checkpoint.pth` | **完整训练状态** | 每个 epoch 覆盖 | 中断后接着训 |

**为什么不合并成一个？**

1. **用途不同**：`best` 需要长期保留（哪怕后面跑崩了，最好结果还在）；`checkpoint` 会被反复覆盖，是"易失"的。
2. **大小不同**：`best` ≈ 45 MB（只有参数）；`checkpoint` ≈ 90 MB（参数 + 动量）。
3. **风险隔离**：如果只存一个文件，某次写入失败就同时丢了"最好结果"和"续训能力"。分开存，最坏情况也能保住 `best`。

**为什么需要 `best` 而不只是最后一个 epoch？**

本项目的日志：`epoch 199 = 79.94%`、`epoch 200 = 80.09%`，但**最佳是 epoch 196 = 80.10%**。
`best_acc = 80.10`，若只存最后一个 epoch 就会得到 80.09%——**损失虽小，但长训练末段掉几个点是常见现象，那时损失就大了**。

### 5.5 中断处理：`KeyboardInterrupt`

一个跑 50 分钟的脚本，用户随时可能按 Ctrl-C。默认行为是**直接抛异常退出**——曲线图没了（画图在训练循环之后），日志文件也可能没关闭。

本项目用 `try / except / finally` 兜住：

```python
interrupted = False
try:
    for epoch in range(start_epoch, args.epochs):
        ...                                    # 训练一个 epoch，存 checkpoint
except KeyboardInterrupt:
    interrupted = True
    print(f"\n⚠ 收到 Ctrl-C 中断（已完成 {len(train_losses)} 个 epoch）")
    print(f"  续训命令： python src/train.py --resume")
finally:
    log_file.close()                           # ★ 无论正常结束还是中断，都要关
```

三个细节：

1. **`log_file.close()` 必须在 `finally` 里**——放 `try` 末尾的话，中断时不会执行，文件句柄泄漏、缓冲区里的行可能没落盘。
2. **中断后照常画曲线**——已经跑完的 epoch 数据是有价值的，不该因为最后没跑完就全丢。
3. **给出续训命令**——中断时人往往是慌的，直接把命令打出来最省事。

**还要防止一种边界**：如果在第一个 epoch 中途就被中断，`best.pth` 还没生成过。所以最终分析前要检查：

```python
if not os.path.exists(best_path):
    print("还没产生 best 权重（第一个 epoch 未跑完），跳过最终分析。")
    return
```

### 5.6 加固：原子写（可选）

普通 `torch.save` 直接往目标文件写。**如果恰好在写入过程中断电，`checkpoint.pth` 会变成半截的坏文件**——既不能读，也覆盖掉了上一个好版本。

改成"先写临时文件，再原子替换"：

```python
tmp = ckpt_path + '.tmp'
torch.save({...}, tmp)
os.replace(tmp, ckpt_path)      # 原子操作：要么是旧的完整文件，要么是新的完整文件
```

`os.replace` 在同一个文件系统内是原子操作，**不存在"半截文件"的中间状态**。

> 本项目**没有加**这两行：断电概率很低，代码保持简洁更重要。
> 如果你要跑几天几夜的大训练，加上它很划算。

---

## VI. 数据加载的性能与多进程陷阱

### 6.1 三个参数各干什么

> 参数含义的速查表在 **sklearn-to-nn.md 的 2.1.4**，这里只讲"为什么这么设"和"坑在哪"。

```python
train_loader = DataLoader(
    base_train, batch_size=batch_size, shuffle=True, drop_last=True,
    num_workers=num_workers, pin_memory=True,
    persistent_workers=(num_workers > 0),
)
```

| 参数 | 本项目 | 为什么 |
|---|---|---|
| `num_workers` | 4 | 数据读取/增强是 **CPU 活**。单进程时 GPU 算完一批要**干等** CPU 准备下一批，利用率掉到很低。多进程并行准备，GPU 不空转 |
| `pin_memory` | True | 锁页内存，让 CPU→GPU 的拷贝可以用 DMA，**仅 GPU 训练时有用** |
| `persistent_workers` | 跟随 `num_workers` | 默认每个 epoch 结束后 worker 进程会销毁、下个 epoch 重建（4 个进程重建 200 次）。开启后**保持存活**，省掉重复的启动开销 |
| `drop_last` | True | 见 6.3 |
| `non_blocking` | 在 `.to(device)` 处 | 配合 `pin_memory`，让拷贝异步进行 |

`persistent_workers=(num_workers > 0)` 这个写法值得注意：**`persistent_workers=True` 在 `num_workers=0` 时会直接报错**（没有 worker 可保持）。所以必须跟随判断。

### 6.2 ★ Windows 上必须有的 `if __name__ == '__main__':`

这是本项目踩过的**最典型的坑**。报错原文：

```
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...
```

**原因**：Linux 用 `fork` 创建子进程（直接复制父进程内存），Windows 用 `spawn`（**启动一个全新的 Python 解释器，重新 import 主模块**）。

于是如果训练代码写在模块顶层：

```python
# ✗ 错误写法
train_loader = DataLoader(..., num_workers=4)
for epoch in range(200):        # 顶层代码
    ...
```

每个 worker 被创建时都会**重新 import 这个文件，又执行一遍顶层代码，又创建 4 个 worker……** 无限递归，Python 检测到就报上面那个错。

**正确写法**：把入口包起来

```python
if __name__ == '__main__':
    main()
```

`__name__` 在"直接运行"时是 `'__main__'`，在被 import 时是模块名。子进程 import 这个文件时 `__name__ != '__main__'`，所以**不会重复执行 `main()`**。

**实测复现**：我在验证环境时写的测试脚本忘了加这个 guard，结果终端里同一段输出**打印了 3 次**（1 次主进程 + 2 个 worker 各一次），并抛出上面那个 `RuntimeError`。这不是理论——是很显眼的现场。

> **判据**：只要用了 `num_workers > 0`，Windows 上就必须有 guard。
> Linux 上不加也能跑（fork 不重新 import），但**写上不会有任何坏处**，所以养成习惯。

### 6.3 `drop_last=True` 的取舍

```
50000 ÷ 128 = 390.625  → drop_last=True 时取 390 个 batch，丢掉最后 80 张
10000 ÷ 128 =  78.125  → 测试集 shuffle=False，不丢
```

| 选择 | 好处 | 坏处 |
|---|---|---|
| `drop_last=True` | 每个 batch 形状固定 `(128,3,32,32)`。**BN 的统计量依赖 batch 大小**，形状固定则统计稳定 | 每轮丢掉 80 张（0.16%） |
| `drop_last=False` | 不浪费数据 | 最后一批只有 80 张，BN 统计量突变，且形状不固定会让某些优化失效 |

**0.16% 的数据换取"每批形状一致"，很划算。** 训练集通常这么设，测试集不用（评估不更新 BN 统计量）。

---

## VII. 评估进阶：从总体准确率到每类准确率

### 7.1 总体准确率的盲区

本项目总体准确率 **80.10%**——听起来是个整齐的数字。但拆开看：

```
最好的类：road      97%
最差的类：man       53%
                    ↑ 差 44 个点
```

**总体数字是一个平均值，而平均值会掩盖分布。** 一个 80.10% 的模型可能同时包含"完全学会的类"和"基本没学会的类"——这两件事对下一步该做什么的指导意义完全不同。

### 7.2 用 `torch.bincount` 按类统计

朴素写法是遍历 100 个类，每类算一次：

```python
for c in range(100):
    mask = (labels == c)
    correct[c] += (predicted[mask] == c).sum()
    total[c] += mask.sum()
```

这样每个 batch 要循环 100 次（10000 张测试图 ≈ 79 个 batch → 约 7900 次循环）。**更快的写法是一次算完**：

```python
predicted = outputs.argmax(1).cpu()
hit = predicted == labels                          # 逐样本的"对/错"布尔向量
correct += torch.bincount(labels[hit], minlength=num_classes)   # 只数"猜对"的样本，按标签归类
total   += torch.bincount(labels, minlength=num_classes)        # 数全部样本，按标签归类
```

`torch.bincount(x, minlength=N)` 返回一个长度 N 的计数数组：**下标 i 的值 = x 中等于 i 的元素个数**。一行顶 100 行循环。

```python
accs = (100.0 * correct.float() / total.clamp(min=1)).tolist()
```

`total.clamp(min=1)` 是防御性写法：万一某个类在测试集里一张都没有，除零会得到 `nan` 污染整张图。

### 7.3 本项目的实证：为什么要按类看

拆开之后，失败呈现出**两条清晰的规律**（详见项目 README）：

**规律一：`people` 粗类几乎全军覆没。**
`man`(53%)、`boy`(57%)、`girl`(58%) 拿下最差榜第 1、3、4 名。而这一组（baby / boy / girl / man / woman）**正是数据探索时 `cifar100_confusing_classes.png` 展示的那一组**——32×32 像素下性别与年龄的视觉差异本来就极小。

**规律二：同一粗类内部可以差 31 个点。**
`household furniture`（bed / chair / couch / table / wardrobe）里，`wardrobe` **97%**（全场最好之一），`couch` **66%**（最差榜第 10）。

> **结论：决定难度的是"类内视觉相似度"，不是"粗类归属"。**
> 粗类只是数据集的组织方式，模型眼里没有这么整齐的层级。

**如果只看 80.10% 这个数字，这两条规律一条也发现不了。** 这就是"按类评估"的价值。

---

## VIII. 日志与可视化

### 8.1 CSV 日志：记什么

`logs/train_log.csv`，4 列：

```csv
epoch,lr,train_loss,test_acc
1,0.010000,4.2447,10.52
2,0.028000,3.9828,14.02
...
200,0.000006,0.8252,80.09
```

| 列 | 含义 |
|---|---|
| `epoch` | 第几轮 |
| `lr` | **该轮开始时**的学习率 |
| `train_loss` | 该轮所有 batch loss 的**平均值** |
| `test_acc` | 该轮结束后的测试准确率 |

**两个设计细节**：

1. **每写一行就 `flush()`**

```python
log_writer.writerow([...])
log_file.flush()
```

不 flush 的话，数据会攒在缓冲区里，可能几分钟才落盘一次。**正在跑的训练你想中途打开 CSV 看进度**——flush 让这件事成立。

2. **`--resume` 时是追加而不是重写**

```python
write_header = not (args.resume and os.path.exists(log_path))
```

续训时文件已存在，不能再写一遍表头，否则 CSV 中间会多出一行标题，解析会出错。

### 8.2 曲线怎么画：为什么 200 个点不需要 marker

**这里有个我踩过的坑。** 之前的 CIFAR-10 项目里，画图函数是这么写的：

```python
ax1.plot(range(1, epochs + 1), train_losses, marker='o', color='blue')
#                                              ^^^^^^^^^^ 每个点画一个圆点
```

50 个 epoch 时，50 个圆点叠在曲线上，**看起来非常乱**——当时我以为"点太多了，得隔几个画一次"。

**其实问题不在点的数量，在 marker。** 曲线本身只需要连线：

```python
ax1.plot(range(1, epochs + 1), train_losses, color='blue', linewidth=1.5)
```

**200 个点连成一条细线，非常干净**（见项目 README 的曲线图）。所以：

> **不需要"每 5 个 epoch 画一次"或"保存中间模型来画图"。**
> 画图用的是内存里的两个 `list`（200 个浮点数，几十 KB），和模型文件毫无关系。

**顺带一个推论**：如果训练崩了、曲线图没了怎么办？——**CSV 里有每个 epoch 的 loss/acc**，随时能重新画。曲线数据是独立于模型保存的。

### 8.3 本项目曲线怎么读

![训练曲线](../../projects/ResNet-18-cifar-100/images/resnet18_cifar100_curves.png)

两个值得注意的形态：

**1. 准确率曲线前期很毛糙。**
`epoch 18 = 57.71%`，`epoch 19 = 51.29%`——单轮摆动 ±4 个点。这是正常的：

- 早期参数变化快，模型每轮都在明显改变
- 测试集只有 10,000 张（**每类仅 100 张**），每类的统计噪声本身就大

后期曲线变得非常平滑，说明模型稳定了。

**2. loss 曲线单调平滑下降，没有"掉到 0"的陡降。**
这正是 `label_smoothing` 的形态特征——**下界被抬到了 0.778**，loss 物理上到不了 0（见 4.1）。

**怎么用这两条曲线诊断**：

| 形态 | 含义 | 该做什么 |
|---|---|---|
| train loss 高 + test acc 低 | 欠拟合 | 加容量、加 epoch，**别加正则化** |
| train loss 持续降 + test acc 掉头向下 | 严重过拟合 | 加正则化 |
| train loss 贴下界 + test acc 走平 | **收敛，剂量合适** | 停（本项目就是这种） |
| loss 震荡不降 | lr 太大 或 忘写 `zero_grad` | 查超参和代码 |

---

## IX. 常见坑和排错清单

> **边界说明**：这里只收**工程类**坑（长训练、多进程、复现性、续训）。
> **概念/用法类**坑（忘写 `zero_grad`、忘切 `eval()`、维度不匹配等）在 **sklearn-to-nn.md 的 V 章**。

| 坑 | 现象 | 原因 | 解决 |
|---|---|---|---|
| **Windows 多进程没写 guard** | `An attempt has been made to start a new process...`；输出重复多次 | Windows 用 spawn，子进程重新 import 主模块 | 入口包在 `if __name__ == '__main__':` 里 |
| **`weights_only` 读不出 checkpoint** | `torch.load` 报错或拿不到 optimizer | PyTorch 2.6+ 默认 `weights_only=True`，拒绝非张量内容 | 读完整训练状态显式传 `weights_only=False` |
| **只存 `state_dict` 就续训** | 动量丢了、lr 从头开始、曲线断开 | 优化器和调度器状态没存 | checkpoint 存 6 项：epoch/model/optimizer/scheduler/best_acc/历史 |
| **`persistent_workers=True` 配 `num_workers=0`** | 直接报错 | 没有 worker 可以保持 | `persistent_workers=(num_workers > 0)` |
| **只存最后一个 epoch** | 最终分数比最佳低 | 长训练末段常掉点（本项目 best 在 196，不是 200） | best 和 checkpoint 分开存 |
| **中断后曲线全丢** | Ctrl-C 后没有图 | 画图代码在训练循环之后，异常直接退出 | `try/except KeyboardInterrupt` + `finally` 关文件 |
| **loss 跨配置比较** | 误判"哪个训得好" | label smoothing 会抬高 loss 下界（本项目下界 0.778） | 比"离下界多远"，或只比测试准确率 |
| **换了 cuDNN 设置还对比旧结果** | 分数有小差异，误以为是改动生效 | `cudnn.benchmark` 改变算法选择 | 换设置后重跑基线，别跨设置对比 |
| **忘了 `scheduler.step()`** | lr 一直不变，后期收敛不到底 | 调度器只是"算新 lr"，要主动让它算 | epoch 循环末尾调一次 |
| **`scheduler.step()` 写在 batch 循环里** | lr 掉得飞快 | 它按"调用次数"推进，不是按 epoch | 放在 epoch 循环末尾 |
| **测试集加了数据增强** | 分数虚高且不可比 | 增强只属于训练集 | 测试集只做 `ToTensor + Normalize` |
| **每类准确率除零** | 图里出现 `nan` | 某个类测试样本为 0 | `total.clamp(min=1)` |
| **CSV 续训后多一行表头** | 解析 CSV 报错 | 续训时文件已存在还写了表头 | `write_header = not (resume and exists)` |

---

## 附录 A. 最小可复用模板

把本项目的骨架剥出来，去掉 CIFAR-100 专有部分。**换数据集只需改 `build_loaders` 和 `build_model`。**

```python
import argparse, csv, os, random
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

DEVICE_AMP = None

# ===== 1. 复现性 =====
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 关掉算法自动搜索 → 可复现
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ===== 2. 数据（改成你的数据集）=====
def build_loaders(batch_size, num_workers):
    train_set, test_set = ..., ...   # 你的 Dataset
    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=(num_workers > 0))
    return (DataLoader(train_set, batch_size, shuffle=True, drop_last=True, **common),
            DataLoader(test_set,  batch_size, shuffle=False, **common))

# ===== 3. 模型（改成你的网络）=====
def build_model(num_classes):
    return ...

# ===== 4. 训练 / 评估 =====
def train_one_epoch(model, loader, loss_fn, optimizer, device, use_amp):
    model.train(); total = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = loss_fn(model(x), y)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total

# ===== 5. 主流程 =====
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    use_amp = (device.type == 'cuda')

    train_loader, test_loader = build_loaders(args.batch_size, args.num_workers)
    model = build_model(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)

    # warmup + cosine（含 epochs < warmup 的边界处理）
    we = min(args.warmup, max(0, args.epochs - 1))
    if we > 0:
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=we),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - we)),
        ], milestones=[we])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path, best_path = 'checkpoint.pth', 'best.pth'
    start_epoch, best_acc, history = 0, 0.0, []
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch, best_acc, history = ck['epoch'], ck['best_acc'], ck['history']

    try:
        for epoch in range(start_epoch, args.epochs):
            lr = optimizer.param_groups[0]['lr']
            loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, use_amp)
            acc = evaluate(model, test_loader, device, use_amp)
            scheduler.step()
            history.append((epoch + 1, lr, loss, acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_path)
            print(f"epoch {epoch+1:3d}: lr={lr:.4f} loss={loss:.4f} acc={acc:.2f}%")
            tmp = ckpt_path + '.tmp'                      # 原子写
            torch.save({'epoch': epoch + 1, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'best_acc': best_acc, 'history': history}, tmp)
            os.replace(tmp, ckpt_path)
    except KeyboardInterrupt:
        print(f"\n中断，续训： python {__file__} --resume")
    print(f"最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':     # ★ Windows 多进程必需
    main()
```

**从模板到本项目，只多了三样东西**：`plot_curves` / `per_class_accuracy` / CSV 日志。
它们都是"记录产出"，不影响训练本身——需要时再补即可。

---

## 相关笔记

- **sklearn-to-nn.md** —— 概念与 API 用法（本文的前置）
  - 2.1.4 DataLoader 参数速查
  - 2.3 损失函数与交叉熵
  - 2.4 优化器与 `StepLR`
  - 2.6 预训练与迁移学习
  - V 章 概念/用法类坑清单
- **项目**：`projects/ResNet-18-cifar-100/` —— 本文所有实测数据的来源
