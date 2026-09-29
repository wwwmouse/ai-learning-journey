# 感受野（Receptive Field）—— 从公式到实测
#
# 配套笔记：sklearn-to-nn.md 的「2.2.3 感受野」
#   笔记讲"为什么"，本脚本负责"量给你看"。
#
# 运行： python notes/deep-learning/receptive_field.py
#
# 三件事：
#   1. 用递推公式算感受野
#   2. 用【梯度回传】实测感受野，和公式对拍        ← 核心，公式是推的，这里是量的
#   3. 算出 ResNet-18 在 CIFAR 上两种 stem 的感受野与分辨率，
#      解释为什么要把 stem 从 7×7 改成 3×3

import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'        # 中文字体
rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def save_plot(filename):
    """保存当前图表到 images/ 文件夹，并弹窗显示。"""
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  已保存: {filepath}")
    plt.show()


# ========================================
# 1. 递推公式
# ========================================

def trace_rf(layers, input_size):
    """
    逐层推进感受野（RF）和特征图尺寸。

    公式（记住这两行就够）：
        RF   = RF + (kernel - 1) × jump      ← 每过一层，RF 扩张一圈
        jump = jump × stride                 ← 每次降采样，后面的层"步子"翻倍

    Args:
        layers: [(名字, kernel, stride, padding), ...]
        input_size: 输入边长（假设正方形）

    Returns:
        [(名字, RF, jump, 特征图边长), ...]  每层之后的状态
    """
    rf, jump, size = 1, 1, input_size
    rows = []
    for name, k, s, p in layers:
        rf = rf + (k - 1) * jump
        jump = jump * s
        size = (size + 2 * p - k) // s + 1
        rows.append((name, rf, jump, size))
    return rows


def resnet18_spec(original_stem):
    """
    ResNet-18 的"感受野相关"骨架。

    original_stem=True  → 原版（7×7 s2 + maxpool 3×3 s2）
    original_stem=False → CIFAR 版（3×3 s1，maxpool 换成 Identity）

    每个 BasicBlock 是「conv3×3 → BN → ReLU → conv3×3 → BN → +x」，
    降采样只发生在 block1 的第一个 conv 上。

    Returns:
        (layers, stages)
        layers: [(名字, kernel, stride, padding), ...]
        stages: 与 layers 等长的阶段名列表，用于把两种 stem 的结果对齐比较
    """
    if original_stem:
        layers = [('conv1 7x7 s2', 7, 2, 3),
                  ('maxpool 3x3 s2', 3, 2, 1)]
        stages = ['stem', 'stem']
    else:
        # maxpool 换成了 nn.Identity()，什么都不做，所以不占一层
        layers = [('conv1 3x3 s1', 3, 1, 1)]
        stages = ['stem']

    for blk in ['layer1', 'layer2', 'layer3', 'layer4']:
        s1 = 1 if blk == 'layer1' else 2        # 只有 layer1 不降采样
        layers += [(f'{blk}.b1.conv1', 3, s1, 1),
                   (f'{blk}.b1.conv2', 3, 1, 1),
                   (f'{blk}.b2.conv1', 3, 1, 1),
                   (f'{blk}.b2.conv2', 3, 1, 1)]
        stages += [blk] * 4
    return layers, stages


def stage_landmarks(rows, stages):
    """
    取每个阶段【末尾】那一层的 (RF, 特征图边长)。

    两种 stem 的层数不同（原版 stem 占 2 层，改造后占 1 层），
    所以不能按行号对齐——必须按阶段名对齐，否则表格会串行。
    """
    out = {}
    for (name, rf, jump, size), st in zip(rows, stages):
        out[st] = (rf, size)        # 同一阶段后面覆盖前面 → 得到该阶段最后一层
    return out



# ========================================
# 2. 梯度回传实测感受野
# ========================================

def measure_rf(n_convs, input_size=15, k=3):
    """
    用梯度回传【实测】感受野。

    做法：
      1. 输入 x 设 requires_grad=True
      2. 前向，拿到输出特征图
      3. 挑【一个】输出像素，把它的梯度设成 1，其余全为 0
      4. backward
      5. 看 x.grad 上哪些像素拿到了非零梯度 ← 那就是这个输出像素的感受野

    这是"量"出来的，不是"推"出来的——所以它能验证公式对不对。

    Args:
        n_convs: 叠几个 k×k 卷积
        input_size: 输入边长（取大一点，让感受野完全落在图内，避开边界效应）
        k: 卷积核大小

    Returns:
        mask:    (input_size, input_size) 布尔数组，True 处就是感受野
        out_shape: 输出特征图尺寸
    """
    x = torch.randn(1, 1, input_size, input_size, requires_grad=True)
    # bias=False：偏置不影响对输入的梯度，去掉更干净
    net = nn.Sequential(*[nn.Conv2d(1, 1, k, padding=0, bias=False) for _ in range(n_convs)])

    out = net(x)
    ci, cj = out.shape[-2] // 2, out.shape[-1] // 2     # 取中心那个输出像素
    out[0, 0, ci, cj].backward()

    mask = (x.grad[0, 0].abs() > 0).numpy()
    return mask, (out.shape[-2], out.shape[-1])


def main():
    print("=" * 66)
    print("感受野（Receptive Field）")
    print("=" * 66)

    # ========================================
    # §1 定义
    # ========================================
    print("\n【§1 定义】")
    print("  输出特征图上的【某一个像素】，是由输入图像上【多大一块区域】算出来的。")
    print("  那块区域的大小，就是这个输出的感受野。\n")
    print("        输入 5×5                3×3 卷积后 3×3")
    print("      ┌───────────┐            ┌─────┐")
    print("      │ a b c d e │            │     │")
    print("      │ f g h i j │   ───►     │  X  │  ← 这个 X 由输入的 3×3 区域算出")
    print("      │ k l m n o │            │     │     所以它的感受野 = 3×3")
    print("      │ p q r s t │            └─────┘")
    print("      │ u v w x y │")
    print("      └───────────┘")
    print("\n  换句话说：感受野回答的是「这一层的每个输出，能看到原图多大范围」。")

    # ========================================
    # §2 公式
    # ========================================
    print("\n【§2 递推公式】")
    print("  RF   = RF + (kernel - 1) × jump")
    print("  jump = jump × stride")
    print("  初始：RF = 1, jump = 1\n")

    print("  手算三档（全是 3×3、stride=1）：")
    for n in [1, 2, 3]:
        rows = trace_rf([('conv', 3, 1, 1)] * n, 32)
        chain = " → ".join(str(r[1]) for r in rows)
        print(f"    {n} 个 3×3 卷积:  RF = {chain}   最终 {rows[-1][1]}×{rows[-1][1]}")

    print("\n  注意：3 个 3×3 的感受野 = 7×7，但这不等于「一个 7×7 卷积」——见 §5。")

    # ========================================
    # §3 实测（核心）
    # ========================================
    print("\n【§3 梯度回传实测 —— 公式是推的，这里是量的】")
    print("  做法：把某一个输出像素的梯度设成 1、其余为 0，反向传播；")
    print("        看输入上哪些像素拿到了非零梯度，那一片就是它的感受野。\n")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for idx, n in enumerate([1, 2, 3]):
        mask, out_shape = measure_rf(n, input_size=15, k=3)

        # 实测到的感受野大小（非零梯度的行数 / 列数）
        rows_hit = np.where(mask.any(axis=1))[0]
        cols_hit = np.where(mask.any(axis=0))[0]
        h = rows_hit.max() - rows_hit.min() + 1
        w = cols_hit.max() - cols_hit.min() + 1

        # 公式预测值
        predicted = trace_rf([('conv', 3, 1, 1)] * n, 15)[-1][1]

        print(f"  {n} 个 3×3 卷积:")
        print(f"      输出特征图 {out_shape[0]}×{out_shape[1]}，"
              f"实测感受野 {h}×{w}，公式预测 {predicted}×{predicted}，"
              f"{'✓ 一致' if h == predicted and w == predicted else '✗ 不一致！'}")

        ax = axes[idx]
        ax.imshow(mask, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{n} 个 3×3 卷积\n实测感受野 {h}×{w}', fontsize=11)
        ax.set_xlabel('输入列')
        ax.set_ylabel('输入行')

    fig.suptitle('梯度回传实测：感受野长什么样（蓝色 = 有梯度 = 被"看到"）', fontsize=13)
    plt.tight_layout()
    save_plot('receptive_field_heatmap.png')

    # ========================================
    # §4 ResNet-18：RF vs 分辨率
    # ========================================
    print("\n【§4 ResNet-18 在 CIFAR 上：感受野 vs 分辨率（最关键的一课）】")
    print("  同一份骨架，只换 stem，逐层算感受野和特征图尺寸：\n")

    layers_o, stages_o = resnet18_spec(original_stem=True)
    layers_n, stages_n = resnet18_spec(original_stem=False)
    rows_orig = trace_rf(layers_o, 32)
    rows_new = trace_rf(layers_n, 32)

    # 按【阶段名】对齐，而不是按行号——两种 stem 占的层数不同（2 层 vs 1 层）
    lm_o = stage_landmarks(rows_orig, stages_o)
    lm_n = stage_landmarks(rows_new, stages_n)

    print(f"  {'阶段':<10}{'原版 stem':>20}{'改造后 stem':>20}")
    print("  " + "-" * 50)
    for st in ['stem', 'layer1', 'layer2', 'layer3', 'layer4']:
        rf1, s1 = lm_o[st]
        rf2, s2 = lm_n[st]
        a = f"RF {rf1:>4}  图 {s1}×{s1}"
        b = f"RF {rf2:>4}  图 {s2}×{s2}"
        print(f"  {st:<10}{a:>20}{b:>20}")

    rf_o, size_o = rows_orig[-1][1], rows_orig[-1][3]
    rf_n, size_n = rows_new[-1][1], rows_new[-1][3]
    print(f"\n  结论（反直觉的那一条）：")
    print(f"    原版 stem  : 感受野 {rf_o}，layer4 特征图 {size_o}×{size_o}")
    print(f"    改造后 stem: 感受野 {rf_n}，layer4 特征图 {size_n}×{size_n}")
    print(f"\n    ★ 改造后感受野【反而更小】（{rf_n} vs {rf_o}），但准确率更高。")
    print(f"      因为输入只有 32×32：RF = {rf_n} 早就把整张图覆盖完了，再大也没用。")
    print(f"      真正决定成败的是【分辨率】—— layer4 还剩 {size_n}×{size_n} 还是塌成 {size_o}×{size_o}。")
    print(f"      原版到 layer4 只剩 1×1，全局平均池化退化成一个恒等操作，")
    print(f"      而 layer4 恰恰占了骨干约 75% 的参数。")
    print(f"\n    → 「看得够不够大」和「还记得多细」是两件事。")

    # 画图：RF 增长 + 分辨率衰减
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    xo = range(1, len(rows_orig) + 1)
    xn = range(1, len(rows_new) + 1)
    ax1.plot(xo, [r[1] for r in rows_orig], 'o-', color='#C44E52', label='原版 stem', markersize=4)
    ax1.plot(xn, [r[1] for r in rows_new], 's-', color='#4C72B0', label='改造后 stem', markersize=4)
    ax1.axhline(32, color='gray', linestyle='--', linewidth=1)
    ax1.text(1.2, 40, '图像只有 32×32', fontsize=9, color='gray')
    ax1.annotate(f'{rf_o}', (len(rows_orig), rf_o), textcoords="offset points",
                 xytext=(-24, 6), fontsize=10, color='#C44E52')
    ax1.annotate(f'{rf_n}', (len(rows_new), rf_n), textcoords="offset points",
                 xytext=(-24, -14), fontsize=10, color='#4C72B0')
    ax1.set_xlabel('层序号（含 stem）')
    ax1.set_ylabel('感受野大小（像素）')
    ax1.set_title('感受野增长：两条都远超 32')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(xo, [r[3] for r in rows_orig], 'o-', color='#C44E52', label='原版 stem', markersize=4)
    ax2.plot(xn, [r[3] for r in rows_new], 's-', color='#4C72B0', label='改造后 stem', markersize=4)
    ax2.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax2.annotate('塌成 1×1\n空间信息全丢', (len(rows_orig), 1),
                 textcoords="offset points", xytext=(-70, 18), fontsize=9, color='#C44E52')
    ax2.set_xlabel('层序号（含 stem）')
    ax2.set_ylabel('特征图边长')
    ax2.set_title('分辨率衰减：原版塌到 1×1')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_plot('receptive_field_growth.png')

    # ========================================
    # §5 两个 3×3 vs 一个 5×5
    # ========================================
    print("\n【§5 为什么用两个 3×3 而不是一个 5×5】")
    C = 64
    p_5x5 = 5 * 5 * C * C
    p_2x3x3 = 2 * (3 * 3 * C * C)
    p_3x3 = 3 * 3 * C * C
    print(f"  设通道数 C = {C}：\n")
    print(f"    1 个 5×5          感受野 5×5    参数 {p_5x5:>9,}")
    print(f"    2 个 3×3 叠起来   感受野 5×5    参数 {p_2x3x3:>9,}   "
          f"少 {(1 - p_2x3x3 / p_5x5) * 100:.0f}%")
    print(f"    1 个 3×3          感受野 3×3    参数 {p_3x3:>9,}")
    print(f"\n  感受野一样大，参数少 28%，而且中间多夹了一次 ReLU（非线性更强）。")
    print(f"  这就是 VGG 的结构论据，也是 ResNet 全部用 3×3 卷积的原因。")

    # ========================================
    # §6 降采样如何加速 RF 增长
    # ========================================
    print("\n【§6 stride / 池化 如何加速感受野增长】")
    print("  看公式里的 jump：每降采样一次，jump 翻倍，之后每一层扩张的幅度也翻倍。\n")
    no_pool = trace_rf([('conv 3x3 s1', 3, 1, 1)] * 6, 32)
    with_pool = trace_rf([('conv 3x3 s1', 3, 1, 1), ('pool 2x2 s2', 2, 2, 0)] * 3, 32)
    print(f"  6 层 3×3 s1，全不降采样  → 最终 RF = {no_pool[-1][1]}")
    print(f"  6 层 3×3 s1，每 1 层后跟一次 2×2 池化 → 最终 RF = {with_pool[-1][1]}")
    print(f"\n  层数一样，感受野差 {with_pool[-1][1] / no_pool[-1][1]:.1f} 倍 —— 因为 jump 被放大了。")
    print("  代价是分辨率掉得更快。所以降采样是「用分辨率换感受野」的交易。")

    print("\n" + "=" * 66)
    print("完成。生成 2 张图在 images/ 目录下：")
    print("  1. receptive_field_heatmap.png   梯度实测的感受野")
    print("  2. receptive_field_growth.png    两种 stem 的 RF 增长与分辨率衰减")
    print("=" * 66)


if __name__ == '__main__':
    main()
