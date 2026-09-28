# CIFAR-100 数据探索
#
# 目的：第一次接触 CIFAR-100，先把它"看明白"，再动手训练。
#
# 这个脚本回答四个问题：
#   1. 数据长什么样？—— 规模 / 尺寸 / 类别数
#   2. 每类有多少张？—— 100 类各 500 张，和 CIFAR-10 的差距在哪
#   3. 100 个细类是怎么归成 20 个粗类的？
#   4. CIFAR-100 到底难在哪？—— 看"32×32 下长得很像"的类别
#
# 终端输出：数据集概况 / 粗类-细类对应表 / 实测 mean·std
# 生成图片：images/ 下 3 张
#
# 运行： python src/explore_data.py

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torchvision import datasets

rcParams['font.family'] = 'SimHei'        # 中文字体，否则标题显示成方块
rcParams['axes.unicode_minus'] = False    # 负号正常显示

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def save_plot(filename):
    """保存当前图表到 images/ 文件夹"""
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"  已保存: {filepath}")


def main():
    print("正在加载 CIFAR-100 ...")
    train_set = datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
    test_set = datasets.CIFAR100(root=DATA_DIR, train=False, download=True)

    data = train_set.data                          # (50000, 32, 32, 3) uint8
    fine = np.array(train_set.targets)             # 细类标签，0~99

    # 粗类标签要直接读原始 pickle：
    #   torchvision 的 CIFAR100 只保留 fine label，实例上【没有】coarse_targets 属性
    #   （CIFAR100 类里只定义了 base_folder / url / train_list / test_list / meta）。
    #   数据的存放顺序和 torchvision 读进来的一致，所以下标可以直接对齐。
    train_pkl = os.path.join(DATA_DIR, 'cifar-100-python', 'train')
    with open(train_pkl, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    coarse = np.array(raw['coarse_labels'])        # 粗类标签，0~19

    assert np.array_equal(fine, np.array(raw['fine_labels'])), \
        "细类标签和 pickle 对不上，说明数据被改过"

    # 元信息（类名）也在 pickle 里——顺便理解 CIFAR 的存储格式
    meta_path = os.path.join(DATA_DIR, 'cifar-100-python', 'meta')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')
    fine_names = meta['fine_label_names']       # 100 个类名
    coarse_names = meta['coarse_label_names']   # 20 个粗类名

    # ========================================
    # 1. 数据集概况
    # ========================================
    print("\n【1. 数据集概况】")
    print(f"  训练集: {len(data)} 张    测试集: {len(test_set)} 张")
    print(f"  图片尺寸: {data.shape[1:]}   (高, 宽, 通道)")
    print(f"  细类(fine): {len(fine_names)} 类     粗类(coarse): {len(coarse_names)} 类")
    print(f"  像素: {data.dtype} 类型，取值 {data.min()}~{data.max()}")
    print(f"  随机猜的基线: {100 / len(fine_names):.2f}%   ← 100 类的数据集，瞎猜只有 1 分")

    # ========================================
    # 2. 每类样本数
    # ========================================
    print("\n【2. 每类样本数】")
    counts = np.bincount(fine, minlength=100)
    print(f"  最少 {counts.min()} 张 / 最多 {counts.max()} 张 / 平均 {counts.mean():.1f} 张")
    print("  → 100 类各 500 张，完全均衡，所以本项目不需要处理类别不平衡")
    print("  → 但对比 CIFAR-10 的每类 5000 张，每类的训练证据只有它的 1/10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # 左图：和 CIFAR-10 的每类样本数对比
    ax1.bar(['CIFAR-10\n(10 类)', 'CIFAR-100\n(100 类)'], [5000, 500],
            color=['#4C72B0', '#C44E52'], width=0.5)
    for i, v in enumerate([5000, 500]):
        ax1.text(i, v + 100, str(v), ha='center', fontsize=11)
    ax1.set_ylabel('每类可用训练样本数')
    ax1.set_title('每类训练证据少了 10 倍')
    ax1.set_ylim(0, 5800)

    # 右图：100 个类各自的样本数——一条平线，说明"均衡"
    ax2.bar(range(100), counts, color='steelblue', width=1.0)
    ax2.axhline(500, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('类别编号 0~99')
    ax2.set_ylabel('样本数')
    ax2.set_title('CIFAR-100 类别分布（完全均衡）')
    ax2.set_ylim(0, 600)

    plt.tight_layout()
    save_plot('cifar100_class_balance.png')
    plt.show()

    # ========================================
    # 3. 粗类 → 细类 的对应关系
    # ========================================
    print("\n【3. 粗类(superclass) → 细类 对应表】")
    print("  CIFAR-100 的 100 个细类可以归成 20 个粗类，每个粗类下 5 个细类。")
    print("  数据集自带两套标签：fine_labels（100 类）和 coarse_labels（20 类）。")

    # 细类 → 粗类 的映射，从数据里实测出来，不靠猜
    fine_to_coarse = {}
    for f, c in zip(fine, coarse):
        fine_to_coarse[int(f)] = int(c)

    print("\n  ⚠ 一个容易踩的坑：粗类在编号上【不连续】。")
    print("    粗类 0 的成员编号是 4/30/55/72/95，不是 0~4。")
    print("    所以不能用 `fine_label // 5` 推算粗类，必须查下面这张表。\n")

    for c in range(20):
        ids = sorted([f for f, cc in fine_to_coarse.items() if cc == c])
        members = ', '.join(fine_names[f] for f in ids)
        print(f"  {c:2d}. {coarse_names[c]:<32s} → {members}")

    # ========================================
    # 4. 每个细类长什么样：100 类抽样总览
    # ========================================
    print("\n【4. 100 个细类抽样总览】")
    print("  每个类随机取 1 张，拼成 10×10。这是亲眼看到“每类只有 500 张”最直接的方式")

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(10, 10, figsize=(15, 16))
    for c in range(100):
        class_idx = np.where(fine == c)[0]
        pick = class_idx[rng.integers(len(class_idx))]
        ax = axes[c // 10, c % 10]
        ax.imshow(data[pick])
        ax.set_title(fine_names[c], fontsize=7)
        ax.axis('off')
    fig.suptitle('CIFAR-100 全部 100 个细类（每类随机 1 张）', fontsize=15)
    plt.tight_layout()
    save_plot('cifar100_sample_grid.png')
    plt.show()

    # ========================================
    # 5. 到底难在哪：看长相极像的类别
    # ========================================
    print("\n【5. 易混淆类别展示】")
    print('  取粗类 "people" 下的 5 个细类，各 8 张：baby / boy / girl / man / woman')
    print("  在 32×32 像素下，这几类的视觉差异极小——这是 CIFAR-100 难度的来源之一")

    people = ['baby', 'boy', 'girl', 'man', 'woman']
    fig, axes = plt.subplots(5, 8, figsize=(14, 9))
    for row, name in enumerate(people):
        # 先取出这个类的【全部】下标，再从中挑 8 张
        # （数据在整体上是打乱的，同一个标签的样本并不连续存放）
        class_idx = np.where(fine == fine_names.index(name))[0]
        for col in range(8):
            ax = axes[row, col]
            ax.imshow(data[class_idx[col]])
            ax.axis('off')
            if col == 0:
                ax.set_title(name, fontsize=11, loc='left')
    fig.suptitle('32×32 下的 "people" 粗类：baby / boy / girl / man / woman', fontsize=13)
    plt.tight_layout()
    save_plot('cifar100_confusing_classes.png')
    plt.show()

    # ========================================
    # 6. 实测 mean / std（用来核对 train.py 里的常量）
    # ========================================
    print("\n【6. 实测归一化统计量】")
    mean = data.mean(axis=(0, 1, 2)) / 255.0   # 在 (样本, 高, 宽) 上统计，得到每个通道一个值
    std = data.std(axis=(0, 1, 2)) / 255.0
    print(f"  mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    #   mean = [0.5071, 0.4865, 0.4409]
    #   std  = [0.2673, 0.2564, 0.2762]
    print("  → 对照 src/train.py 里的 CIFAR100_MEAN / CIFAR100_STD")
    print("  → 从零训练用这组；预训练微调要用 ImageNet 的 mean/std，别搞混")
    
    print("\n完成。生成 3 张图片在 images/ 目录下：")
    print("  1. cifar100_class_balance.png     每类样本数（对比 CIFAR-10）")
    print("  2. cifar100_sample_grid.png       100 个细类各 1 张")
    print("  3. cifar100_confusing_classes.png 易混淆的 people 粗类")


if __name__ == '__main__':
    main()
