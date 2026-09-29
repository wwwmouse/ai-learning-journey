# CIFAR-100 · ResNet-18 从零训练（from scratch）
#
# 一句话：不使用任何预训练权重，从随机初始化开始训练 ResNet-18。
#
# 和"预训练微调"唯一的本质区别是【起点】：
#   本项目：  weights=None            → 随机初始化，200 epoch，lr=0.1
#   微调版：  weights=IMAGENET1K_V1   → 好起点，    20  epoch，lr=0.005
# 但由这一个区别引出的连锁改动有 7 处，见 README 的对照表。
#
# 本脚本【自包含】：数据、模型、训练、评估、保存全在一个文件里。
# 理由是这里只有一个实验脚本，拆出 utils.py 只会让读代码时来回跳文件。
# 等出现第 2 个需要共享这些函数的脚本时再抽。
#
# 运行：
#   python src/train.py               # 完整 200 epoch（约 1~1.5 小时）
#   python src/train.py --epochs 3    # 冒烟测试：全量数据只跑 3 轮，约 1 分钟
#   python src/train.py --resume      # 从 checkpoint 断点续训

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'        # 中文字体
rcParams['axes.unicode_minus'] = False

# ===== 路径 =====
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

# ===== 常量 =====
NUM_CLASSES = 100

# CIFAR-100 数据集自身的归一化统计量。
# ★ 从零训练用这一组；预训练微调要用 ImageNet 的 mean/std，别搞混。
# 想自己核算，跑 src/explore_data.py 最后一段会打印实测值。
CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]

# 改完 stem 后的 CIFAR 版 ResNet-18 参数量，用来验证结构没改错
EXPECTED_PARAMS = 11_220_132


# ===== 1. 随机种子 =====

def set_seed(seed):
    """
    固定所有随机源，并强制 cuDNN 只使用确定性算法。

    关于 cudnn.benchmark：
      设成 True 时，cuDNN 会在第一次遇到某个输入尺寸时试跑几种卷积算法、
      挑最快的缓存下来。对固定输入尺寸的训练循环确实能提速 10~20%。
      但"挑中哪个算法"依赖硬件计时，两次运行的算法选择可能不同，
      于是即使把种子固定，结果也无法逐位复现（分数可能差在小数点后两位）。

      本项目要和后续"预训练微调"的实验比分数，宁可慢一点，
      也要让"我只改了一处，分数因此动了几点"这件事是可归因的。
      这和 pytorch-cnn-project/src/utils.py 的取法保持一致。

    补充：还有一个更狠的 torch.use_deterministic_algorithms(True)，
      它会直接禁止任何非确定性算子（缺确定性实现的算子会直接报错）。
      ResNet-18 用不到这么严格，这里不开。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True    # 只选确定性卷积算法
    torch.backends.cudnn.benchmark = False       # 关闭算法自动搜索


# ===== 2. GPU 探测 =====

def get_device():
    """自动检测 GPU，不可用则回退 CPU。"""
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()               # 试探性操作，确认 GPU 真的能用
            print(f"使用设备: cuda  ({torch.cuda.get_device_name(0)})")
            return torch.device('cuda')
        except Exception as e:
            print(f"GPU 不兼容，回退 CPU。原因: {e}")
    print("使用设备: cpu（未检测到可用 CUDA）")
    return torch.device('cpu')


# ===== 3. 数据 =====

def build_loaders(batch_size, num_workers):
    """搭数据管道，返回 (类别名, 训练集, 训练 loader, 测试 loader)。"""

    # 训练集变换：增强只加在这里
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # 随机平移，位置不变性
        transforms.RandomHorizontalFlip(),                             # 左右镜像，翻转不变性
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(p=0.25),   # 随机遮住一小块，逼模型别死盯局部像素
    ])

    # 测试集变换：只转 Tensor + 归一化，【一条增强都不许加】
    # 加了增强测出来的分数会失真，且和其它实验不可比
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    base_train = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform)
    base_test = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform)
    class_names = base_train.classes          # 100 个细类名

    train_loader = DataLoader(
        base_train, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        base_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return class_names, base_train, train_loader, test_loader


# ===== 4. 模型 =====

def build_model():
    """
    搭建"CIFAR 版 ResNet-18"。

    注意：改完 stem 之后，严格说它已经不是原始 ResNet-18 了，
    而是社区通用的 32×32 适配版——README 里要写清楚，别让人以为跑的是原版。
    """
    # ★ 从零训练的关键：weights=None
    # 这一行是整个项目与"预训练微调"唯一的本质区别
    model = models.resnet18(weights=None)

    # --- 改造 1：stem ---
    # 原版是 7×7 stride2 + 3×3 maxpool，连续两次降采样，进来就砍到 1/4。
    # 32×32 走完会变成：32 → 8 → 8 → 4 → 2 → 1，layer4 只剩 1×1，
    # 空间信息全丢；而 layer4 恰恰占了骨干约 75% 的参数。
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()        # 什么都不做的占位层

    # --- 改造 2：分类头 1000 类 → 100 类 ---
    model.fc = nn.Linear(512, NUM_CLASSES)

    # --- 改造 3：残差块末尾 BN 零初始化（只有从零训练才加）---
    # 残差块输出 = F(x) + x。把 bn2.weight 置零后 F(x) 初始恰好为 0，
    # 整个块初始等价于恒等映射——网络一开始"什么都没做"，梯度通路干净，长训练更稳。
    # ★ 微调版绝对不能加这个，会把预训练学到的权重破坏掉。
    for m in model.modules():
        if isinstance(m, models.resnet.BasicBlock):
            nn.init.zeros_(m.bn2.weight)

    return model


# ===== 5. 训练 / 评估 =====

def train_one_epoch(model, loader, loss_fn, optimizer, device, use_amp):
    """跑一个 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()                 # 1. 清空上一轮的梯度
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(images)           # 2. 前向传播
            loss = loss_fn(outputs, labels)   # 3. 算 loss
        loss.backward()                       # 4. 反向传播
        optimizer.step()                      # 5. 更新参数

        total_loss += loss.item()             # .item() 取数字，避免计算图越攒越大

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """在测试集上算准确率（%）。"""
    model.eval()                              # 切换 BN / Dropout 到测试模式
    correct, total = 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(images)
        predicted = outputs.argmax(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


@torch.no_grad()
def per_class_accuracy(model, loader, device, use_amp, num_classes):
    """
    每个类别各自的准确率。

    比总体准确率信息量大得多——CIFAR-100 的总体数字会掩盖
    "有些细类几乎学不会"这个事实（比如 leopard / tiger / lion）。
    """
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(images)
        predicted = outputs.argmax(1).cpu()
        hit = predicted == labels
        correct += torch.bincount(labels[hit], minlength=num_classes)
        total += torch.bincount(labels, minlength=num_classes)

    accs = (100.0 * correct.float() / total.clamp(min=1)).tolist()
    return accs, total.tolist()


# ===== 6. 画图 =====

def plot_curves(train_losses, test_accs, save_path, title):
    """loss + 准确率双图。两条曲线的开口就是过拟合的可视化证据。"""
    epochs = len(train_losses)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, epochs + 1), train_losses, color='blue', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — 训练 loss')
    ax1.grid(alpha=0.3)

    ax2.plot(range(1, epochs + 1), test_accs, color='green', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} — 测试准确率')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线已保存到 {save_path}")


def plot_per_class(accs, class_names, save_path, title):
    """100 个类各自的准确率柱状图，按准确率排序。"""
    order = np.argsort(accs)[::-1]
    sorted_accs = [accs[i] for i in order]
    sorted_names = [class_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(sorted_accs)), sorted_accs, color='steelblue', width=1.0)
    ax.axhline(np.mean(accs), color='red', linestyle='--', linewidth=1,
               label=f'平均 {np.mean(accs):.1f}%')
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=5)
    ax.set_ylabel('准确率 (%)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"每类准确率图已保存到 {save_path}")


# ===== 7. 主流程 =====

def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 ResNet-18 从零训练')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=5, help='线性 warmup 的 epoch 数')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='从 checkpoint.pth 续训')
    parser.add_argument('--show', action='store_true',
                        help='训练结束后弹出图表窗口（默认不弹，图已存盘）')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    use_amp = (device.type == 'cuda')       # bf16 混合精度：省显存、提速

    # ===== 数据 =====
    class_names, train_set, train_loader, test_loader = build_loaders(
        args.batch_size, args.num_workers
    )
    print(f"训练集 {len(train_set)} 张 / {len(train_loader)} 个 batch，"
          f"测试集 {len(test_loader.dataset)} 张")

    # ===== 模型 =====
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    if n_params != EXPECTED_PARAMS:
        print(f"  ⚠ 预期 {EXPECTED_PARAMS:,}，对不上说明结构改错了（先查 stem 和 fc）")
    else:
        print(f"  ✓ 与预期一致，结构改造正确")

    # ===== 损失 / 优化器 / 调度器 =====
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)

    # 5 epoch 线性 warmup：lr 从 0.1×lr 线性升到 lr。
    # 从零训练 + BatchNorm 在最初几个 epoch 容易震荡，warmup 能稳住开局。
    warmup_epochs = min(args.warmup, max(0, args.epochs - 1))
    if warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs))
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ===== 续训 =====
    ckpt_path = os.path.join(MODELS_DIR, 'checkpoint.pth')
    best_path = os.path.join(MODELS_DIR, 'resnet18_cifar100_best.pth')
    start_epoch = 0
    best_acc = 0.0
    train_losses, test_accs = [], []

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        train_losses = ckpt['train_losses']
        test_accs = ckpt['test_accs']
        print(f"已从 {ckpt_path} 续训：从 epoch {start_epoch + 1} 开始，当前最佳 {best_acc:.2f}%")

    # ===== 日志 =====
    log_path = os.path.join(LOGS_DIR, 'train_log.csv')
    write_header = not (args.resume and os.path.exists(log_path))
    log_file = open(log_path, 'a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(['epoch', 'lr', 'train_loss', 'test_acc'])

    # ===== 训练 =====
    print(f"\n开始训练：{args.epochs} epoch，lr={args.lr}，"
          f"weight_decay={args.weight_decay}，label_smoothing={args.label_smoothing}")
    print("随时可以 Ctrl-C 中断：每个 epoch 都存了 checkpoint，之后用 --resume 接着跑。\n")

    interrupted = False
    try:
        for epoch in range(start_epoch, args.epochs):
            current_lr = optimizer.param_groups[0]['lr']

            avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, use_amp)
            acc = evaluate(model, test_loader, device, use_amp)
            scheduler.step()

            train_losses.append(avg_loss)
            test_accs.append(acc)
            log_writer.writerow([epoch + 1, f'{current_lr:.6f}', f'{avg_loss:.4f}', f'{acc:.2f}'])
            log_file.flush()

            flag = ''
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_path)
                flag = '  ← 最佳，已保存'

            print(f"epoch {epoch + 1:3d}/{args.epochs}: "
                  f"lr={current_lr:.4f}  train_loss={avg_loss:.4f}  test_acc={acc:.2f}%{flag}")

            # 每个 epoch 存一次断点：笔记本上跑 1 个多小时，中途休眠/断电是常态
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'test_accs': test_accs,
            }, ckpt_path)
    except KeyboardInterrupt:
        # 中断不属于正常结束，但已经跑完的 epoch 是有价值的——
        # 存好日志、照常画曲线，别让 1 个多小时白费。
        interrupted = True
        print(f"\n\n⚠ 收到 Ctrl-C 中断（已完成 {len(train_losses)} 个 epoch）")
        print(f"  续训命令： python src/train.py --resume")
    finally:
        log_file.close()

    # ===== 收尾 =====
    if interrupted:
        print(f"\n中断退出。当前最佳测试准确率: {best_acc:.2f}%")
    else:
        print(f"\n训练结束。最佳测试准确率: {best_acc:.2f}%")
    print(f"最佳权重: {best_path}")
    print(f"逐 epoch 日志: {log_path}")

    if not train_losses:
        print("一个 epoch 都没跑完，没有曲线可画。")
        return

    plot_curves(train_losses, test_accs,
                os.path.join(IMAGES_DIR, 'resnet18_cifar100_curves.png'),
                'CIFAR-100 ResNet-18 (from scratch)')

    # 用【最佳】权重做最后的详细分析，而不是最后一轮的权重
    # （若在第一个 epoch 中途就被中断，best 权重还没存过，跳过分析）
    if not os.path.exists(best_path):
        print("还没产生 best 权重（第一个 epoch 未跑完），跳过最终分析。")
        return

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_acc = evaluate(model, test_loader, device, use_amp)
    print(f"\n用最佳权重复评: {final_acc:.2f}%")
    print(f"（随机猜的基线是 {100 / NUM_CLASSES:.2f}%，"
          f"社区常见区间约 75~78%，未逐篇核实）")

    accs, counts = per_class_accuracy(model, test_loader, device, use_amp, NUM_CLASSES)
    plot_per_class(accs, class_names,
                   os.path.join(IMAGES_DIR, 'resnet18_cifar100_per_class.png'),
                   'CIFAR-100 每类准确率（按准确率降序）')

    order = np.argsort(accs)
    print("\n最容易混淆的 10 个类（准确率最低）:")
    for i in order[:10]:
        print(f"  {class_names[i]:<16s} {accs[i]:5.1f}%   ({counts[i]} 张测试图)")
    print("\n学得最好的 5 个类:")
    for i in order[::-1][:5]:
        print(f"  {class_names[i]:<16s} {accs[i]:5.1f}%")

    # 图已经存盘，默认不弹窗口——plt.show() 会阻塞进程直到窗口被关掉，
    # 长训练跑完卡在这里等一个人来点关闭是很糟糕的体验。
    if args.show:
        plt.show()


if __name__ == '__main__':
    # ★ Windows 上 num_workers>0 时，训练入口必须在这个 guard 里面，
    #   否则子进程会重复导入主模块，导致报错或死循环。
    main()
