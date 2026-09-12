# CIFAR-10 · 迁移学习（一）：冻结预训练 ResNet-18，只训分类头
#
# 一句话：预训练模型在 ImageNet（1400 万张图、1000 类）上学到的"视觉常识"，
#         能不能直接搬到 CIFAR-10 上用？——冻结 backbone 就是"把特征提取器当现成零件买"。
#
# 对照对象：src/cifar10/cnn.py（从零训练 5 层 CNN，80.03%）
# 与 cnn_finetune.py 的唯一区别：本文件冻结 backbone，且只训 3 个 epoch。
#
# 为什么冻结能work：CNN 学到的特征是分层且通用的——
#   第 1~2 层：边缘、色块、简单纹理      ← 任何图像任务都需要，与"分什么类"无关
#   第 3~4 层：纹理组合、局部部件（眼睛、轮子、尖耳朵）
#   第 5 层+ ：任务相关的语义组合        ← 只有这里是任务特定的
# 所以前面整块可以搬，只有最后那层"决策"要换成自己的 10 类。
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils import set_seed, get_device, train_one_epoch, evaluate, plot_curves, save_model

# ImageNet 的归一化统计量。预训练权重就是在这个输入分布上学出来的，
# 所以喂给它的图必须用同一套均值/标准差，否则前几层的"常识"会失配。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model():
    # 预训练 ResNet-18 + 32×32 适配 + 换分类头
    # 这三步才是本脚本真正"自己的工作"—— models.resnet18() 那一行只是起点
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 改 stem 之前，先把 7×7 的预训练卷积核留一份。
    # ★ 下面那行 nn.Conv2d(...) 会新建一个【随机初始化】的层，把预训练权重整个丢掉 ——
    #   这件事在"微调版"里能靠训练自己补回来，但在"冻结版"里是致命的：
    #   冻结 = 它永远不会被训练，等于让一个随机层守在特征提取器的入口。
    #   【第一版就是这个 bug：冻结版只跑到 46.99%；
    #     实测 conv1 与初始值的平均绝对差 = 0.00000000（精确为零，证明它一动没动）】
    pre_conv1 = model.conv1.weight.data.clone()

    # ResNet-18 原本第一层是 7×7 卷积 stride 2，后面还跟一个 3×3 最大池化，
    # 一上来就把图缩到 1/4。224×224 的 ImageNet 受得了，32×32 的 CIFAR 经不起 ——
    # 换成 3×3 stride 1（尺寸保持不变），并把池化换成恒等映射。
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 把留好的 7×7 卷积核插值成 3×3 装回去 —— "保住预训练知识"的关键一步。
    # （微调版不做也行，conv1 会自己学出来；冻结版必须做。）
    model.conv1.weight.data = F.interpolate(pre_conv1, size=(3, 3), mode='bilinear', align_corners=False)

    model.maxpool = nn.Identity()

    # 换分类头：ImageNet 是 1000 类，CIFAR-10 是 10 类
    model.fc = nn.Linear(512, 10)
    return model


def main():
    set_seed(42)
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')

    device = get_device()

    # ===== 数据 =====
    # 比从零训练那版多了 Normalize(ImageNet)；增强用 CIFAR 上最常见的 RandomCrop + 水平翻转
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_data = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_data  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    # batch_size=128（从零训练那版用的是 64）：ResNet 比 5 层 CNN 深得多，
    # 而 BatchNorm 的统计量需要足够大的 batch 才稳。这一项和"从零 vs 预训练"无关，不算对照变量。
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)

    # ===== 模型 =====
    model = build_model().to(device)

    # ★ 本脚本与 cnn_finetune.py 的唯一区别在这里：
    #   先把全部参数冻住（requires_grad=False），再只解开最后的分类头。
    #   冻结的参数不参与反向传播，等于把 ResNet 当一个固定的特征提取器用。
    #
    # 容易忽略的一点：requires_grad=False 只停"梯度"，不停 BatchNorm。
    # utils.train_one_epoch() 开头会调 model.train()，此时冻结层里的 BN 仍然会用新数据
    # 更新 running_mean / running_var —— 相当于让统计量适配 CIFAR 的分布，通常是好事。
    # 若想让 BN 也完全冻结（保持 ImageNet 的统计量），得给冻结层单独调 .eval()，本项目不做。
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,}  ({trainable / total * 100:.2f}%)")

    loss_fn = nn.CrossEntropyLoss()
    # 只把可训练参数交给优化器（把冻结的也传进来 PyTorch 会直接报错）
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    # ===== 训练 =====
    # 只训 3 个 epoch：特征已经现成，要学的只有"512 维特征 → 10 类"这一层映射
    epochs = 3
    train_losses, test_accs = [], []

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        acc = evaluate(model, test_loader, device)
        train_losses.append(avg_loss)
        test_accs.append(acc)
        print(f"epoch {epoch+1:2d}: train_loss={avg_loss:.4f}, test_acc={acc:.2f}%")

    # ===== 保存 =====
    IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
    plot_curves(train_losses, test_accs,
                os.path.join(IMAGES_DIR, 'cifar10_pretrained.png'),
                'CIFAR-10 ResNet-18 (frozen backbone)')
    save_model(model, os.path.join(MODELS_DIR, 'cifar10_pretrained.pth'))

    print(f"\n最终测试准确率: {test_accs[-1]:.2f}%   （对照：从零训练 5 层 CNN = 80.03%）")
    # 最终测试准确率: 63.63%   （对照：从零训练 5 层 CNN = 80.03%）

if __name__ == '__main__':
    main()
