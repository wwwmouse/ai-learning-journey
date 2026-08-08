# PyTorch 训练通用工具函数

# 六个脚本(mnist/cifar10 下 mlp/cnn/cnn_optimized)的公共逻辑抽到这里
# 改训练循环只改这一个文件。


import os
import random
import torch
import matplotlib.pyplot as plt


# ===== 1. 固定随机种子 =====

def set_seed(seed=42):
    #固定 Python / PyTorch / GPU 所有随机源，保证每次训练结果可复现

    # 对学习阶段至关重要——让每次改代码后能确认
    # 分数变化来自改动，而非随机初始化的运气
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   # 强制确定性卷积算法
        torch.backends.cudnn.benchmark = False       # 关闭自动算法搜索


# ===== 2. GPU 探测 =====

def get_device():
    # 自动检测 GPU，兼容性检查，失败自动回退 CPU。

    # Returns:
    #     torch.device: 'cuda' 或 'cpu'
    device = torch.device('cpu')
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()               # 试探性 GPU 操作
            device = torch.device('cuda')
            print(f"使用设备: cuda")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"  GPU 不兼容，回退到 CPU。原因: {e}")
            print(f"  修复: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
    else:
        print(f"使用设备: cpu (未检测到 CUDA)")
        print("  如需 GPU 加速请安装 CUDA 版 PyTorch")
    return device


# ===== 3. 训练一个 epoch =====

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    # 跑一个 epoch 的训练：五步循环，返回平均 loss

    # 五步顺序（写死在参数里的知识）：
    #    zero_grad  →  forward  →  loss  →  backward  →  step

    # Args:
    #     model:      nn.Module
    #     loader:     DataLoader（训练集）
    #     loss_fn:    损失函数
    #     optimizer:  优化器
    #     device:     torch.device

    # Returns:
    #     float: 这轮 epoch 的平均 loss
    
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # ① 清空上一轮梯度
        outputs = model(images)          # ② 前向传播
        loss = loss_fn(outputs, labels)  # ③ 算 loss
        loss.backward()                  # ④ 反向传播——自动求梯度
        optimizer.step()                 # ⑤ 真的去调参数

        total_loss += loss.item()

    return total_loss / len(loader)


# ===== 4. 评估 =====

def evaluate(model, loader, device):
    # 在测试集上评估准确率

    # 不做的事（非常重要）：
    #   - 绝不调用 optimizer.step()
    #   - 用 torch.no_grad() 关掉计算图
    #   - model.eval() 让 Dropout/BatchNorm 切换到测试模式

    # Args:
    #     model:  nn.Module
    #     loader: DataLoader（测试集）
    #     device: torch.device

    # Returns:
    #     float: 准确率（0~100）
    
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)     # 多分类：取最大值的位置
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ===== 5. 画训练曲线 =====

def plot_curves(train_losses, test_accs, save_path, title="Training Curves"):
    # 画 loss（左）和 accuracy（右）双图并保存。

    # Args:
    #     train_losses: list[float]  每个 epoch 的平均 loss
    #     test_accs:    list[float]  每个 epoch 的测试准确率 (%)
    #     save_path:    str          保存路径
    #     title:        str          图表总标题
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = len(train_losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：loss 曲线
    ax1.plot(range(1, epochs + 1), train_losses, marker='o', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Training Loss')

    # 右图：准确率曲线
    ax2.plot(range(1, epochs + 1), test_accs, marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} — Test Accuracy')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"训练曲线已保存到 {save_path}")


# ===== 6. 保存模型 =====

def save_model(model, save_path):
    # 保存模型参数（state_dict）到磁盘。

    # 只保存 w 和 b 的值——加载前需要先搭一个结构相同的空壳。

    # Args:
    #     model:     nn.Module
    #     save_path: str 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")
