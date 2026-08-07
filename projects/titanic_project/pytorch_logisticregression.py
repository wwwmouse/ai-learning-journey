import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.preprocess import load_and_preprocess

# ===== 数据转换=====
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
x_t=torch.tensor(X_train,dtype=torch.float32)
y_t=torch.tensor(y_train,dtype=torch.float32)
x_test_t=torch.tensor(X_test,dtype=torch.float32)
y_test_t=torch.tensor(y_test,dtype=torch.float32)

model=nn.Linear(7,1) # 里面就是 7 个 w + 1 个 b，自动随机初始化
# numpy: w = np.random.randn(7)*0.01; b = 0

loss_fn=nn.BCEWithLogitsLoss() # sigmoid + 交叉熵二合一
# numpy: pred=1/(1+np.exp(-z))
# numpy: loss = -np.mean(y * np.log(pred + 1e-8) + (1-y) * np.log(1-pred + 1e-8))

optimizer=optim.SGD(model.parameters(),lr=0.01)
#  Stochastic Gradient Descent（随机梯度下降）
#  w-= lr × w的梯度  b -= lr × b的梯度

for step in range(2001):
    z=model(x_t).squeeze() 
    # model(x_t) 的输出形状是 (623, 1)
    # .squeeze() 把形状从 (623,1) 压成 (623,)
    # numpy: z = x@w + b; pred = sigmoid(z)
    
    loss=loss_fn(z,y_t)
    # 损失函数，越大说明模型越垃圾
    # 把 623 个 z 压成 623 个概率（sigmoid）
    # 用这 623 个概率和真实答案 y_t 比，算损失函数
    # numpy: loss = -np.mean(y*log(pred) + (1-y)*log(1-pred))
    
    optimizer.zero_grad() 
    # 清空上一轮的梯度
    # 因为PyTorch 把梯度存在 w 和 b 自己身上，默认往上面加，而非覆盖
    # 所以每次循环要清空上一轮的梯度（ numpy 里每次循环 dw/db 自动覆盖所以不用清）
    loss.backward()
    # 自动算 dw 和 db（取代手写的梯度公式）
    # 从 loss 出发，顺着计算过程往回走
    # 经过 loss_fn → 经过 sigmoid → 经过 x@w+b，一直回溯到w和b，算出每个参数该怎么调
    # 这行跑完，每个参数上都贴了一个标签，写着"往这个方向调，loss 会降得最快
    optimizer.step() 
    # w -= lr*dw
    # b -= lr*db
    # 真正修改w,b参数
    
    if step % 100==0:
        with torch.no_grad():  # 评估时不需要梯度，省计算
            pred=(torch.sigmoid(z)>=0.5).float() 
            # 使用sigmoid函数把z压成 0~1 的概率，概率 ≥ 50% 就判存活，否则判死亡
            acc= (pred == y_t).float().mean() 
            # 预测和真实答案比较的准确率
        print(f"第{step}步:loss={loss.item():.4f},准确率={acc:.3f}")
        # loss 本身是个 tensor，外面包着 PyTorch 的盒子。.item() 把它变成一个普通的 Python 小数
    
with torch.no_grad():
    z_test=model(x_test_t).squeeze()
    pred_test=(torch.sigmoid(z_test)>=0.5).float()
    test_acc=(pred_test==y_test_t).float().mean()
print(f"测试集准确率:{test_acc:.4f}")
# 第2000步:loss=0.4409,准确率=0.801
# 测试集准确率:0.7948