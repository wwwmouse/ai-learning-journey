import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.preprocess import load_and_preprocess

# ===== 第 0 步：准备数据=====
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
x=X_train
y=y_train

# ===== 第 1 步：随机初始化参数（瞎猜一个起点）=====
w=np.random.randn(7)*0.01
b=0
learning_rate=0.01

# ===== 第 2 步：训练循环 =====
for step in range(2001):
    # --- 2a. 前向传播：用当前的 w 和 b 算预测值 ---
    z=x@w+b
    pred=1/(1+np.exp(-z)) # exp计算自然指数
    
    # --- 2b. 算 loss（看看现在猜得有多差）---
    # 交叉熵：实际=1 用 -log(p)，实际=0 用 -log(1-p)
    loss=-np.mean(y*np.log(pred+1e-8)+(1-y)*np.log(1-pred+1e-8))
    
    # --- 2c. 算梯度（每个 w 该往哪个方向调）---
    # 推导出的公式，先记住
    dz=pred-y
    dw=(1/x.shape[0])*(x.T@dz)
    db=np.mean(dz)
    
    # --- 2d. 更新参数（往梯度反方向走一小步）---
    w-=learning_rate*dw
    b-=learning_rate*db
    
    # --- 2e. 每 100 步看一眼 ---
    if step%100==0:
        acc=np.mean((pred>=0.5).astype(int)==y)
        print(f"第{step}步:loss={loss:.4f},准确率={acc:.3f}")
        # 第900步:loss=0.4543,准确率=0.793
        # 第2000步:loss=0.4413,准确率=0.799

# 训练完成后，在测试集上评估
z_test = X_test @ w + b
pred_test = 1 / (1 + np.exp(-z_test))
test_acc = np.mean((pred_test >= 0.5).astype(int) == y_test)
print(f"测试集准确率: {test_acc:.4f}")
# 测试集准确率: 0.7948
# 几乎等于train.py内调用API的准确率0.794776
# 说明效果尚可^^
