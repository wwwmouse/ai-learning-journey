import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#   之前的探索发现了这些问题，现在逐一解决：

#    Cabin缺77%                         直接删列         
#    NameTicket是文字但没规律            直接删列         
#    PassengerId是序号                  直接删列         
#    Age缺20%                           用中位数填充  
#    Embarked 缺 2 个                    用众数填充   
#    Sex是male/female文字               转成数字         
#    Embarked是S/C/Q文字                转成数字         
#    各特征数值范围差很大（Fare 0~512）   标准化到同一尺度 

# 项目根目录（脚本所在目录的上一级）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(PROJECT_DIR, '..', 'data', 'train.csv'))

# 1.删除无用列

df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#这些列对结果没意义

# 2.中位数填充年龄缺失值

age_median=df['Age'].median()
print(f'年龄中位数:{age_median}')
df['Age']=df['Age'].fillna(age_median)
print(df.head(10))

# 3.Sex,Embarked转数字

# Sex：male→1, female→0（按字母序，female在前所以是0）
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Embarked：S→2, C→0, Q→1（字母序排列）
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
# Embarked 只缺 2个值（0.22%），用众数填就行

# LabelEncoder 做的事：把所有不同的文本值按字母排序，然后从 0 开始编号。
# fit_transform 拆开理解：
#   fit:扫描这一列，记住"这个数据里有哪几种值"
#   transform:把这些值替换成数字
# Embarked 先 fillna 填了众数（mode()[0] 是出现最多的值），因为它缺了 2 个，不填会报错。

# 4.划分X,y，切训练/测试集

X = df.drop('Survived', axis=1)   # 特征：除了Survived以外的所有列
y = df['Survived']                 # 标签：我们要预测的东西

X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=0.3,      # 30% 做测试，70% 做训练
      random_state=42,    # 固定随机种子——任何人跑出来切的一样
      stratify=y          # 按 y 的比例分层抽样,保持训练和测试集里的存活率都是 38.38%
  )

# random_state=42：每次切数据是用随机数打乱的，固定种子下次运行切出来一样
# 42是社区传统数字（科幻小说《银河系漫游指南》里"生命、宇宙和万物的答案"）^^
# stratify=y：如果不加，可能训练集存活率 40%、测试集 30%，两个集合分布不一致，评估会偏。加了之后两边都是 38.38%
# 这个函数的意义是把891个人的标签和特征分成测试集和训练集，一叠用来教模型，一叠用来考模型

# 5.标准化

# 标准化就是在不改变数据相对关系的前提下，把所有列拉到同一个尺度上。
# Age 的 80 岁是老人这件事没变，Fare 的 512是富豪也没变，但数字的绝对大小不再有区别。

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 在训练集上：计算均值标准差,将其存入scaler + 转换
X_test = scaler.transform(X_test)        # 在测试集上：无法计算均值标准差,只用训练集的参数转换！

# 1. fit — scaler 扫描 X_train 每一列，记住 Age 均值是 29、Fare 均值是 32、Age 标准差是 14……
# 2. transform — 用刚才记住的数据，把 X_train 里每个数替换成标准化后的值,新值 = (原始值 - 该列均值) / 该列标准差

#   StandardScaler 把每列变成"均值为 0，标准差为 1"：
#   原始 Age: [2, 22, 28, 35, 80]
#   标准化后: [-1.2, -0.3, 0.0, 0.4, 2.7]   ← 负数表示低于平均，正数表示高于平均。

print(f'预处理完成。训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}')
