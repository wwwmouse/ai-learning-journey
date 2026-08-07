import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# 项目根目录（脚本所在目录的上一级）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_preprocess():
    """读数据、清洗、编码、切分、标准化，返回 X_train, X_test, y_train, y_test, feature_names"""

    df = pd.read_csv(os.path.join(PROJECT_DIR, '..', 'data', 'train.csv'))

    # 1. 删除无用列
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # 2. 中位数填充年龄缺失值
    age_median = df['Age'].median()
    print(f'年龄中位数: {age_median}')
    df['Age'] = df['Age'].fillna(age_median)

    # 3. Sex, Embarked 转数字
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
#   fit — 扫描一遍数据，记住"有哪些不同的值"（["female", "male"]）
#   transform — 按字母顺序给每个值分配编号（female → 0，male → 1）
#   等价于： le = LabelEncoder()       创建一个编码器实例
#           le.fit(df['Sex'])         扫描：发现 female 和 male
#           df['Sex'] = le.transform(df['Sex'])  替换：female→0, male→1
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) #对于极少的缺失值，使用众数填充最保险
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])   #同上
    
    # 4. 划分 X, y，切训练/测试集
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, # 特征矩阵，标签列
        test_size=0.3, # 训练集规模
        random_state=42, # 固定随机种子
        stratify=y # 分层依据，通常为标签列，作用为把数据按照这个数组的不同类别分组，保证按照相同比例拆分训练集和测试集
    ) 

    # 5. 标准化:把所有特征拉到同一个尺度上
    # 标准化后X_train会从 DataFrame 会变成 numpy 数组，列名就丢了
    # 因此需要提前记录列名
    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # 从训练集计算均值和标准差，然后应用到训练集
    # 等价于先 fit 再 transform：
    # fit(X_train)：计算训练集的均值和标准差，把这些数值存到 scaler 实例属性中（scaler.mean_ 和 scaler.scale_ ）
    # transform(X_train)：用刚才存下来的均值和标准差，对训练集进行转换
    
    X_test = scaler.transform(X_test) # 此时scaler内已经记录了刚才fit训练集的均值和标准差，直接用其来转换测试集
    # 为什么不能对测试集也 fit_transform：
    # 因为"考试"时不知道考试数据的真实分布，用测试集自己的均值和标准差去标准化，等于偷看了考试数据
    # 造成数据泄露（data leakage），导致模型评估结果虚高，是初学者最容易犯的错误之一

    # 统一转为 numpy 数组，避免下游混淆（X 是 ndarray 但 y 是 Series）
    y_train = y_train.values
    y_test = y_test.values

    print(f'预处理完成。训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}')
    return X_train, X_test, y_train, y_test, feature_names


# 如果直接运行这个文件（而不是被 import），就执行预处理并打印摘要
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
    print(f'\n训练集标签分布:\n{pd.Series(y_train).value_counts()}')
    print(f'测试集标签分布:\n{pd.Series(y_test).value_counts()}')
