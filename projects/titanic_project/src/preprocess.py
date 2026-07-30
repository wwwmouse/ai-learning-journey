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
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    # 4. 划分 X, y，切训练/测试集
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # 5. 标准化
    # 标准化之前记下列名，之后 DataFrame 会变成 numpy 数组，列名就丢了
    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f'预处理完成。训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}')
    return X_train, X_test, y_train, y_test, feature_names


# 如果直接运行这个文件（而不是被 import），就执行预处理并打印摘要
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
    print(f'\n训练集标签分布:\n{pd.Series(y_train).value_counts()}')
    print(f'测试集标签分布:\n{pd.Series(y_test).value_counts()}')
