"""
泰坦尼克项目：两个排查脚本共用的数据与工具。

为什么单独拆出来
----------------
`check_saturation.py` 和 `explore_routes.py` 必须用**同一份切分、同一套预处理**，
否则两边的"基线"不是同一个数，数字就没法互相引用。
（这正是之前踩过的坑：同一个随机森林基线，一处 0.791、一处 0.795，
 差一个测试样本，说不清是配置不同还是笔误。）

另一个原因：`preprocess.py` 把 Name 整列删了，而「称谓」特征要从 Name 里抽
——所以路线探索没法直接复用它。这里保留 Name，两边的共同逻辑集中在这一份。

口径纪律（改这里之前先读）
--------------------------
1. 切分必须与 preprocess.py 逐参数一致：test_size=0.3, random_state=42, stratify=y
   → 训练 623 / 测试 268。改了这里，两个脚本的数字全部作废。
2. 测试集全程只用来"最后考一次"，不参与任何选择。
3. 每条路线只改一件事，其余全部锁死 —— 这样"没涨"才能干净地归因。
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

SEED = 42
TEST_SIZE = 0.3
N_JOBS = int(os.environ.get('TITANIC_N_JOBS', '-1'))

BASE_FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
N_TRAIN, N_TEST = 623, 268

W = 80  # 输出宽度


# ──────────────────────────────────────────────────────────────
#  输出排版（中文按 2 格宽算，否则表格是歪的）
# ──────────────────────────────────────────────────────────────

def rule(ch='-'):
    print('  ' + ch * (W - 2))


def head(title, sub=None):
    print('=' * W)
    print('  ' + title)
    if sub:
        print('  ' + sub)
    print('=' * W)


def vwidth(s):
    """显示宽度：中文/全角算 2 格，ASCII 算 1 格。"""
    return sum(2 if ord(c) >= 0x1100 else 1 for c in str(s))


def pad(s, n, align='<'):
    """按显示宽度补空格。align: '<' 左对齐, '>' 右对齐。"""
    s = str(s)
    fill = ' ' * max(0, n - vwidth(s))
    return s + fill if align == '<' else fill + s


# ──────────────────────────────────────────────────────────────
#  数据
# ──────────────────────────────────────────────────────────────

def find_csv():
    """定位 train.csv（脚本放 src/ 下时是 ../data/train.csv）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, '..', 'data', 'train.csv'),
                 os.path.join(here, 'data', 'train.csv')):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError('找不到 train.csv，请把脚本放在 titanic_project/src/ 下运行。')


def build_frame():
    """复刻 preprocess.py 的清洗流程，额外保留 Name 用来造「称谓」特征。

    行顺序和缺失值填充策略必须和 preprocess.py 一致，
    否则切出来的 623/268 就不是同一份数据。
    """
    df = pd.read_csv(find_csv())
    df = df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)

    # 「称谓」：Mr / Miss / Mrs / Master 是主要四类，其余（Dr、Rev、Col、Countess…）
    # 样本太少，合并成 Rare。它带来的是 Sex 和 Age 都没有的信息：
    # Sex 只分男女，分不出"成年男性"和"小男孩"；而 Age 有 177 个缺失值被中位数 28
    # 填掉了，小男孩的年龄信号在 Age 里基本被抹平。
    title = df['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
    title = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    title = title.where(title.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Rare')
    df['Title'] = title

    # 「家庭规模」：SibSp + Parch + 1。注意这是个"假"新特征 ——
    # 它是已有两列的线性组合，信息量为零。留着它专门用来对照"加列数 ≠ 加信息量"。
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df = df.drop(['Name'], axis=1)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    return df


def make_xy(df, extra=()):
    """返回 X, y。extra 是要额外加进来的列名。

    称谓必须 one-hot：它没有大小顺序，用 0/1/2/3 编码等于硬塞了
    「Mr < Miss < Mrs < Master」这种不存在的次序，线性模型会当真 ——
    这个坑踩过：当时结论是"加称谓对逻辑回归几乎无效"，改成 one-hot 后
    逻辑回归反而涨得最多（0.795 → 0.836）。
    """
    cols = BASE_FEATURES + list(extra)
    X = df[cols].copy()
    for c in extra:
        if c == 'Title':
            X = X.drop(columns=['Title']).join(pd.get_dummies(df['Title'], prefix='Title'))
    return X, df['Survived'].values


def split_indices(y_all):
    """返回 tr_idx, te_idx —— 与 preprocess.py 逐参数一致，保证 623/268。"""
    idx = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=SEED, stratify=y_all)
    assert len(tr_idx) == N_TRAIN and len(te_idx) == N_TEST, \
        '切分与 preprocess.py 不一致，先查这里'
    return tr_idx, te_idx


def split_val(tr_idx, y_all):
    """训练集内部再切一份验证集，专门用来选阈值（测试集一次都不碰）。"""
    return train_test_split(tr_idx, test_size=0.2, random_state=SEED,
                            stratify=y_all[tr_idx])


def scale(X_tr, X_te):
    """只用训练集 fit，测试集只 transform（否则是数据泄露）。"""
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


# ──────────────────────────────────────────────────────────────
#  指标
# ──────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    """统一只看「存活」这一类（pos_label=1）。"""
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def n_right(y_true, y_pred):
    """答对几题 —— 比"涨了 0.011"直观：268 题里多对 3 题。"""
    return int(round(accuracy_score(y_true, y_pred) * len(y_true)))


def right(y_true, y_pred):
    return f'{n_right(y_true, y_pred)}/{len(y_true)}'


def pick_threshold(model, X_val, y_val):
    """在验证集上扫阈值，选 F1 最高的那个。

    绝不能在测试集上扫 —— 那叫数据泄露，扫出来的"提升"是假的。
    """
    proba = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.50, -1.0
    for t in np.arange(0.20, 0.81, 0.01):
        f1 = f1_score(y_val, (proba >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), f1
    return best_t, best_f1
