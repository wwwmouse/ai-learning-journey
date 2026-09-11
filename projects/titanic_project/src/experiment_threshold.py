"""
泰坦尼克项目：把汇报里用到的每个数字，一次全跑出来。

为什么有这个文件
----------------
"四组排查"和"四条候选路线"这些数字，一开始是分几次临时跑的，脚本没留下来，
导致同一个随机森林基线在 train.py 里是 0.791、在笔记里另一处写成 0.795 —— 差一个
测试样本，说不清是哪次配置不同。这个脚本用同一份切分、同一套预处理，把所有路线
放在一次运行里比较，消除口径不一致。

跑出来的东西对应 PPT 的两页
--------------------------
【第一部分】模型侧的四项检查 —— 对应"模型侧已饱和"那张表
    ① 练够了吗（收敛）：扫 max_iter，看 n_iter_
    ② 模型够大吗（容量）：扫决策树 max_depth，同时看训练集/测试集
    ③ 堆料有用吗（规模）：扫随机森林 n_estimators
    ④ 参数调好了吗（超参）：GridSearchCV 穷举参数网格

【第二部分】四条候选路线 —— 对应"探索候选路线"那张表
    基线 / 调阈值 / 类别加权 / 加「称谓」特征

三条口径纪律（很重要，别改）
--------------------------
1. 切分与 preprocess.py 完全一致：test_size=0.3, random_state=42, stratify=y
   → 训练集 623 / 测试集 268。不一致的话数字就和 train.py 对不上。
2. 测试集全程只用来"最后考一次"，不参与任何选择。
   阈值一律在训练集内部再切出的验证集上选；选完用全部 623 重训再考。
   在测试集上扫阈值 = 数据泄露，扫出来的"提升"是假的。
3. 每条路线只改一件事，其余全部锁死 —— 这样"没涨"才能干净地归因。

运行
----
    cd src
    python experiment_threshold.py

    如果机器跑不动网格搜索的并行，可以降到单进程：
    set TITANIC_N_JOBS=1   （CMD；PowerShell 用 $env:TITANIC_N_JOBS=1）
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_JOBS = int(os.environ.get('TITANIC_N_JOBS', '-1'))

BASE_FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


# ══════════════════════════════════════════════════════════════
#  数据准备
# ══════════════════════════════════════════════════════════════

def build_frame():
    """复刻 preprocess.py 的清洗流程，额外保留 Name（用来造「称谓」特征）。

    必须和 preprocess.py 保持同样的行顺序和同样的缺失值填充策略，
    否则切出来的 623/268 就不是同一份数据，数字自然对不上。
    """
    df = pd.read_csv(os.path.join(PROJECT_DIR, '..', 'data', 'train.csv'))
    df = df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)

    # 「称谓」：从 Name 里抽头衔。preprocess.py 把 Name 整列删了，所以这里自己留一份。
    # Mr / Miss / Mrs / Master 是主要四类，其余（Dr、Rev、Col、Countess…）样本太少，合并成 Rare。
    #
    # 为什么这个特征有用：'Master' 是未婚小男孩的称谓，而泰坦尼克是"妇女和儿童优先"。
    # 原来的 Sex 列只能分男/女，分不出"成年男性"和"小男孩"，而这两类人生还率差很多。
    # 更关键的是 Age 有 177 个缺失值、被中位数 28 填掉了 —— 小男孩的年龄信号在 Age 里
    # 基本被抹平，只能靠 Master 这个称谓找回来。所以称谓带来的是 Sex 和 Age 都没有的信息。
    title = df['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
    title = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    title = title.where(title.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Rare')
    df['Title'] = title

    # 「家庭规模」：同船的兄弟姐妹/配偶 + 父母/子女 + 自己。
    # 注意这是个"假"新特征 —— 它等于 SibSp + Parch + 1，是已有两列的线性组合，
    # 信息量为零。加它只是为了说明"加列数"和"加信息量"是两回事（见 main 里的对照）。
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df = df.drop(['Name'], axis=1)

    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)

    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    return df


def make_xy(df, extra=()):
    """返回 X, y。extra 是要额外加进来的列名。"""
    cols = BASE_FEATURES + list(extra)
    X = df[cols].copy()
    # 称谓必须 one-hot：它没有大小顺序，用 0/1/2/3 编码等于硬塞了
    # 「Mr < Miss < Mrs < Master」这种不存在的次序，线性模型会当真 —— 这个坑踩过，
    # 当时的结果是"加称谓对逻辑回归几乎无效"，改成 one-hot 后逻辑回归反而涨得最多。
    for c in extra:
        if c == 'Title':
            X = X.drop(columns=['Title']).join(pd.get_dummies(df['Title'], prefix='Title'))
    return X, df['Survived'].values


def scale(X_tr, X_te):
    """和 preprocess.py 一样：只用训练集 fit，测试集只 transform。"""
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def metrics(y_true, y_pred):
    """统一只看「存活」这一类（pos_label=1）—— 汇报里所有 P/R/F1 都对齐这一列。"""
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    )


def right(y_true, y_pred):
    """答对几题 —— 比"涨了 0.011"直观得多：268 题里多对 3 题。"""
    n = int(round(accuracy_score(y_true, y_pred) * len(y_true)))
    return f'{n}/{len(y_true)}'


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


# ══════════════════════════════════════════════════════════════
#  第一部分：模型侧的四项检查
# ══════════════════════════════════════════════════════════════

def check_convergence(X_tr, X_te, y_tr, y_te):
    """① 练够了吗 —— 扫 max_iter。

    关键不是看准确率变不变，而是看 n_iter_：sklearn 会报告"实际走了几步就停"。
    准确率不变有两种可能 —— 真收敛了，或者卡住了；n_iter_ 才能区分。
    """
    print('\n① 练够了吗（收敛）—— 只改 max_iter，其余全不动')
    for mi in [100, 1000, 10000, 100000]:
        m = LogisticRegression(max_iter=mi, random_state=SEED).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))
        print(f'   max_iter={mi:<7} n_iter_={m.n_iter_[0]:<3} 准确率={acc:.4f}  答对 {right(y_te, m.predict(X_te))}')
    print('   → n_iter_ 远远小于上限，说明早就到最优了，不是"没练够"')


def check_capacity(X_tr, X_te, y_tr, y_te):
    """② 模型够大吗 —— 扫决策树 max_depth，同时看训练集和测试集。

    这是这组排查的灵魂：如果容量不够，加深应该两边都变好；
    如果是过拟合，加深会让训练集冲上去、测试集反而掉下来。
    """
    print('\n② 模型够大吗（容量）—— 只改决策树 max_depth')
    for d in [2, 5, 10, 20, None]:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, y_tr)
        tr, te = accuracy_score(y_tr, m.predict(X_tr)), accuracy_score(y_te, m.predict(X_te))
        print(f'   max_depth={str(d):<5} 训练集={tr:.4f}  测试集={te:.4f}  差距={100 * (tr - te):.1f} 个点')
    print('   → 越深训练集越高、测试集越低，差距冲到 20 个点以上 —— 是吃撑了，不是没吃饱')


def check_scale(X_tr, X_te, y_tr, y_te):
    """③ 堆料有用吗 —— 扫随机森林 n_estimators。"""
    print('\n③ 堆料有用吗（规模）—— 只改随机森林的树数')
    for n in [10, 100, 200, 500, 1000]:
        m = RandomForestClassifier(n_estimators=n, random_state=SEED).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))
        print(f'   n_estimators={n:<5} 准确率={acc:.4f}  答对 {right(y_te, m.predict(X_te))}')
    print('   → 而且不是单调上升的：200 和 500 一样，说明是噪声级抖动')
    print('   → 这是"棵树"和"深度"的区别：容量调一棵树长多深，规模调种多少棵树')


def check_hyperparam(X_tr, X_te, y_tr, y_te):
    """④ 参数调好了吗 —— GridSearchCV 穷举参数网格（和 train.py 用同一个网格）。"""
    print('\n④ 参数调好了吗（超参）—— GridSearchCV 穷举')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=SEED), param_grid,
                        cv=5, scoring='accuracy', n_jobs=N_JOBS)
    grid.fit(X_tr, y_tr)
    base = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_tr, y_tr)
    n_fits = len(grid.cv_results_['params']) * 5
    print(f'   组合数 {len(grid.cv_results_["params"])} × 5 折 = {n_fits} 次训练')
    print(f'   最优参数 {grid.best_params_}')
    print(f'   交叉验证 {grid.best_score_:.4f} → 测试集 {grid.score(X_te, y_te):.4f}')
    print(f'   默认参数（100 棵 / 不限深度）测试集 {accuracy_score(y_te, base.predict(X_te)):.4f}')
    print('   → 180 次训练只换来 +1.1 个百分点（多答对 3 题）')
    print('   → 注意：参数是按交叉验证分数选的，测试集从头到尾没参与选择，所以这 1.1 个点是干净的')

    # ⚠️ 这里算出来的 +1.1 个百分点，和上面"规模"那组 100 → 1000 棵树的结果**恰好是同一个数**
    #    （都是 0.7910 → 0.8022 = 215/268）。这不是笔误，也不是复制粘贴错了，而是关键证据：
    #    网格搜索选出的最优组合是 max_depth=10 + 200 棵树 → 0.8022；
    #    但单看 200 棵树（不限深度）只有 0.7948，而单看 1000 棵树（不限深度）也是 0.8022。
    #    也就是说这点"提升"和多堆几棵树是同一个量级，而且两者不叠加 —— 都停在 215 题。
    #    真正落在 215 题上的只有「堆树数 / 网格搜索 / 加家庭规模」这三种；调阈值是 213、类别加权是 214。
    #    它们的共同点不是"数值相同"，而是**全都挤在 212~215 这 4 题宽的窄带里** —— 那是噪声带。
    #    别把这句说成"三条路都撞在 215"，那是不对的（调阈值不撞 215）。
    print('   ⚠️ 注意这个 +3 题和"规模"那组的 +3 题是同一个 215/268 —— 不是笔误：')
    print('      网格搜索的 0.8022 来自"200 棵树 + 深度 10"，而单看 200 棵树只有 0.7948；')
    print('      单看 1000 棵树（不限深度）也是 0.8022。两条路不叠加，说明都在噪声里。')

    # 高分点连成一片"平地"：比"只涨 1%"更能说明超参不是杠杆
    order = grid.cv_results_['rank_test_score'].argsort()[:5]
    print('   CV 前 5 名（注意 2~5 名分数完全相同，但参数南辕北辙）：')
    for i in order:
        print(f'     cv={grid.cv_results_["mean_test_score"][i]:.4f}  {grid.cv_results_["params"][i]}')


# ══════════════════════════════════════════════════════════════
#  第二部分：四条候选路线
# ══════════════════════════════════════════════════════════════

def line(tag, m, rt, note=''):
    acc, p, r, f1 = m
    print(f'  {tag:<34}{acc:>8.4f}{p:>8.4f}{r:>8.4f}{f1:>8.4f}   {rt}  {note}')


def run_routes(df, tr_idx, te_idx, tr_sub, val_sub):
    """四条路线：基线 / 调阈值 / 类别加权 / 加「称谓」。

    前三条都没换模型、没加数据 —— 只在"同一个模型、同一组预测概率"上做手脚；
    只有第四条真的改了输入。这就是"曲线内滑动"和"曲线外推"的区别。
    """
    print('\n' + '═' * 78)
    print('第二部分 · 四条候选路线（同一个模型，每次只改一件事）')
    print('═' * 78)
    print(f'  {"路线":<33}{"准确率":>9}{"存活P":>9}{"存活R":>9}{"存活F1":>9}   答对')

    for name, extra in [('7 个特征（基线）', ()),
                        ('+「称谓」(one-hot)', ('Title',)),
                        ('+「家庭规模」(假特征)', ('FamilySize',))]:
        print(f'\n【{name}】')
        X, y = make_xy(df, extra)
        # 必须标准化后再切分：逻辑回归对尺度敏感，不标准化结果会掉；
        # 随机森林对尺度不敏感，所以漏掉这一步时森林那行看不出来，专坑逻辑回归
        X_tr, X_te = scale(X.values[tr_idx], X.values[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]
        tr_mask, val_mask = np.isin(tr_idx, tr_sub), np.isin(tr_idx, val_sub)

        for model_name, make in [
            ('逻辑回归', lambda: LogisticRegression(max_iter=1000, random_state=SEED)),
            ('随机森林', lambda: RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ]:
            # ① 基线：默认阈值 0.50，用全部 623 训练
            m = make().fit(X_tr, y_tr)
            line(f'{model_name} · 基线（阈值 0.50）', metrics(y_te, m.predict(X_te)),
                 right(y_te, m.predict(X_te)))

            # ② 调阈值：只在 498 子训练集上选阈值（隔离出验证集），
            #    选完再用全部 623 重训一次，最后才碰测试集
            m_sub = make().fit(X_tr[tr_mask], y_tr[tr_mask])
            t, val_f1 = pick_threshold(m_sub, X_tr[val_mask], y_tr[val_mask])
            m_full = make().fit(X_tr, y_tr)
            pred = (m_full.predict_proba(X_te)[:, 1] >= t).astype(int)
            line(f'{model_name} · 调阈值（验证集选出 {t:.2f}）', metrics(y_te, pred),
                 right(y_te, pred), f'验证集F1={val_f1:.3f}')

            # ③ 类别加权：其他全不动，只加这一个参数
            m_bal = make()
            m_bal.set_params(class_weight='balanced')
            m_bal.fit(X_tr, y_tr)
            line(f'{model_name} · class_weight=balanced', metrics(y_te, m_bal.predict(X_te)),
                 right(y_te, m_bal.predict(X_te)))


# ══════════════════════════════════════════════════════════════

def main():
    df = build_frame()
    y_all = df['Survived'].values

    # ── 切分：与 preprocess.py 逐参数一致，保证是同一份 623/268 ──
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y_all)
    assert len(tr_idx) == 623 and len(te_idx) == 268, '切分与 preprocess.py 不一致，先查这里'

    # 训练集内部再切一份验证集，专门用来选阈值（测试集仍然一次都不碰）
    tr_sub, val_sub = train_test_split(tr_idx, test_size=0.2, random_state=SEED,
                                       stratify=y_all[tr_idx])

    print('=' * 78)
    print(f'切分: 训练 {len(tr_idx)} / 验证 {len(val_sub)} / 测试 {len(te_idx)}')
    print('称谓分布: ' + '  '.join(f'{k}={v}' for k, v in df['Title'].value_counts().items()))
    print('=' * 78)

    # 第一部分用 7 个特征的基线数据
    X, y = make_xy(df, ())
    X_tr, X_te = scale(X.values[tr_idx], X.values[te_idx])
    y_tr, y_te = y[tr_idx], y[te_idx]

    print('\n' + '═' * 78)
    print('第一部分 · 模型侧的四项检查（都在回答："是不是我没把模型调到最好？"）')
    print('═' * 78)
    check_convergence(X_tr, X_te, y_tr, y_te)
    check_capacity(X_tr, X_te, y_tr, y_te)
    check_scale(X_tr, X_te, y_tr, y_te)
    check_hyperparam(X_tr, X_te, y_tr, y_te)
    print('\n   四项都排除了 → 才轮到第五个假设：信息量（看下面第二部分）')

    run_routes(df, tr_idx, te_idx, tr_sub, val_sub)


if __name__ == '__main__':
    main()
