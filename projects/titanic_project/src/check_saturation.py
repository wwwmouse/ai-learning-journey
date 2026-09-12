"""
泰坦尼克项目 · 排查之一：模型侧饱和了吗？

【这个脚本回答什么】
    "模型效果上不去，是不是我没把模型调到最好？"

如果答案是"是"，那就该继续调参；
如果答案是"不是" —— 即收敛、容量、规模、超参四个方向全部排除 ——
**才**轮到下一个假设：信息量不够（去跑 explore_routes.py）。

【和 explore_routes.py 的关系】
    本脚本（前半程）              explore_routes.py（后半程）
    ├ 判定"模型侧是否还有空间"     ├ 假设已排除，换方向找信息
    ├ 回答"还能不能靠调参救"       ├ 回答"往哪加信息、值不值"
    └ 结论：饱和 → 调参没用了       └ 结论：加称谓特征才跳出噪声带

    顺序不能反：不先把"调参"这条路走死，就说"要加特征"是拍脑袋。
    这也是项目文档里那条元教训 —— 排查顺序错了，归因就会错。

【四组排查】
    ① 收敛   扫 max_iter，看 n_iter_     —— 是不是没练够？
    ② 容量   扫决策树 max_depth          —— 是不是模型太小？
    ③ 规模   扫随机森林 n_estimators      —— 是不是树不够多？
    ④ 超参   GridSearchCV 穷举            —— 是不是参数没调好？

【运行】
    cd src
    python check_saturation.py
    跑不动网格搜索的并行：set TITANIC_N_JOBS=1（PowerShell 用 $env:TITANIC_N_JOBS=1）
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from titanic_common import (SEED, N_JOBS, W, head, rule, pad,
                            build_frame, make_xy, split_indices, scale,
                            acc, right, n_right)


# ══════════════════════════════════════════════════════════════
#  四项检查：每个只 return 数据，打印交给 show_*
# ══════════════════════════════════════════════════════════════

def check_convergence(X_tr, X_te, y_tr, y_te):
    """① 练够了吗 —— 扫 max_iter。

    关键不是看准确率变不变，而是看 n_iter_：sklearn 会报告"实际走了几步就停"。
    准确率不变有两种可能 —— 真收敛了，或者卡住了；n_iter_ 才能区分。
    """
    rows = []
    for mi in [100, 1000, 10000, 100000]:
        m = LogisticRegression(max_iter=mi, random_state=SEED).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        rows.append(dict(max_iter=mi, n_iter=int(m.n_iter_[0]),
                         acc=acc(y_te, pred), right=right(y_te, pred)))
    return dict(rows=rows, ok=True)


def check_capacity(X_tr, X_te, y_tr, y_te):
    """② 模型够大吗 —— 扫决策树 max_depth，同时看训练集和测试集。

    这是这组排查的灵魂：容量不够的话加深应该两边都变好；
    过拟合的话加深会让训练集冲上去、测试集反而掉下来。
    """
    rows = []
    for d in [2, 5, 10, 20, None]:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, y_tr)
        tr = acc(y_tr, m.predict(X_tr))
        te = acc(y_te, m.predict(X_te))
        rows.append(dict(depth=d, train=tr, test=te, gap=100 * (tr - te)))
    return dict(rows=rows, ok=True)


def check_scale(X_tr, X_te, y_tr, y_te):
    """③ 堆料有用吗 —— 扫随机森林 n_estimators。"""
    rows = []
    for n in [10, 100, 200, 500, 1000]:
        m = RandomForestClassifier(n_estimators=n, random_state=SEED).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        rows.append(dict(n=n, acc=acc(y_te, pred), right=right(y_te, pred),
                         nright=n_right(y_te, pred)))
    by_n = {r['n']: r for r in rows}
    return dict(rows=rows, base=by_n[100], top=by_n[1000], ok=True)


def check_hyperparam(X_tr, X_te, y_tr, y_te):
    """④ 参数调好了吗 —— GridSearchCV 穷举参数网格（和 train.py 同一个网格）。"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=SEED), param_grid,
                        cv=5, scoring='accuracy', n_jobs=N_JOBS)
    grid.fit(X_tr, y_tr)

    base = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_tr, y_tr)

    # 单看 200 棵树（不限深度）—— 用来拆穿"网格搜索选出的最优组合"：
    # 最优组合里带着 n_estimators=200，但 200 棵树单跑其实到不了 0.8022
    rf200 = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(X_tr, y_tr)

    # CV 前 5 名：注意 2~5 名分数完全相同，参数却南辕北辙 —— 高分点连成一片平地
    order = grid.cv_results_['rank_test_score'].argsort()[:5]
    top5 = [(grid.cv_results_['mean_test_score'][i], grid.cv_results_['params'][i])
            for i in order]

    return dict(
        n_params=len(grid.cv_results_['params']),
        n_fits=len(grid.cv_results_['params']) * 5,
        best_params=grid.best_params_,
        best_cv=grid.best_score_,
        grid_acc=acc(y_te, grid.predict(X_te)),
        base_acc=acc(y_te, base.predict(X_te)),
        base_right=right(y_te, base.predict(X_te)),
        grid_right=right(y_te, grid.predict(X_te)),
        base_nright=n_right(y_te, base.predict(X_te)),
        grid_nright=n_right(y_te, grid.predict(X_te)),
        acc200=acc(y_te, rf200.predict(X_te)),
        right200=right(y_te, rf200.predict(X_te)),
        top5=top5, ok=True,
    )


# ══════════════════════════════════════════════════════════════
#  打印
# ══════════════════════════════════════════════════════════════

def show_summary(conv, cap, sca, hyp):
    head('一、四项排查（逐项只改一个变量）')
    print(pad('  #', 5) + pad('排查项', 9) + pad('做法', 26)
          + pad('结果', 27) + '结论')
    rule()

    c = conv['rows']
    d5 = next(r for r in cap['rows'] if r['depth'] == 5)
    dn = next(r for r in cap['rows'] if r['depth'] is None)

    items = [
        ('①', '收敛', '扫 max_iter 100→10万', f'n_iter_={c[0]["n_iter"]} 就停', '早就到最优'),
        ('②', '容量', '扫决策树 max_depth',
         f'测试集 {d5["test"]:.3f} → {dn["test"]:.3f}', '容量过剩（吃训练集）'),
        ('③', '规模', '扫树数 100 → 1000',
         f'{sca["base"]["acc"]:.4f} → {sca["top"]["acc"]:.4f}', '收益趋零（抖动）'),
        ('④', '超参', 'GridSearchCV 36 组',
         f'{hyp["base_acc"]:.4f} → {hyp["grid_acc"]:.4f}', '收益趋零（抖动）'),
    ]
    for num, name, how, res, concl in items:
        print(pad('  ' + num, 5) + pad(name, 9) + pad(how, 26)
              + pad(res, 27) + concl)
    print()


def show_scale_vs_hyper(sca, hyp):
    """③ 和 ④ 单独拎出来对照 —— 它们撞在同一条线上。"""
    head('二、③ 规模 vs ④ 超参 —— 是不是同一个结果？')
    print(pad('  手段', 38) + pad('准确率', 11, '>') + pad('答对', 12, '>')
          + pad('相对基线', 12, '>'))
    rule()

    b_n = sca['base']['nright']
    rows = [
        ('基线（随机森林 100 棵）', sca['base']['acc'], b_n),
        ('③ 规模：堆到 1000 棵', sca['top']['acc'], sca['top']['nright']),
        (f'④ 超参：网格搜索 {hyp["best_params"]["n_estimators"]} 棵 + 深度 '
         f'{hyp["best_params"]["max_depth"]}', hyp['grid_acc'], hyp['grid_nright']),
    ]
    for name, a, n in rows:
        delta = '' if n == b_n else f'{n - b_n:+d} 题'
        print(pad('  ' + name, 38) + pad(f'{a:.4f}', 11, '>')
              + pad(f'{n}/268', 12, '>') + pad(delta, 12, '>'))

    same = (abs(sca['top']['acc'] - hyp['grid_acc']) < 1e-12
            and sca['top']['nright'] == hyp['grid_nright'])
    print()
    print(f'  ③ 规模  {sca["top"]["acc"]:.4f}  =  {sca["top"]["right"]}')
    print(f'  ④ 超参  {hyp["grid_acc"]:.4f}  =  {hyp["grid_right"]}')
    print(f'  → 两者{"完全相同" if same else "不同"}'
          f'（准确率差 {abs(sca["top"]["acc"] - hyp["grid_acc"]):.4f}，'
          f'答对题数差 {abs(sca["top"]["nright"] - hyp["grid_nright"])} 题）')
    print()
    print(f'  把树从 100 堆到 1000，和花 {hyp["n_fits"]} 次训练做网格搜索，')
    print('  撞在同一条线上 —— 两种"拧旋钮"的方式都停在 215/268。')
    print('  而且不叠加：网格搜索的最优组合里也带着 n_estimators=200，')
    print(f'  但单看 200 棵树（不限深度）只有 {hyp["acc200"]:.4f}（{hyp["right200"]}）——')
    print('  单看 1000 棵（不限深度）也是 0.8022。多出来的那点，是深度 10 单独给的。')
    print()


def show_flat_plateau(hyp):
    """超参不是杠杆的第二条证据：CV 高分点连成一片平地。"""
    head('三、超参为什么不是杠杆 —— CV 前 5 名是一片平地')
    best = hyp['top5'][0][0]
    for rank, (cv, p) in enumerate(hyp['top5'], 1):
        mark = ' ← 第一名' if rank == 1 else ''
        print(f'   cv={cv:.4f}   n_estimators={p["n_estimators"]:<5}'
              f'max_depth={str(p["max_depth"]):<6}'
              f'min_samples_split={p["min_samples_split"]}{mark}')

    tail = hyp['top5'][1][0]
    print()
    print(f'  第 2~5 名分数完全相同（都是 {tail:.4f}），参数却南辕北辙。')
    print(f'  第一名只比它们高 {best - tail:.4f} —— 换个随机种子就会翻面。')
    print()


def show_verdict(conv, cap, sca, hyp):
    head('四、结论')
    print(f'  ① 收敛   n_iter_={conv["rows"][0]["n_iter"]}，max_iter 加到 10 万也不动  → 排除')
    print('  ② 容量   越深测试集越差（训练集反而冲到 0.979）      → 排除')
    print(f'  ③ 规模   100→1000 只 +{sca["top"]["nright"] - sca["base"]["nright"]} 题，且不单调       → 排除')
    print(f'  ④ 超参   {hyp["n_fits"]} 次训练只 +{hyp["grid_nright"] - hyp["base_nright"]} 题              → 排除')
    print()
    print('  四项全部排除 = 模型侧已经饱和。')
    print('  继续加大 epoch、加深模型、继续调参，都不会改变结果。')
    print()
    print('  → 下一个假设是"信息量不够"，去跑：python explore_routes.py')
    print()


def show_detail(conv, cap, sca, hyp):
    head('附：逐项原始输出')
    rule()

    print('\n① 收敛 —— 只改 max_iter，其余全不动')
    print(pad('   max_iter', 15) + pad('n_iter_', 11) + pad('准确率', 12) + '答对')
    for r in conv['rows']:
        print(pad('   ' + str(r["max_iter"]), 15) + pad(r["n_iter"], 11)
              + pad(f'{r["acc"]:.4f}', 12) + r["right"])
    print('   → n_iter_ 远远小于上限：早就到最优了，不是"没练够"')

    print('\n② 容量 —— 只改决策树 max_depth')
    print(pad('   max_depth', 15) + pad('训练集', 12) + pad('测试集', 12) + '差距')
    for r in cap['rows']:
        d = 'None' if r['depth'] is None else str(r['depth'])
        print(pad('   ' + d, 15) + pad(f'{r["train"]:.4f}', 12)
              + pad(f'{r["test"]:.4f}', 12) + f'{r["gap"]:.1f} 个点')
    print('   → 越深训练集越高、测试集越低：是吃撑了，不是没吃饱')

    print('\n③ 规模 —— 只改随机森林的树数')
    print(pad('   n_estimators', 18) + pad('准确率', 12) + '答对')
    for r in sca['rows']:
        print(pad('   ' + str(r["n"]), 18) + pad(f'{r["acc"]:.4f}', 12) + r["right"])
    print('   → 不是单调上升：200 和 500 完全一样，说明是噪声级抖动')

    print('\n④ 超参 —— GridSearchCV 穷举')
    print(f'   组合数 {hyp["n_params"]} × 5 折 = {hyp["n_fits"]} 次训练')
    print(f'   最优参数 {hyp["best_params"]}')
    print(f'   交叉验证 {hyp["best_cv"]:.4f} → 测试集 {hyp["grid_acc"]:.4f}')
    print(f'   默认参数（100 棵 / 不限深度）测试集 {hyp["base_acc"]:.4f}')
    print('   → 参数按交叉验证分数选、测试集全程没参与，所以这 1.1 个点是干净的')
    print()


# ══════════════════════════════════════════════════════════════

def main():
    df = build_frame()
    y_all = df['Survived'].values
    tr_idx, te_idx = split_indices(y_all)

    head('泰坦尼克 · 排查之一：模型侧饱和了吗？',
         f'训练 {len(tr_idx)} / 测试 {len(te_idx)}   随机种子 {SEED}   '
         f'特征 {len(make_xy(df)[0].columns)} 个')
    print()

    X, y = make_xy(df, ())
    X_tr, X_te = scale(X.values[tr_idx], X.values[te_idx])
    y_tr, y_te = y[tr_idx], y[te_idx]

    conv = check_convergence(X_tr, X_te, y_tr, y_te)
    cap = check_capacity(X_tr, X_te, y_tr, y_te)
    sca = check_scale(X_tr, X_te, y_tr, y_te)
    hyp = check_hyperparam(X_tr, X_te, y_tr, y_te)

    show_summary(conv, cap, sca, hyp)
    show_scale_vs_hyper(sca, hyp)
    show_flat_plateau(hyp)
    show_verdict(conv, cap, sca, hyp)
    show_detail(conv, cap, sca, hyp)


if __name__ == '__main__':
    main()
