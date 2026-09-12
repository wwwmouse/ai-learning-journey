"""
泰坦尼克项目 · 排查之二：候选路线往哪走？

【这个脚本回答什么】
    模型侧已经饱和（见 check_saturation.py），那还能怎么办？
    加信息、调阈值、还是类别加权 —— 哪条路真的有用？

【和 check_saturation.py 的关系】
    前半程 check_saturation.py          本脚本 explore_routes.py
    ├ 判定"模型侧还有没有空间"          ├ 假设已排除，换方向找信息
    ├ 回答"还能不能靠调参救"            ├ 回答"往哪加信息、值不值"
    └ 结论：饱和 → 调参没用了            └ 结论：加称谓特征才跳出噪声带

    顺序不能反。不先把"调参"这条路走死就说"要加特征"，那是拍脑袋 ——
    这正是项目文档里那条元教训：排查顺序错了，归因就会错。

【覆盖四条路线】
    ① 基线          7 个特征，随机森林 / 逻辑回归
    ② 调阈值        在验证集上选切点（不产生新信息）
    ③ 类别加权      class_weight='balanced'（能提召回，代价是精确率）
    ④ 加特征        +「称谓」one-hot（真的增加了信息）
      （另加对照组：+「家庭规模」，一个信息量为零的假特征）

【每条路线的性质 —— 这是本脚本的核心结论】
    ② ③ 是在**同一条 PR 曲线内滑动**：把一个指标换给另一个，F1 几乎不动。
    ④ 是把**整条 PR 曲线往外推**：准确率和精确率同时上升。
    这就是为什么只有加特征能真正跳出噪声带。

【运行】
    cd src
    python explore_routes.py
"""

from numpy import isin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from titanic_common import (SEED, W, head, rule, pad,
                            build_frame, make_xy, split_indices, split_val,
                            scale, metrics, right, pick_threshold)

# 路线性质标注（结论来自实测，不是先验判断）
NATURE = {
    'base':      '基线',
    'threshold': '曲线内滑动（不产生新信息）',
    'weight':    '曲线内滑动（提召回，掉精确率）',
    'feature':   '曲线外推（真的增加了信息）',
    'fake':      '零信息（只是列数变多）',
}


def run_set(df, tr_idx, te_idx, tr_sub, val_sub, extra):
    """跑一组特征下的全部路线，返回每行结果（dict 列表）。

    每条路线只改一件事，其余全部锁死 —— 这样"没涨"才能干净地归因。
    """
    X, y = make_xy(df, extra)
    # 必须标准化后再切分：逻辑回归对尺度敏感，不标准化结果会掉；
    # 随机森林对尺度不敏感，所以漏掉这一步时森林那行看不出来，专坑逻辑回归
    X_tr, X_te = scale(X.values[tr_idx], X.values[te_idx])
    y_tr, y_te = y[tr_idx], y[te_idx]
    tr_mask, val_mask = isin(tr_idx, tr_sub), isin(tr_idx, val_sub)

    rows = []
    for model_name, make in [
        ('逻辑回归', lambda: LogisticRegression(max_iter=1000, random_state=SEED)),
        ('随机森林', lambda: RandomForestClassifier(n_estimators=100, random_state=SEED)),
    ]:
        # ① 基线：默认阈值 0.50，用全部 623 训练
        m = make().fit(X_tr, y_tr)
        rows.append(dict(model=model_name, route='基线（阈值 0.50）', nature='base',
                         m=metrics(y_te, m.predict(X_te)),
                         right=right(y_te, m.predict(X_te)), note=''))

        # ② 调阈值：只在子训练集上选阈值（隔离出验证集），
        #    选完再用全部 623 重训一次，最后才碰测试集
        m_sub = make().fit(X_tr[tr_mask], y_tr[tr_mask])
        t, val_f1 = pick_threshold(m_sub, X_tr[val_mask], y_tr[val_mask])
        m_full = make().fit(X_tr, y_tr)
        pred = (m_full.predict_proba(X_te)[:, 1] >= t).astype(int)
        rows.append(dict(model=model_name, route=f'调阈值（验证集选出 {t:.2f}）',
                         nature='threshold', m=metrics(y_te, pred),
                         right=right(y_te, pred), note=f'验证集F1={val_f1:.3f}'))

        # ③ 类别加权：其他全不动，只加这一个参数
        m_bal = make()
        m_bal.set_params(class_weight='balanced')
        m_bal.fit(X_tr, y_tr)
        rows.append(dict(model=model_name, route="class_weight='balanced'",
                         nature='weight', m=metrics(y_te, m_bal.predict(X_te)),
                         right=right(y_te, m_bal.predict(X_te)), note=''))
    return rows


def show_section(title, _nature, rows):
    """打印一组路线。"""
    print(f'\n【{title}】')
    print(pad('   路线', 39) + pad('准确率', 10, '>') + pad('存活P', 9, '>')
          + pad('存活R', 9, '>') + pad('存活F1', 9, '>') + pad('答对', 12, '>')
          + pad('验证集', 14))
    for r in rows:
        a, p, rc, f1 = r['m']
        print(pad('   ' + f'{r["model"]} · {r["route"]}', 39)
              + pad(f'{a:.4f}', 10, '>') + pad(f'{p:.4f}', 9, '>')
              + pad(f'{rc:.4f}', 9, '>') + pad(f'{f1:.4f}', 9, '>')
              + pad(r['right'], 12, '>') + pad(r['note'], 16))


def main():
    df = build_frame()
    y_all = df['Survived'].values
    tr_idx, te_idx = split_indices(y_all)
    tr_sub, val_sub = split_val(tr_idx, y_all)

    head('泰坦尼克 · 排查之二：候选路线（模型侧已饱和之后）',
         f'训练 {len(tr_idx)} / 验证 {len(val_sub)} / 测试 {len(te_idx)}   '
         f'随机种子 {SEED}')

    base = run_set(df, tr_idx, te_idx, tr_sub, val_sub, ())
    title = run_set(df, tr_idx, te_idx, tr_sub, val_sub, ('Title',))
    fake = run_set(df, tr_idx, te_idx, tr_sub, val_sub, ('FamilySize',))

    show_section('路线 ①②③ —— 7 个特征（基线）', 'base', base)
    show_section('路线 ④ —— +「称谓」one-hot（真特征）', 'feature', title)
    show_section('对照组 —— +「家庭规模」（假特征，信息量为零）', 'fake', fake)

    # ── 核心结论：基线 vs 加特征，逐模型对照 ──
    print()
    head('结论：哪条路真的有用？')
    print(pad('   模型', 15) + pad('手段', 24) + pad('准确率', 10, '>')
          + pad('答对', 11, '>') + '相对本模型基线')
    rule()

    for i, model_name in enumerate(['逻辑回归', '随机森林']):
        b = base[i * 3]            # 该模型的基线行
        b_acc, b_n = b['m'][0], int(b['right'].split('/')[0])
        w = base[i * 3 + 2]        # 类别加权
        th = base[i * 3 + 1]       # 调阈值
        f = title[i * 3]           # 加称谓后的基线

        for label, row in [(f'{model_name} · 基线', b),
                           (f'{model_name} · 调阈值', th),
                           (f'{model_name} · 类别加权', w),
                           (f'{model_name} · +称谓', f)]:
            a = row['m'][0]
            n = int(row['right'].split('/')[0])
            delta = '' if a == b_acc else f'{n - b_n:+d} 题'
            print(pad('   ' + label, 39) + pad(f'{a:.4f}', 10, '>')
                  + pad(f'{n}/268', 11, '>') + pad(delta, 10))

    # 加特征相对基线的提升（这是唯一的大跳跃）
    print()
    print('   三条路线的性质完全不同：')
    for i, model_name in enumerate(['逻辑回归', '随机森林']):
        b = base[i * 3]
        th = base[i * 3 + 1]
        w = base[i * 3 + 2]
        f = title[i * 3]
        b_f1, th_f1, w_f1, f_f1 = b['m'][3], th['m'][3], w['m'][3], f['m'][3]
        b_p, w_p = b['m'][1], w['m'][1]
        print(f'\n   · {model_name}')
        print(f'     调阈值     F1 {b_f1:.3f} → {th_f1:.3f}'
              f'（准确率 {b["m"][0]:.4f} → {th["m"][0]:.4f}）'
              f'  —— 只是换切点，把漏判和误判互换')
        print(f'     类别加权   F1 {b_f1:.3f} → {w_f1:.3f}'
              f'（精确率 {b_p:.3f} → {w_p:.3f} 掉了）'
              f'  —— 在同一条曲线上滑动')
        print(f'     加称谓     F1 {b_f1:.3f} → {f_f1:.3f}'
              f'（准确率 {b["m"][0]:.4f} → {f["m"][0]:.4f}，'
              f'精确率 {b_p:.3f} → {f["m"][1]:.3f}）'
              f'  —— 整条曲线往外推')

    # ── 假特征对照 ──
    print()
    head('附：为什么"加列数"不等于"加信息量"')
    print('   「家庭规模」= SibSp + Parch + 1，是已有两列的线性组合，信息量为零。')
    print()
    print(pad('   模型', 14) + pad('7 个特征', 14) + pad('+家庭规模', 14) + '变化')
    for i, model_name in enumerate(['逻辑回归', '随机森林']):
        b = base[i * 3]
        k = fake[i * 3]
        same = abs(b['m'][0] - k['m'][0]) < 1e-12
        mark = '完全一样' if same else f'{k["m"][0]:.4f}'
        print(pad('   ' + model_name, 14) + pad(f'{b["m"][0]:.4f}', 14)
              + pad(mark, 14)
              + ('一位小数都不动 → 加列数≠加信息量' if same else '有变化'))
    print()
    print('   → 加特征要看是否真的增加了信息，不是列数变多。')
    print('     （踩过的坑：给称谓用 LabelEncoder 编成 0/1/2/3，等于硬塞了')
    print('       「Mr < Miss < Mrs < Master」这种不存在的次序，结论完全反了。）')
    print()
    print('   → 推荐顺序：加特征 > 类别加权（必须提召回时用）> 调参/调阈值（已实测无用）')
    print('     回到前半程：python check_saturation.py')
    print()


if __name__ == '__main__':
    main()
