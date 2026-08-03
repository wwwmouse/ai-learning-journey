import os

# 数据处理
from sklearn.model_selection import GridSearchCV
import joblib

# 模型
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 评估
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 画图
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# 项目根目录（脚本所在目录的上一级）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 数据预处理（从 preprocess.py 导入）==========
from preprocess import load_and_preprocess
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

#   scikit-learn 里所有模型的用法完全统一:
#   model = 某模型(参数)      1. 选模型（还没开始学）
#   model.fit(X_train, y_train)    2. 学（喂题目+答案）
#   y_pred = model.predict(X_test)   3. 考（只看题目，不看答案）

#    训练逻辑回归模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) # 预测结果
print('准确率:', accuracy_score(y_test, y_pred)) # 对比预测结果与标准答案，计算准确率
# 准确率0.794776
# 我用最直白最通俗最容易懂最不饶弯子的话告诉讲明白：这玩意就是个废物

print('\n混淆矩阵:')
print(confusion_matrix(y_test, y_pred)) # 直接根据预测结果和最终结果计算混淆矩阵
#    混淆矩阵：
#                 预测死亡  预测存活
#    实际死亡   [[   140      25]
#    实际存活    [   30       73]]

print('\n分类报告:')
print(classification_report(y_test, y_pred)) # 同上，直接计算分类报告

# 分类报告:                                                                                                                            
#        precision    recall  f1-score   support

#  0       0.82      0.85      0.84       165
#  1       0.74      0.71      0.73       103

#   support是每一行数据的总数，在这里指的是实际死亡165、实际存活103

# 类别 0（死亡）：
#   precision=0.82(死亡精确率)：模型说"死亡"的 140+30=170 人里，140 人真的死了 140/170=82%
#   recall=0.85(死亡召回率)：    165 个真死的人里，模型找出了 140 个 140/165=85%
#
# 类别 1（存活）：
#   precision=0.74(存活精确率)：模型说"存活"的 25+73=98 人里，73 人真的活了 → 说"活"时 26% 在瞎说
#   recall=0.71(存活召回率)：    103 个真活的人里，模型只找到 73 个 → 漏了 30 个（29%）

# 绝大多数情况下，召回率和精确率为反比例
# 但我们需要两者在可以接受的范围内都尽可能高，因此我们需要一个可以综合评价模型的数值

# f1-score ≈ 0.73：精确率(0.74)和召回率(0.71)的调和平均
# 调和平均数的特点是会"惩罚"偏科的情况，因此更能综合评价模型能力
# 模型明显更擅长判断"死亡"（f1=0.84），判断"存活"比较吃力（f1=0.73）
# 说明模型整体倾向于保守预测——宁判死，不漏活

#   多种模型对比
models = {
      'KNN': KNeighborsClassifier(n_neighbors=5), 
      #KNN 找最近的 5 个人投票，太小了不稳定，太大了模糊
      '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
      #迭代1000次，各部分权重收敛时停止
      '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
      #决策树最多问 5 层问题就停。不加这个限制树会疯长到每张叶子只放一个人（过拟合）
      '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
      #n_estimators=100：随机森林种 100 棵树投票。越多越稳，但越慢
  }
print('\n' + '='*50)
print('四模型对比')
print('='*50)

for name, model in models.items():
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      acc = accuracy_score(y_test, y_pred)
      print(f'\n--- {name} ---')
      print(f'准确率: {acc:.4f}')
      print(classification_report(y_test, y_pred))

# --- KNN ---
# 准确率: 0.7873
#               precision    recall  f1-score   support

#            0       0.81      0.85      0.83       165
#            1       0.74      0.68      0.71       103

#     accuracy                           0.79       268
#    macro avg       0.78      0.77      0.77       268
# weighted avg       0.79      0.79      0.79       268

# --- 逻辑回归 ---
# 准确率: 0.7948
#               precision    recall  f1-score   support

#            0       0.82      0.85      0.84       165
#            1       0.74      0.71      0.73       103

#     accuracy                           0.79       268
#    macro avg       0.78      0.78      0.78       268
# weighted avg       0.79      0.79      0.79       268

# --- 决策树 ---
# 准确率: 0.7910
#               precision    recall  f1-score   support

#            0       0.79      0.91      0.84       165
#            1       0.81      0.60      0.69       103

#     accuracy                           0.79       268
#    macro avg       0.80      0.76      0.77       268
# weighted avg       0.79      0.79      0.78       268

# --- 随机森林 ---
# 准确率: 0.7910
#               precision    recall  f1-score   support

#            0       0.80      0.87      0.84       165
#            1       0.76      0.66      0.71       103

#     accuracy                           0.79       268
#    macro avg       0.78      0.77      0.77       268
# weighted avg       0.79      0.79      0.79       268

results = {
      'KNN':      0.7873,
      '逻辑回归': 0.7948,
      '决策树':   0.7910,
      '随机森林': 0.7910,
  }

plt.figure(figsize=(8, 5))
names = list(results.keys())
scores = list(results.values())
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = plt.bar(names, scores, color=colors)

# 柱子上方标数字
for bar, score in zip(bars, scores):
      plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{score:.4f}', ha='center', va='bottom', fontsize=12)

plt.ylim(0.75, 0.82)   # 把 y 轴范围缩窄，差值更明显
plt.ylabel('准确率')
plt.title('四模型准确率对比')
plt.savefig(os.path.join(PROJECT_DIR, '..', 'images', 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('已保存: model_comparison.png')

# 随机森林调参

# 先跑一个"基准版"随机森林
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    rf_model, X_test, y_test,
    cmap='Blues',           # 蓝色渐变，越深数字越大
    display_labels=['死亡', '存活']
)
plt.title('随机森林 - 混淆矩阵')
plt.savefig(os.path.join(PROJECT_DIR, '..', 'images', 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print('已保存: confusion_matrix.png')

# 正式开始调参

# 构建参数网络 
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
}
#  param_grid 中共计 3 × 4 × 3 = 36 种组合:
#  n_estimators : 森林里种多少棵树。少了模型不准，多了训练太慢。50/100/200 三个档位看拐点在哪
#  max_depth : 每棵树最多问几层问题。None 是不限制，让树自由生长；限制到 5 或 10 可以防止那棵树死记硬背训练集
#  min_samples_split : 一个节点至少要有几个人才继续分叉。设成 10 的意思是"少于 10 个人的组别再细分"，防止过拟合。

# 利用参数网络进行调参，寻找最优参数组合
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
# GridSearchCV 做的事：param_grid中的36种组合暴力穷举找最优
# 每种用 5 折交叉验证评估，选分数最高的那个。
# cv=5 把训练集再切5份轮流转，比单次划分更可信。n_jobs=-1 并行跑，CPU 有几个核就同时跑几个

grid.fit(X_train, y_train)

print('\n====== 调参结果 ======')
print(f'最佳参数: {grid.best_params_}')
print(f'最佳交叉验证分数: {grid.best_score_:.4f}')
print(f'测试集分数: {grid.score(X_test, y_test):.4f}')
#   最佳参数: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
#   最佳交叉验证分数: 0.8330  (在训练过程中测的)
#   测试集分数: 0.8022       (在真正的"陌生人"上测的)

best_model = grid.best_estimator_ # 记录最优模型
y_pred_best = best_model.predict(X_test) # 记录最优模型的测试答案
print('\n调参后分类报告:')
print(classification_report(y_test, y_pred_best)) # 输出最优模型的分类报告
# 调参后分类报告:
#             precision    recall  f1-score   support

#         0       0.81      0.89      0.85       165
#         1       0.79      0.66      0.72       103
# accuracy                            0.80       268
# macro avg       0.80      0.78      0.78       268
# weighted avg    0.80      0.80      0.80       268

# 保存模型
models_dir = os.path.join(PROJECT_DIR, '..', 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(models_dir, 'rf_model.pkl'))
print(f'模型已保存: models/rf_model.pkl')


# 准确率从 79.1% 涨到 80.2%，涨了 1%。不多，但在 ML 里每涨一点都是真的提升了——不是运气
# 存活 recall 还是 66%，没动。这说明 66% 可能是你这 7 个特征能达到的天花板——漏掉的 34%
# 是那些特征上看起来"应该死但活了"的人（比如"末等舱的年轻女性"），现有特征无法区分，除非你构造新的特征。


# ========== 随机森林 - 特征重要性 ==========
# feature_names 由 load_and_preprocess() 返回，不需要手动写了
importances = best_model.feature_importances_

# 特征重要性可视化
sorted_idx = importances.argsort()[::-1]
# argsort() 从小到大排 -1将整个列表倒过来
# 最终就是重要性由高到低的编号
print('\n====== 特征重要性 ======')
for i in sorted_idx:
    print(f'  {feature_names[i]:12s}  {importances[i]:.4f}')

# 画横柱图（带数字标注）
plt.figure(figsize=(8, 5))
names_sorted = [feature_names[i] for i in sorted_idx]
values_sorted = importances[sorted_idx]
bars = plt.barh(names_sorted, values_sorted, color='#2ecc71')

# 每根柱子右端标具体数值
for bar, val in zip(bars, values_sorted):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='left', va='center', fontsize=11)

plt.xlabel('重要性')
plt.title('随机森林 - 特征重要性排名')
plt.savefig(os.path.join(PROJECT_DIR, '..', 'images', 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print('已保存: feature_importance.png')