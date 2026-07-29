import os

# 数据处理
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# ========== 数据预处理（与 preprocess.py 相同）==========
df = pd.read_csv(os.path.join(PROJECT_DIR, '..', 'data', 'train.csv'))
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#   训练逻辑回归模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('准确率:', accuracy_score(y_test, y_pred))
# 准确率0.794776
# 我用最直白最通俗最容易懂最不饶弯子的话告诉讲明白：这玩意就是个废物

print('\n混淆矩阵:')
print(confusion_matrix(y_test, y_pred))
#    混淆矩阵：
#                 预测死亡  预测存活
#    实际死亡   [[   140      25]
#    实际存活    [   30       73]]

print('\n分类报告:')
print(classification_report(y_test, y_pred))

# 分类报告:                                                                                                                            
# precision    recall  f1-score   support

#  0       0.82      0.85      0.84       165
#  1       0.74      0.71      0.73       103

#   precision（准确率）= 你说它活了，它真的活了吗
#   模型说"这人能活"的 73+25=98 个人里，只有 73 个真的活了。74% 的精确率意思是：模型判存活的时候，有 26% 的概率在瞎说。

#   recall（召回率）= 真活着的人，你找到了几个
#   103 个真活着的人里，模型只找到了 73 个，漏了 30 个。71% 的召回率就是"存活的人里漏了 29%"。

#   f1-score = 精确率和召回率的中间平衡值
#   73% 是 74% 和 71% 的调和平均。这个值不高，说明模型对"存活"的判断比较吃力——更倾向于判死亡。

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


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
}
#  param_grid 需要搜 3 × 4 × 3 = 36 种组合:
#  n_estimators — 森林里种多少棵树。少了不准、多了慢。50/100/200 三个档位看拐点在哪
#  max_depth — 每棵树最多问几层问题。None 是不限制，让树自由生长；限制到 5 或 10 可以防止那棵树死记硬背训练集
#  min_samples_split — 一个节点至少要有几个人才继续分叉。设成 10 的意思是"少于 10 个人的组别再细分了，没意义"

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
# GridSearchCV 做的事：36 种组合，每种用 5 折交叉验证评估，选分数最高的那个。cv=5 把训练集再切 5
# 份轮流转，比单次划分更可信。n_jobs=-1 并行跑，你 CPU 有几个核就同时跑几个
grid.fit(X_train, y_train)

print('\n====== 调参结果 ======')
print(f'最佳参数: {grid.best_params_}')
print(f'最佳交叉验证分数: {grid.best_score_:.4f}')
print(f'测试集分数: {grid.score(X_test, y_test):.4f}')

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print('\n调参后分类报告:')
print(classification_report(y_test, y_pred_best))

# 最佳参数: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
#   max_depth=10：每棵树最多问 10 层问题。
#   之前你手填的是 None（不限制），现在发现限到 10反而更好——不限制会让树"背题目"（过拟合），10 层刚好够又不至于背
#   min_samples_split=2：一个节点只要有 2 个人就可以继续分裂。
#   搜了 5 和 10，都不如 2 好——你的数据只有 623行训练集，设大了节点太少，学不够
#   n_estimators=200：200 棵树比 100 棵好，说明多树投票在这个数据上还能再稳一点

# 最佳交叉验证分数: 0.8330  在训练过程中测的
# 测试集分数: 0.8022       在真正的"陌生人"上测的

# 调参后分类报告:
#             precision    recall  f1-score   support

#         0       0.81      0.89      0.85       165
#         1       0.79      0.66      0.72       103
# accuracy                            0.80       268
# macro avg       0.80      0.78      0.78       268
# weighted avg    0.80      0.80      0.80       268

# 准确率从 79.1% 涨到 80.2%，涨了 1%。不多，但在 ML 里每涨一点都是真的提升了——不是运气
# 存活 recall 还是 66%，没动。这说明 66% 可能是你这 7 个特征能达到的天花板——漏掉的 34%
# 是那些特征上看起来"应该死但活了"的人（比如"末等舱的年轻女性"），现有特征无法区分，除非你构造新的特征。


# ========== 随机森林 - 特征重要性 ==========
# 注意：X_train 经过 StandardScaler 后变成了 numpy 数组，列名丢失。
# 这里按 preprocess 的顺序手动重建列名。顺序错了重要性就对不上！
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
importances = best_model.feature_importances_

# 从大到小排序打印
sorted_idx = importances.argsort()[::-1]
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