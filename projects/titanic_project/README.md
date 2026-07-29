# 泰坦尼克号生存预测

基于 scikit-learn 的传统机器学习分类项目，对比四种模型预测泰坦尼克号乘客存活情况。

## 项目结构

```
titanic_project/
├── data/train.csv          # 原始数据（891行×12列）
├── src/
│   ├── explore.py          # 数据探索与可视化
│   ├── preprocess.py       # 数据预处理
│   └── train.py            # 模型训练、对比、调参、评估
├── images/                 # 所有输出图表（6张）
├── requirements.txt
├── README.md
└── LEARNING.md
```

## 运行方法

项目使用 `os.path` 定位数据和图片目录，无需刻意切换工作目录。以下任一方式均可：

```bash
# 方式一：从项目根目录运行
python src/explore.py
python src/preprocess.py
python src/train.py

# 方式二：进入 src 目录运行
cd src
python explore.py
python preprocess.py
python train.py

# 方式三：从任意目录直接用绝对路径运行
python /path/to/titanic_project/src/explore.py
```

## 方法

- 数据预处理：删无用列 → 中位数填充 → 类别编码 → 标准化
- 模型对比：KNN / 逻辑回归 / 决策树 / 随机森林
- 调参：GridSearchCV（5折交叉验证）
- 评估指标：准确率、精确率、召回率、F1、混淆矩阵

## 结果

| 模型 | 准确率 | 存活 F1 |
|------|--------|---------|
| KNN | 78.7% | 0.71 |
| 逻辑回归 | 79.5% | 0.73 |
| 决策树 | 79.1% | 0.69 |
| 随机森林（默认） | 79.1% | 0.71 |
| 随机森林（调参后） | 80.2% | 0.72 |

## 结论

逻辑回归在本数据集上 F1 最均衡。性别和舱位等级是强线性信号，逻辑回归天然适配。

随机森林调参后准确率最高（80.2%），但存活召回率始终在 66% 左右——现有 7 个特征无法区分那些「按常理应该死却活了」的人。提升的真正方向不是继续调参，而是构造新特征（家庭规模、姓名称谓等）。

## 学习总结

见 `LEARNING.md`
