import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
#指定用微软雅黑或黑体渲染文字
matplotlib.rcParams['axes.unicode_minus'] = False
  
df=pd.read_csv("AI-learning/projects/titanic_project/data/train.csv")

print(df.head())
print(df.shape) #(891,12)
# print(df.dtypes) 

print(df.isna().sum())
print(df.isna().mean()*100)
# age缺失19.865320
# cabin确实77.104377

print(df.describe())
# age 
# min 0.4200
# max 80.0000
# mean 29.6991

# fare
# min 0.0000
# max 512.3292
# mean 32.2042

#存活统计

print(f"生存率:{df['Survived'].mean()}")
sns.countplot(x='Survived', data=df)
plt.title('生存人数统计 (0=死亡, 1=存活)')
plt.savefig('AI-learning/projects/titanic_project/images/survival_count.png', dpi=150, bbox_inches='tight')

#性别VS存活

male_target=df[df['Sex']=="male"]
print(f"男性存活率:{male_target['Survived'].mean()}")

female_target=df[df['Sex']=="female"]
print(f"女性存活率:{female_target['Survived'].mean()}")

plt.figure()  # 新开一张画布，不和上一张重叠
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('性别与生存关系')
plt.savefig('AI-learning/projects/titanic_project/images/sex_survival.png', dpi=150, bbox_inches='tight')

#舱位等级VS存活

print(f"一等舱存活率:{df[df['Pclass']==1]['Survived'].mean()}")
print(f"二等舱存活率:{df[df['Pclass']==2]['Survived'].mean()}")
print(f"三等舱存活率:{df[df['Pclass']==3]['Survived'].mean()}")
plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('舱位等级与生存关系')
plt.savefig('AI-learning/projects/titanic_project/images/pclass_survival.png', dpi=150, bbox_inches='tight')

#年龄 vs 存活

plt.figure()
sns.histplot(
    x='Age',        #横轴：年龄
    hue='Survived', #按是否存活分层着色
    data=df,        #数据来源
    kde=True,       #加一条平滑趋势线
    bins=30,        #切成 30 个年龄区间统计
    multiple='stack'#死亡和存活的柱子垒起来，别叠在一起
    )
plt.title('年龄分布与生存关系')
plt.savefig('AI-learning/projects/titanic_project/images/age_survival.png', dpi=150, bbox_inches='tight')