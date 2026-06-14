import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib import rcParams
rcParams['font.family']= 'SimHei'
#windows处理中文

preview = pd.read_excel('projects\wechat-bill\datas.xlsx', header=None, nrows=50)#先截前50行找数据起点位置

header_row = None
for i in range(len(preview)):
    if str(preview.iloc[i, 0]) == '交易时间':
        header_row = i #找到数据列索引那一行对应的行索引
        break
    
del preview #手动释放

datas = pd.read_excel('projects\wechat-bill\datas.xlsx', header=header_row)#从列索引那一行开始读

print(datas.isna().sum())
# 空白检查，确认每列都没有空的

print(datas.duplicated().sum())
# 重复检查，确认没重复内容

datas['交易年月']=datas['交易时间'].dt.to_period('M')
# 保存交易年月

datas = datas[datas['交易年月']!='2026-06']
datas = datas[~datas['收/支'].str.contains('/', na=False)]
datas = datas[~datas['交易对方'].str.contains('/', na=False)]
datas = datas[~datas['商品'].str.contains('/', na=False)]
# 扔掉2026-06的数据
# 去掉收支、交易对方和商品列中的特殊项'/'


# 1.绘制月份收支折线图
# print(datas.dtypes)
# 检查数据类型发现时间已经是datetime64[us]类型了

datas=datas.sort_values('交易年月')
datasm=datas.groupby(['交易年月','收/支'],as_index=False)['金额(元)'].sum()
#分组聚合，形成每个月总金额图
print(datasm)

datasm_w=datasm.pivot(index='交易年月',columns='收/支',values='金额(元)')
datasm_w=datasm_w.reset_index() #令交易年月从索引变为普通列，不然没法从列取时间
datasm_w.columns.name=None

print(datasm_w)
# 长表变宽表，分开收/支
expend=datasm_w['支出'].round(2)
income=datasm_w['收入'].round(2)

times=datasm_w['交易年月']
times=times.astype(str)
plt.figure(figsize=(15,8))

plt.plot(
    times,
    income,
    marker='o',
    linestyle='--',
    color='blue',
    label='收入',
    linewidth=2
)
plt.plot(
    times,
    expend,
    marker='o',
    linestyle='--',
    color='red',
    label='支出',
    linewidth=2
)

plt.title('2025-09至2026-05资金流动',fontsize=20,color='black')

plt.xlabel('时间',fontsize=15,color='black')
plt.ylabel('金额',fontsize=15,color='black',rotation=0)#让纵轴标题水平过来

plt.legend(loc='upper left')

plt.xticks(rotation=45,fontsize=12)

plt.grid(True,alpha=0.2,color='black')
#添加网格线，方向、透明度、颜色

for x,y in enumerate(expend):
    plt.text(x+0.1,y+0.1,str(y),ha="left",va="bottom",fontsize=10,color='red',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8))
    
for x,y in enumerate(income):
    plt.text(x-0.1,y-0.1,str(y),ha="right",va="top",fontsize=10,color='blue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
    
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '资金流动图.png'), dpi=300, bbox_inches='tight')
plt.show()

