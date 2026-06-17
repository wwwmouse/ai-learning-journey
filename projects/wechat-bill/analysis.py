import pandas as pd
import matplotlib.pyplot as plt
import os

def save_chart(filename):
    """保存图表到 results-pictures 文件夹"""
    base_dir = os.path.dirname(__file__)
    save_dir = os.path.join(BASE_DIR, 'results-pictures')
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(
        os.path.join(save_dir, filename),
        dpi=300,
        bbox_inches='tight'
    )
#保存图片，封装成函数

from matplotlib import rcParams
rcParams['font.family']= 'SimHei'
#windows处理中文

# 代码文件所在目录（E:\VScode\wechat-bill）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 拼接路径
data_path = os.path.join(BASE_DIR, 'datas.xlsx')
preview = pd.read_excel(data_path, header=None, nrows=50)#先截前50行找数据起点位置

header_row = None
for i in range(len(preview)):
    if str(preview.iloc[i, 0]) == '交易时间':
        header_row = i #找到数据列索引那一行对应的行索引
        break
    
del preview #手动释放

datas = pd.read_excel(data_path, header=header_row)#从列索引那一行开始读

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
#print(datasm)

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

plt.title('月度收支趋势图',fontsize=20,color='black')

plt.xlabel('时间',fontsize=15,color='black',labelpad=10)
plt.ylabel('金额',fontsize=15,color='black',labelpad=15,rotation=0)#让纵轴标题水平过来

plt.legend(loc='upper left')

plt.yticks(rotation=45,fontsize=12)
plt.xticks(rotation=45,fontsize=12)

plt.grid(True,alpha=0.2,color='black')
#添加网格线，方向、透明度、颜色

for x,y in enumerate(expend):
    plt.text(x+0.1,y+0.1,str(y),ha="left",va="bottom",fontsize=10,color='red',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
    
for x,y in enumerate(income):
    plt.text(x-0.1,y-0.1,str(y),ha="right",va="top",fontsize=10,color='blue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.8))
    
plt.tight_layout()
save_chart("月度收支趋势图.png")
plt.show()

# 2.绘制消费时间条形图
plt.figure(figsize=(15,8))
datas['交易小时']=datas['交易时间'].dt.hour
datash=datas.groupby(["交易小时","收/支"],as_index=False)["金额(元)"].sum()
#分组聚合成长表，分别每个小时的支出总和与收入总和

expend=datash[datash['收/支']=='支出'].round(2)
#筛选支出总和，制成新表

expend=expend.pivot(index='交易小时',columns='收/支',values='金额(元)')
#以交易小时为index,收/支为column(此时横轴只剩支出),金额为values制成新表

expend=expend.reindex(range(24), fill_value=0)
#存在有些时间段没有任何支出，需要在这里补齐0-23小时，没有的填0

expend=expend.reset_index()
#将'交易小时'从index变为普通列

expend.columns.name=None
#去掉columns.name收/支

print(expend)


print(f"消费金额最大的小时:{expend.iloc[expend['支出'].idxmax(),0]}")
print(f"该小时消费总金额:{round(expend['支出'].max(),2)}")

day_hours=expend["交易小时"]
day_money=expend["支出"].round(2)

colors = ['#2c3e50' if h < 6 else    # 凌晨深蓝
          '#f39c12' if h < 12 else   # 上午橙色
          '#27ae60' if h < 18 else   # 下午绿色
          '#e74c3c' for h in day_hours]  # 晚上红色

plt.bar(day_hours, day_money, color=colors, width=0.5)

plt.title("消费时段分布图",fontsize=20,color="black")
plt.xlabel("小时",fontsize=15,color="black",labelpad=10)
plt.ylabel("支出",fontsize=15,color="black",rotation=0,labelpad=15)

plt.yticks(rotation=45,fontsize=12)
plt.xticks(fontsize=12)

plt.ylim(0,max(day_money)+501)
plt.grid(True, axis='y', alpha=0.6, color="gray")

for x,y in enumerate(day_money):
     plt.text(x,y,str(y),ha="center",va="bottom",fontsize=10,color='black')
     
plt.tight_layout()
save_chart("消费时段分布图.png")
plt.show()