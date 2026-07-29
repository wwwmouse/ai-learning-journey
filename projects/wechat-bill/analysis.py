import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib import rcParams
rcParams['font.family']= 'SimHei'
# windows处理中文

# 代码文件所在位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def save_chart(filename):
#保存图表到 results-pictures 文件夹
    save_dir = os.path.join(BASE_DIR, 'results-pictures')
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(
        os.path.join(save_dir, filename),
        dpi=300,
        bbox_inches='tight'
    )


# ── 关键词分类 ──
# 按优先级从上到下匹配，命中即返回，未命中归入"未分类"
def classify_counterparty(name):
    name=str(name)

    # 外卖（放在餐厅前面，避免被"美食"等误匹配）
    if any(k in name for k in ['美团','饿了么','外卖','熊猫来了']):
        return '外卖'
    # 饮品
    if any(k in name for k in ['瑞幸','咖啡','奶茶','蜜雪','星巴克','茶百道']):
        return '饮品'
    # 餐饮（食堂/小吃/餐厅/常见菜名）
    if any(k in name for k in ['餐厅','食堂','美食','小吃','餐饮','大学','学院']):
        return '餐饮'
    # 超市/便利店
    if any(k in name for k in ['超市','便利店','商贸','商店','商场','每一天','京东','淘宝','拼多多','闲鱼']):
        return '购物'
    # 打印
    if any(k in name for k in ['打印','复印','快印','云印']):
        return '打印'
    # 交通
    if any(k in name for k in ['地铁','公交','滴滴','打车','出行','铁路','高德']):
        return '交通'
    # 商场/娱乐
    if any(k in name for k in ['赛格','影城','KTV','抖音','网易云音乐','快手','酷狗',]):
        return '娱乐'

    return '未分类'


# 拼接路径，读取datas.xlsx
data_path = os.path.join(BASE_DIR, 'datas.xlsx')

preview = pd.read_excel(data_path, header=None, nrows=50)#先截前50行找数据起点位置

header_row = None
for i in range(len(preview)):
    if str(preview.iloc[i, 0]) == '交易时间':
        header_row = i #找到数据列索引那一行对应的行索引
        break

# 上一步只读了前50行来找表头位置，现在用找到的行号正式读取全部数据
datas = pd.read_excel(data_path, header=header_row)#从列索引那一行开始读

# 数据清洗：去除关键列缺失、重复行、含'/'的异常行
# 空白检查，确认关键数据没有缺失值
print(datas.isna().sum())
core_data=['交易时间','收/支','金额(元)']
datas=datas.dropna(subset=core_data)

# 重复检查，确认没重复内容
print(f'重复{datas.duplicated().sum()}行')
datas=datas.drop_duplicates()

datas['交易年月']=datas['交易时间'].dt.to_period('M')
# 保存交易年月

# 去掉收支、交易对方和商品列中的特殊项'/'
datas = datas[~datas['收/支'].str.contains('/', na=False)]
datas = datas[~datas['交易对方'].str.contains('/', na=False)]
datas = datas[~datas['商品'].str.contains('/', na=False)]


# 1.绘制月份收支折线图
# print(datas.dtypes)
# 检查数据类型发现时间已经是datetime64[us]类型了

datas=datas.sort_values('交易年月')
monthly_summary=datas.groupby(['交易年月','收/支'],as_index=False)['金额(元)'].sum()
#分组聚合，形成每个月总金额图
#print(monthly_summary)

monthly_pivot=monthly_summary.pivot(index='交易年月',columns='收/支',values='金额(元)')
monthly_pivot=monthly_pivot.reset_index() #令交易年月从索引变为普通列，不然没法从列取时间
monthly_pivot.columns.name=None

print(monthly_pivot)
# 长表变宽表，分开收/支

expend=monthly_pivot['支出'].round(2)
income=monthly_pivot['收入'].round(2)

times=monthly_pivot['交易年月']
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
plt.ylabel('金额',fontsize=15,color='black',labelpad=15,rotation=0)
#让纵轴标题水平过来

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
#一定要新开一张图！plt.show()会重置画布大小

datas['交易小时']=datas['交易时间'].dt.hour
hourly_summary=datas.groupby(["交易小时","收/支"],as_index=False)["金额(元)"].sum()
#分组聚合成长表，分别每个小时的支出总和与收入总和

expend=hourly_summary[hourly_summary['收/支']=='支出'].round(2)
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


# 3.绘制消费分类饼图
expense = datas[datas['收/支']=='支出'].copy()
expense['类别'] = None  # 先占位，防止后面 isna() 空列报错

# 转账/红包/群收款 → 交易对方是个人，无法按商户归类，单独列出
transfer_mask = expense['交易类型'].isin(['转账','红包','群收款','微信红包'])
expense.loc[transfer_mask, '类别'] = '转账/红包'

# 其余的正常跑关键词
keyword_mask = expense['类别'].isna()
expense.loc[keyword_mask, '类别'] = expense.loc[keyword_mask, '交易对方'].apply(classify_counterparty)

print('=== 分类命中统计 ===')
print(expense['类别'].value_counts())
print()

# 列出所有被归为"其他"的交易对方，方便补关键词
others = expense[expense['类别']=='未分类']['交易对方'].value_counts()
if len(others) > 0:
    print(f'未归类的交易对方（{len(others)}个）')
    print(others.to_string())
    print()

plt.figure(figsize=(15,8))

cat_stats = expense.groupby('类别')['金额(元)'].sum().sort_values(ascending=False)

# 小类别（不足总数3%）合并为一个"其他（杂项）"
total=cat_stats.sum()
main=cat_stats[cat_stats >= total*0.03]
small_sum=cat_stats[cat_stats < total*0.03].sum()
if small_sum>0:
    main['其他（杂项）']=small_sum

# 配色
color_map={
    '餐饮':'#e74c3c',
    '外卖':'#f39c12',
    '购物':'#2ecc71',
    '饮品':'#3498db',
    '打印':'#9b59b6',
    '交通':'#1abc9c',
    '娱乐':'#e67e22',
    '转账/红包':'#e91e63',
    '未分类':'#95a5a6',
    '其他（杂项）':'#bdc3c7',
}
colors=[color_map.get(c,'#95a5a6') for c in main.index]

wedges,texts,autotexts=plt.pie(
    main.values,
    labels=main.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    pctdistance=0.6,
    textprops={'fontsize':13}
)

for at in autotexts:
    at.set_fontsize(11)
    at.set_color('white')

plt.title('消费分类构成',fontsize=20,pad=20)

# 图例：类别 + 金额
legend_labels=[f'{cat}  ￥{amt:.2f}' for cat,amt in zip(main.index,main.values)]
plt.legend(wedges,legend_labels,title='类别 / 金额',
           loc='center left',bbox_to_anchor=(1,0,0.5,1),fontsize=10)

plt.tight_layout()
save_chart('消费分类饼图.png')
plt.show()