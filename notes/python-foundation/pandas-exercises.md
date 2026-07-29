## series
### 1. 学生成绩统计
创建一个包含十名学生成绩的Series，成绩范围在50-100之间
计算平均分、最高分、最低分，并找出高于平均分的学生人数

```python
import pandas as pd
import numpy as np

data=np.random.randint(50,101,10)
index=["学生"+str(i) for i in range(1,11)]
scores=pd.Series(data,index=index)
print(f"平均分：{scores.mean()}")
print(f"最高分：{scores.max()}")
print(f"最低分：{scores.min()}")

avg=scores.mean()
print(f"高于平均分的学生：")
print(scores[scores>avg])
print(f"共计{len(scores[scores>avg])}人")

```

### 2. 温度数据分析
给定某城市一周每天的最高温度Series
温度>30度的天数
计算平均温度
将温度从高到低排序
```python
tem=pd.Series([28,31,29,32,30,27,33],index={"周一","周二","周三","周四","周五","周六","周日"})
print(f"温度>30的天数为：{len(tem[tem>30])}")
print(f"平均温度为：{round(tem.mean(),2)}度")
print(tem.sort_values(ascending=False))
```

### 3. 股票价格分析
给定某股票连续十个交易日的收盘价s
计算每日收益率(当日收盘价/前日收盘价-1)
找出收益率最高和最低的日期
计算收益率的标准差

```python
date=pd.date_range("2026-01-01",periods=10)
prices=np.array([102.3,103.5,105.1,104.8,106.2,107.0,106.5,108.1,109.3,110.2])
s=pd.Series(prices,index=date)

per=s.pct_change() #计算收益率的函数，首位为NaN
print(per)
print(f"收益率最高的一天是：{per.idxmax()}")
print(f"收益率最低的一天是：{per.idxmin()}")
print(f"收益率的标准差：{round(per.std(),3)}")
```


## dataframe
### 1. 学生成绩分析
场景：某班级学生成绩数据如下，请完成以下任务
1. 表格中加入每位学生总分与平均分
2. 找出数学成绩高于90分或英语成绩高于85分的学生
3. 按总分从高到低排序，并输出前三名学生
```python
scores=pd.DataFrame(
    {
        "姓名":['张三','李四','王五','赵六','钱七'],
        "数学":[85,92,78,88,95],
        "英语":[90,88,85,92,80],
        "物理":[75,80,88,85,90]
    }
)
scores['平均分']=scores.iloc[0:5,1:4].mean(axis=1)
scores['总分']=scores[['数学','英语','物理']].sum(axis=1)
## ==scores['总分']=scores.iloc[0:5,1:4].sum(axis=1)

target=(scores['数学'] >90 )| (scores['英语']>85)
print(scores[target])
## ==print(scores[(scores['数学']>90)|(scores['英语']>85)])

scores['总分']=scores[['数学','英语','物理']].sum(axis=1)
new_s=scores.sort_values('总分',ascending=0)
print(new_s.head(3))
```

### 2.销售数据分析
场景：某公司销售数据如下
1. 计算每种产品的总销售额
2. 找出销售额最高的产品
3. 按销售额从高到低排序，并输出所有产品信息

```python
data=pd.DataFrame(
    {
        'name'={'A','B','C','D'},
        'prices'=[100,150,200,120],
        'sales'=[50,30,20,40]
    }
)
data['sumprices']=data['prices']*data['sales'] #使用series乘法，对两列每个元素分别相乘

print(data.iloc[data.sumprices.argmax()])#argmax()取最大值的行索引，iloc[]用索引取行

new_data=data.sort_values('sumprices',ascending=0)#对sumprices排序，ascending=0为升序
print(new_data)
```
