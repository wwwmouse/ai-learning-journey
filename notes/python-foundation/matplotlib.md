# matplotlib

**Matplotlib 是一个底层绘图库，提供对图表元素（坐标轴、线条、标记、文本等）的精细控制**
它不像高层工具（如 Seaborn）那样便于直接使用，而是像"数字画布"，令使用者可以精确控制每一个像素

与其他工具对比
|工具|说明|优点|缺点|
|----|----|---|-----|
|`matplotlib`|Python最基础的可视化库|**灵活强大、定制性强**|代码多、风格基础|
|`seaborn`|基于matplotlib的高级接口|**风格美观、统计图方便**|对于简单图略繁琐|
|`pandas plot`|快速图表、调用.plot()|**快捷、适配EDA**|图表样式少|

数据可视化可以将抽象的数据变得直观
让数据背后的规律、异常和趋势一目了然

## 常见的图表及其选择

- 分布图：直方图、箱线图，快速寻找异常与偏态数据
- 关系图：散点图、折线图，便于发现变量间的相关性与趋势
- 组成图：饼图，容易理解数据组成
- 对比图：柱状图、面积图：比较多组数据

| 图表类型      | 适用场景       | Matplotlib 函数                                  | 关键参数/备注                     |
| --------- | ---------- | ---------------------------------------------- | --------------------------- |
| **直方图**   | 数据分布、频率    | `plt.hist(data, bins=10)`                      | `bins` 分组数，`density` 归一化    |
| **箱线图**   | 分布、异常值检测   | `plt.boxplot(data)`                            | `vert=False` 水平显示           |
| **散点图**   | 相关性、分布关系   | `plt.scatter(x, y)`                            | `s` 点大小，`c` 颜色映射            |
| **折线图**   | 趋势变化、时间序列  | `plt.plot(x, y)`                               | `marker` 标记点，`linestyle` 线型 |
| **饼图**    | 占比、构成比例    | `plt.pie(sizes, labels=)`                      | `autopct='%1.1f%%'` 显示百分比   |
| **柱状图**   | 分类比较、数量对比  | `plt.bar(x, height)`                           | `width` 柱宽，`color` 颜色       |
| **面积图**   | 累积趋势、部分与整体 | `plt.fill_between(x, y)`                       | `alpha` 透明度                 |



## 折线图：plot
基础操作
```python
import matplotlib.pyplot as plt
from matplotllib import rcParams
rcParams['font.family']= 'SimHei

plt.figure(figsize=(10,5))
#创建画布，设置大小
```
若存在中文显示不出来的情况
文件开头加上
`from matplotlib import rcParams`
`rcParams['font.family']= 'SimHei'`

**windows用SimHei,Mac用STHeiti**

**对于图的具体配置**
```python
month=['Jan','Feb','Mar','Apr']
sales=[100,150,80,130]


plt.plot(
    month,sales,
    label='产品A',
    color='orange',
    linestyle="--",
    marker='o'
    )
#绘制图表

plt.title('2026销售趋势')
#添加标题

plt.xlabel('月份',fontsize=10)
plt.ylabel('销售额（万元）',fontsize=10)
#添加坐标轴标签

plt.xticks(rotation=10,fontsize=12)
plt.xticks(rotation=10,fontsize=12)
#设置坐标轴刻度角度、大小

plt.ylim(50,160)
#设置坐标轴范围

plt.legend(loc='upper left')
#控制添加图例位置（左上角）

plt.grid(True,alpha=0.2,color='black')
#添加网格线，方向、透明度、颜色

for x,y in zip(month,sales):
    plt.text(x,y,str(y),ha='center',va='bottom',fontsize=10)
#在每个数据点旁添加数值标签
#x,y确定数值标签在坐标系中的位置
#str(y)是要输出的内容，str()是因为text第三个参数只接受字符串
#ha:水平对齐方式
#va:垂直对齐方式

plt.show()
#显示图表
```





