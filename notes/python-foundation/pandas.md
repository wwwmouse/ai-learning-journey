# pandas

## pandas简介
pandas是**基于numpy**构建的专门为处理**表格和混杂数据**设计的库
主要充当**数据读取、清洗、分析、统计与输出**的高效工具
其提供了易于使用的数据结构和数据分析工具，特别适合于处理**结构化数据**

核心设计理念包括：
- 标签化结构数据：用标签而非纯数字索引
- 灵活处理缺失数据：内置NaN机制
- 智能数据对齐：自动按标签对齐
- 时间序列处理：支持日期时间处理和频率转换

|工具|功能特色|适用场景|
|---|----|--------|
|Excel|图形界面，易于上手|人工分析、小规模数据|
|SQL|高效读写，最终数据源|数据库查询和联表|
|Python+Pandas|算法和分析部署核心|数据清洗，统计分析，可视化|

**pandas 的底层是numpy，因此继承了numPy 的高性能，同时增加了标签、缺失值、分组聚合等表格操作能力。**

## series

一个**纵向一维且带索引**的numpy数组
创建时默认显示dtype

### series的创建

**pd.Series(data, index, dtype, name, copy)**
| 参数      | 作用                | 是否必须           |
| ------ | ---------------- | ------------- |
| data  | 数据内容（列表、数组、字典、标量） |  必须           |
| index | 行标签               | 可选，默认 0,1,2... |
| dtype | 数据类型              | 可选，**自动推断**    |
| name  | Series 的名字        | 可选，默认 None     |
| copy  | 是否复制数据            | 可选，默认 False    |

#### 基础创建
每次输出Series时最左侧会自带一列索引
```bash
import pandas as pd
s=pd.Series([10,2,3,4,5])
print(s)

# 0      10
# 1      2
# 2      3
# 3      4
# 4      5

#自定义索引、名称
s=pd.Series([10,2,3,4,5],index=['A','B','C','D','E'],name="月份")
print(s)
# A     10
# B     2
# C     3
# D     4
# E     5
# Name: 月份, dtype: int64
```

#### 通过字典创建

```bash
s=pd.Series({'a':1,'b':2,'c':3,'d':4,'e':5})

print(s)
print(s['a'],s['e'])
# a      1
# b      2
# c      3
# d      4
# e      5
# dtype:int64
# 1 5
```
### 数据对齐

**Series 按标签对齐相加**

```bash
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])

print(s1 + s2)
# a     NaN    s2 没有 'a'
# b    12.0    2 + 10，按标签 'b' 对齐
# c    23.0    3 + 20，按标签 'c' 对齐
# d     NaN    s1 没有 'd'
```

### series的属性
**整体和numpy大同小异**
|属性|说明|
|----|----|
|index|索引对象|
|values|值|
|dtype|元素类型|
|shape|形状|
|size|元素个数|

需要注意的是，series定义为一维数组，所以ndim属性恒为1，shape恒为(n,)

### series的常用方法
numpy 的设计是**面向数组**的函数库，Pandas 的设计是**面向对象**的封装
所以很多numpy里的函数到series里变成了类自带的方法

例如：从numpy的`np.median(arr)`变为了series的`arr.median()`

| 方法                   | 作用      | 
| --------------------- | ------ | 
| `s.mean()` / `s.sum()` | 均值 / 求和 | 
| `s.value_counts()`     | 计数      | 
| `s.head(n)`             | 查看前n行数据，默认5|
| `s.tail(n)`             | 查看后n行数据，默认5|
| `s.min()` /  `s.max()`  | 最小/大值|
| `s.std()` /  `s.var()`  | 标准差/方差|
| `s.median()`/ `s.mode()`| 中位数/众数|
| `s.idxmax()`/ `s.idxmin()`| 最大/小值标签索引|
| `s.sort_values()`      | 返回将值从低到高排序的新数组|
| `s.value_counts()`	   | 每个值出现次数|
| `s.describe()`           | 查看所有描述性信息|

### series 与 numpy 的核心区别

| 特性 | Numpy ndarray | Pandas Series |
|---|---|---|
| 索引方式 | 只能用数字位置 | **可以用任意标签**（名字、日期等）|
| 数据对齐 | 按位置对齐 | **按标签自动对齐** |
| 缺失值 | 不支持，需特殊处理 | **原生支持 `NaN`** |
| 筛选结果 | 返回裸数组 | **返回带标签的 Series** |

**Series = NumPy 数组 + 标签索引 + 自动对齐 + 缺失值处理**

