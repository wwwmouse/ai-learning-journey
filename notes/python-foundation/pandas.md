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

## 1.series

一个**纵向一维且带索引**的numpy数组
创建时默认显示dtype

### series的属性
**整体和numpy大同小异**
|属性|说明|
|----|----|
|shape|形状|
|size|元素个数|
|ndim|维度数|
|index|每个元素的索引|
|values|值|
|dtypes|元素类型|

需要注意的是，series定义为一维数组，所以ndim属性恒为1，shape恒为(n,)

### 1.1 series的常用方法
numpy 的设计是**面向数组**的函数库，Pandas 的设计是**面向对象**的封装
所以很多numpy里的函数到series里变成了类自带的方法

series对于s[]的行为是**优先当标签索引，找不到才fallback当位置索引**
为了避免可能存在的歧义，**通常使用loc[]、iloc[]**

例如：从numpy的`np.median(arr)`变为了series的`s.median()`

| 方法                   | 作用      | 
| --------------------- | ------ | 
| `s.mean()` / `s.sum()` | 均值 / 求和 | 
| `s.loc[]`             | 按**标签**索引 | 
| `s.iloc[]`            | 按**位置**索引 |
| `s.value_counts()`     | 计数      | 
| `s.head(n)`             | 查看前n行数据，默认5|
| `s.tail(n)`             | 查看后n行数据，默认5|
| `s.min()` /  `s.max()`  | 最小/大值|
| `s.std()` /  `s.var()`  | 标准差/方差|
| `s.median()`/ `s.mode()`| 中位数/众数|
| `s.idxmax()`/ `s.idxmin()`| 最大/小值标签索引(独有)|
| `s.argmax()`/ `s.argmin()`| 最大/小值位置索引(兼容ndarray)|
| `s.to_numpy().argmax()`/ `s.to_numpy().argmin()`| 最大/小值位置索引|
| `s.sort_values()`      | 返回将指定从低到高排序的新数组|
| `s.describe()`           | 查看所有描述性信息|

### 1.2 series的创建

**pd.Series(data, index, dtype, name, copy)**
| 参数      | 作用                | 是否必须           |
| ------ | ---------------- | ------------- |
| data  | 数据内容（列表、数组、字典、标量） |  必须           |
| index | 行标签               | 可选，默认 0,1,2... |
| dtype | 数据类型              | 可选，**自动推断**    |
| name  | Series 的名字        | 可选，默认 None     |
| copy  | 是否复制数据            | 可选，默认 False    |

#### 1.2.1 基础创建
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

#### 1.2.2 通过字典创建

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
### 1.3 Series的运算
**Series中的运算和ndarray大致一样,无需刻意记忆**
**但请注意Series 按index对齐相加**

```bash
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])

print(s1 + s2)
# a     NaN    s2 没有 'a'
# b    12.0    2 + 10，按标签 'b' 对齐
# c    23.0    3 + 20，按标签 'c' 对齐
# d     NaN    s1 没有 'd'
```

### 1.4 series 与 numpy 的核心区别

| 特性 | Numpy ndarray | Pandas Series |
|---|---|---|
| 索引方式 | 只能用数字位置 | **可以用任意标签**（名字、日期等）|
| 数据对齐 | 按位置对齐 | **按标签自动对齐** |
| 缺失值 | 不支持，需特殊处理 | **原生支持 `NaN`** |
| 筛选结果 | 返回裸数组 | **返回带标签的 Series** |

**Series = NumPy 数组 + 标签索引 + 自动对齐 + 缺失值处理**

## 2. DataFrame

每行/列都有各自名称的二维数组
每一列都是一个series
**DataFrame = 多个 Series 共享同一个行索引**

### 2.1 DataFrame的创建

| 参数        | 作用                      | 是否必须           |
| :-------- | :---------------------- | :------------- |
| `data`    | 数据内容（字典、列表、数组、Series 等） |  必须           |
| `index`   | 行标签（行索引）                | 可选，默认 0,1,2... |
| `columns` | 列标签（列名）                 | 可选，默认按 data 推断 |
| `dtype`   | 强制指定数据类型                | 可选，默认自动推断      |
| `copy`    | 是否复制数据                  | 可选，默认 False    |

用字典+Series创建时，根据 Series 的索引名称自动对齐，**key 就是列名**
注意：**columns 参数优先级高于字典 key**，用于筛选/重排序列，指定不存在的列会补 NaN

```python
s1 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
s2 = pd.Series([6, 7, 8, 9, 10], index=['a', 'b', 'c', 'd', 'e'])
df = pd.DataFrame({"第一列": s1, "第二列": s2})
#两个series的索引对齐

df = pd.DataFrame({
    "第一列": [1, 2, 3, 4, 5],
    "第二列": [6, 7, 8, 9, 10]
}, index=['a', 'b', 'c', 'd', 'e'])
#传入的是list，按照顺序自动对齐
#     第一列  第二列
# a     1     6
# b     2     7
# c     3     8
# d     4     9
# e     5     10
```

**若存在多个Series的索引不一致，会在冲突的位置表示为 NaN**

```python
s1 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
s2 = pd.Series([6, 7, 8, 9, 10], index=['b', 'c', 'd', 'e', 'f'])

df = pd.DataFrame({"第一列": s1, "第二列": s2})
print(df)
#    第一列  第二列
# a    1.0   NaN   s2 没有'a'
# b    2.0   6.0   
# c    3.0   7.0
# d    4.0   8.0
# e    5.0   9.0
# f    NaN  10.0   s1 没有'f'
```

### 2.2 DataFrame的属性

|属性|说明|
|----|----|
|shape|形状|
|size|元素个数|
|ndim|维度数|
|index|行标签|
|columns|列标签|
|values|值|
|dtypes|元素类型|

### 2.3 DataFrame的常见方法
| 方法           | 作用           | 
| :----------- | :----------- | 
| `df.mean()` / `df.sum()` | 按列求均值/求和 | 
| `df.loc[]`                | 按标签取           |
| `df.iloc[]`               | 按位置取           | 
| `df.head(n)`              | 查看前 n 行，默认 5 | 
| `df.tail(n)`              | 查看后 n 行，默认 5 |
| `df.count()`               | 返回每列非NaN的个数 |
| `df.describe()`          | 数值列的统计摘要 |


### 2.4 DataFrame的数据访问
```bash
df=pd.DataFrame({
    "id":[1,2,3,4,5],
    "name":["Tom","Jack","Alice","Bob","Allen"],
    "age":[15,17,20,26,30],
    "score":[60.5,80,30.6,70,83.5]
},index=['a','b','c','d','e'])
```
#### 2.4.1 取单行
没有简洁写法，只能df.loc[]/df.iloc[]

```bash
print(df.loc['a']) #取'a'行的内容
# id       1
# name     Tom
# age      15
# score    60.5
# Name: a, dtype: object
```
#### 2.4.2 取单列

三种写法等价
**df.name==df["name"]==df.loc[:,"name"]**

```bash
print(df.loc[:,"name"]) #取name列
# 等价于print(df.name)
# 等价于print(df["name"])

# a       Tom
# b       Jack
# c       Alice
# d       Bob
# e       Allen
# Name: name, dtype: str
```
#### 2.4.3 精确位置
```bash
print(df.loc['c',"name"]) #取'c'行"name"列
# Alice
```
因为DataFrame 的每一列一定是 Series
所以在学习了取单列后，我们可以通过取出特定的一列series，再使用series方法实现目的
```bash
df.score.sum() 
#df.score先取出"score"列，这一列是个series
#再用Series.sum()方法来实现

df["score"].sum()
#两种写法等价
```
**注意：df[]这种结构用法不统一，可以取行可以取列**
造成这种混乱的原因是pandas又想混合**字典["key"]取值**的语法，还想**用numpy切片**的语法
因此直接用df[n]的话，n不会被视作**行索引**，而会被视作**列名**
| `[]` 内部          | 理解           | 例子                      |
| :--------------- | :------------ | :---------------------- |
| **单个标签**（字符串/整数） | **列名** → 取列   | `df['A']`、`df[0]`（找列名0） |
| **标签列表**         | **多列名** → 取多列 | `df[['A', 'B']]`        |
| **切片/范围**        | **行位置** → 取行  | `df[0:2]`               |
| **布尔 Series/列表** | **行筛选** → 取行  | `df[df['A'] > 5]`       |

非常容易混淆
建议**df[]的写法统一用来选取列**
**df.iloc[]再用来找行**

| 写法             | 含义             | 
| :------------- | :------------- | 
| `data[0]`      | **列名为 0 的列**  | 
| `data.iloc[0]` | **第0行**       |               

## 3. 数据分析

1. 数据收集：把文件数据导入dataframe等数据结构中
2. 数据清洗：处理缺失值、错误数据、格式混乱
3. 数据分析：统计(mean,max)，分组对比
4. 数据可视化：与matploytlib有关，折线图，柱状图，散点图

### 3.1 数据收集与导出
| 操作       | 方法                             | 
| :------- | :----------------------------- | 
| 读 CSV    | `pd.read_csv('路径')`            | 
| 读 Excel  | `pd.read_excel('路径')`          | 
| 读 JSON   | `pd.read_json('路径')`           | 
| 保存 CSV   | `df.to_csv('路径', index=False)` | 
| 保存 Excel | `df.to_excel('路径')`            |  

参数说明：
- encoding='utf-8' 处理中文乱码
- header=0 指定表头行
- index_col=0 指定某列为索引

### 3.2 数据清洗

| 问题    | 方法             | 示例                  |
| :---- | :-------------- | :------------------ |
| 缺失值   | `df.isnull()`/ `df.dropna()`/ `df.fillna(值)`| `df.fillna(df.mean())` |
| 重复值   | `df.duplicated()` / `df.drop_duplicates()`|     |
| 异常值   | 条件筛选 + 统计判断         | `df[df['年龄'] < 150]`     |
| 类型转换  | `df['列'].astype('类型')`   | `astype('int')`  |
| 字符串处理 | `df['列'].str.xxx()`    | `.str.contains()`, `.str.replace()` |
| 日期解析  | `pd.to_datetime(df['日期列'])` |                |

#### 3.2.1 缺失值

##### 3.2.1.1 查找缺失值

```python
df=pd.DataFrame([[1,None,2],[2,5,7],[pd.NA,0,3]])
print(df)
print(df.isna())# 输出一个bool类型的表格，判断df每个位置是否为缺失值

print(df.isna().sum())# 按列统计缺失值个数

print(df.isna().mean() * 100)  # 缺失百分比（更直观）
```

##### 3.2.1.2 剔除缺失值
**对于质量很差，缺失值很多的数据，我们会选择剔除缺失值**
主要用到`dropna()`函数,它的参数都很实用

| 参数        | 取值          | 说明                       |
| :-------- | :---------- | :----------------------- |
| `axis`    | `0`（默认）     | 删除**行**（某行有缺失就删）         |
|           | `1`         | 删除**列**（某列有缺失就删）         |
| `how`     | `'any'`（默认） | 只要有缺失就删                  |
|           | `'all'`     | 全部缺失才删                   |
| `thresh`  | 整数          | 保留至少有 `thresh` 个非缺失值的行/列 |
| `subset`  | 列名列表        | 只检查这些列的缺失情况              |
| `inplace` | `False`（默认） | 返回新 DataFrame，原数据不变      |
|           | `True`      | 原地修改，不返回                 |

##### 3.2.1.3 填充缺失值
对于大部分数据，我们使用填充缺失值来处理
主要用到`fillna()`函数
| 参数        | 取值            | 说明            |
| :-------- | :------------ | :------------ |
| `value`   | 标量（0, '未知'）   | 全部填这个值        |
|           | 字典 `{'列': 值}` | 按**列名**指定填充值       |
|           | Series        | 用Series的值对应填充 |
| `method`  | `'ffill'`     | 用前一个非缺失值填充    |
|           | `'bfill'`     | 用后一个非缺失值填充    |
| `axis`    | `0`（默认）       | 按列方向填充        |
|           | `1`           | 按行方向填充        |
| `inplace` | `False`（默认）   | 返回新DataFrame  |
|           | `True`        | 原地修改          |
| `limit`   | 整数            | 最多连续填充几个      |

#### 3.2.2 重复值

对于df
```python
data={
    "name":['Alice','Alice','Bob','Alice','Jack','Bob'],
    "age":[26,25,30,25,35,30],
    "city":['NY','NY','LA','NY','SF','LA']  
}
df=pd.DataFrame(data)
#     name  age city 
# 0  Alice   26   NY 
# 1  Alice   25   NY
# 2    Bob   30   LA
# 3  Alice   25   NY
# 4   Jack   35   SF
# 5    Bob   30   LA 
```
容易发现Alice,Bob各有一对完全重复

##### 3.2.2.1 检测重复值

`df.duplicated()`
默认标记**第二次及以后**出现的**整行重复**行为True，首次出现为False

##### 3.2.2.2 去除重复值
`df.drop_duplicates()`去掉duplicated()标记为True的整行

#### 3.2.3 数据类型转换
读入数据后我们往往需要统一某行/列的数据类型方可进行处理
比如统计max(),mean()，但有的数字被读入成str就会报错
为了方便进行数据分析，我们需要将指定内容进行类型转换

##### 3.2.3.1 `astype()`
可强制转换已存在的数据为指定类型
请注意,转换为int64类型的数据若是遇到NaN会报错，为了保留空整数，需要写成Int64

##### 3.2.3.2 `pd.to_numeric()`
有些时候，数字会和杂质混在一起，比如"50元"," ","50%"等
此时使用astype会报错，为了处理这种更复杂的情况，我们需要使用`pd.to_numeric()`
| 参数         | 取值             | 作用   |
| ---------- | --------------------- | -------- |
| `errors`   | `'raise'`（默认）      | 遇到错误直接报错 |
|            | `'coerce'`    | **无法转换的设为 `NaN`**      |
|            | `'ignore'`    | 无法转换的保持原样，返回原 Series |
| `downcast` | `'integer'` / `'float'` | 自动向下压缩类型，节省内存    |

```python
s = pd.Series(['1', '2', '3', ' '])

# astype 会失败，无法处理空格
# s.astype('int')  # ValueError

# to_numeric 可以处理
pd.to_numeric(s, errors='coerce')  
# 0    1.0
# 1    2.0
# 2    3.0
# 3    NaN   无法转换的变成 NaN
# dtype: float64
```
##### 3.2.3.3 category类型
对于某些列的数据，他们**存在大量重复值，且涵盖的范围有限**
比如df['性别']列下只有'男','女'两种
此时我们可以将其**用astype转为category类型**来实现节省内存
```python
df = pd.DataFrame({
    '性别': ['男', '女', '女', '男', '女'] * 10000  # 5万行，只有2个值
})

# 默认 object 类型
print(df['性别'].dtype)  # object
 # 约 400KB

# 转为 category
df['性别'] = df['性别'].astype('category')
print(df['性别'].dtype)  # category
 # 约 50KB，省 8 倍内存
```
##### 3.2.3.4 pd_to_datetime() 日期转换
日期格式众多，写法各异，单纯用字符串储存难以统一格式与共同处理
pd_to_datetime()可将各种格式的日期字符串统一转为datetime类型

| 参数          | 作用             | 示例                     |
| ----------- | -------------- | ---------------------- |
| `format`    | 指定格式，加速解析      | `format='%Y-%m-%d'`/`format='mixed'`    |
| `errors`    | 同 `to_numeric` | `'coerce'` 把脏日期变 `NaT` |
| `dayfirst`  | 日/月顺序模糊时优先日    | `dayfirst=True`（欧洲格式）  |
| `yearfirst` | 优先年            | `yearfirst=True`       |

```python
dates = pd.Series(['2024-01-01', '2024/02/01', '01-03-2024', 'Jan 4, 2024'])
#四种日期字符串写法

pd.to_datetime(dates)
# 0   2024-01-01
# 1   2024-02-01
# 2   2024-01-03   ← 自动推断月日顺序
# 3   2024-01-04
# dtype: datetime64[ns]
```

#### 3.2.4数据变形
数据变形（Data Reshaping）是 pandas 中从**"宽表"和"长表"之间转换**的操作
核心主要三个函数：melt、pivot、pivot_table

宽表：一列一个变量，适合阅读、作为最终展示格式
| 姓名 | 语文 | 数学 | 英语 |
| :- | :- | :- | :- |
| 张三 | 90 | 85 | 88 |
| 李四 | 78 | 92 | 80 |


长表：一列一个维度，维度内部还会分不同元素，适合分组聚合、可视化、数据库存储
| 姓名 | 科目 | 分数 |
| :- | :- | :- |
| 张三 | 语文 | 90 |
| 张三 | 数学 | 85 |
| 张三 | 英语 | 88 |
| 李四 | 语文 | 78 |
| 李四 | 数学 | 92 |
| 李四 | 英语 | 80 |

##### 3.2.4.1 宽表变长表:melt
把**列名**压缩成**新维度下的值**

```python
import pandas as pd

df_wide = pd.DataFrame({
    '姓名': ['张三', '李四'],
    '语文': [90, 78],
    '数学': [85, 92],
    '英语': [88, 80]
})

# 核心参数
df_long = df_wide.melt(
    id_vars='姓名',           # 保持不变的列（标识列）
    value_vars=['语文', '数学', '英语'],  # 要融化的列（不写默认融化所有非id列）
    var_name='科目',          # 融化后，原列名变成什么列
    value_name='分数'         # 融化后，原值变成什么列
)
print(df_long)
#    姓名  科目  分数                                                                
# 0  张三  语文  90                                                   
# 1  李四  语文  78                                                   
# 2  张三  数学  85                                                     
# 3  李四  数学  92
# 4  张三  英语  88                                                           
# 5  李四  英语  80   
```

##### 3.2.4.2 长表变宽表:pivot
**重塑数据布局**
```python
# 用上面的长表转回宽表
df_wide_back = df_long.pivot(
    index='姓名',      # 新行索引
    columns='科目',    # 新列名
    values='分数'      # 新表的值来自'分数'列
)

print(df_wide_back)
# 科目  数学  英语  语文
# 姓名                
# 张三    85   88   90
# 李四    92   80   78
```

我们会注意到，行索引和列名前分别有一个'姓名'和'科目'
这其实是`df.index.name/.columns.name`,只是在平时很少使用这个属性

**我们令"姓名"下的值成了新的index，那么"姓名" 本身作为列名就会变成`index.name`**,也就是行索引名
**同理"科目"下的值成了新的columns，"科目"就会变成`columns.name`**,也就是列索引名

为了去掉这两个不该有的数据，在使用pivot后可以直接接以下步骤
```python
df_wide_back.columns.name = None   # 把 columns.name 设回 None，取消列索引名
df_wide_back = df_wide_back.reset_index() #把index变成普通列，此时行索引名变成普通column
print(df_wide_back)
#     姓名    数学    英语    语文
# 0   张三    85      88      90
# 1   李四    92      80      78
```
