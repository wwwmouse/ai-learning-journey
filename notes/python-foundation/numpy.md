# numpy

numpy是Python中科学计算的基础包
它让Python能够像MATLAB或C语言一样，快速、高效地处理大规模数值数据

## numpy特点
- 向量化：提供多维数组对象、各种派生对象以及用于对数组快速操作的方法
- 高性能：底层用C语言实现，运算速度快于纯Python
- 内存高效：数组在内存中连续存储

## ndarry
numpy中的数据结构，类似C语言的数组

### 特性

- 多维性：支持0维（标量）、1维（向量）、2维（矩阵）以及更高
```bash
import numpy as np
arr=np.array(3) #创立0维的ndarray数组
print(arr)
print(arr.ndim)

arr2=np.array([[1,2],[3,4]]) #创立二维的ndarry数组
print(arr2)
print(arr.ndim)
```
- 同质性：内存元素的类型必须一致
```bash
a=np.array([1,"Have a nice day!^^"]) #int string
print(a)
```
结果会是['1' 'Have a nice day!^^'] ，显然发生了强制类型转换，int->string
这是为了保证ndarray内部元素类型一致

- 高效性：基于内存块连续存储，支持向量化运算

### 属性
|名称|输出|
|----|-----|
| shape |  (行数，列数) |
| ndim  |  数组维度数量 |
| size   | 总元素个数 |
| dtype | 元素类型 |
| T     | 转置后矩阵 |

>使用方式均为对象.名称，就是cpp中class的属性

### 创建

#### 1. 基础构造
```bash
import numpy as np
arr1=np.array(5) #创造一个0维数组，元素只有5

a=[1,2,3]
arr2=np.array(a) #传入列表a，此时arr2是一个一维数组

arr3=np.array(a,dtype=float)
#print(arr3)结果为[1. 2. 3.]，因为我们用dtype强行指定了数据类型为float

arr4=np.copy(arr1)
arr5=arr1.copy()
#两种写法完全等价
```
**初始化ndarray**
- 第一个参数为**object：要转换成数组的数据（列表，元组等）**
- 第二个参数为**dtype：指定数据类型，和之前的属性同名**

**关于np.copy()**
1. 若参数数组类型不为object，则**深拷贝**整个数组，完全独立
2. 若为object，则只**浅拷贝**数组外壳，仅复制整个数组的指针，不复制对象，内部数值共享！
   因此对于object类型的数组，其内部的**可变对象**(list,dict,set)与复制的新数组共享，
```bash
# 数值数组：深拷贝，完全独立
a = np.array([1, 2, 3])
b = np.copy(a)
b[0] = 999
print(a[0])   # 1，不受影响

# object 数组：浅拷贝元素对象
c = np.empty(2, dtype=object)
c[0] = [1, 2]
c[1] = [3, 4]
d = np.copy(c)
d[0].append(99)
print(c[0])   # [1, 2, 99]也被修改！
print(d[0])   # [1, 2, 99]

```
a[0].append(99)对object数组内部的inner指针所指对象进行修改
>因为dtype=object&&list可变，所以两个数组都会变化

#### 2. 预定义形状
大致分为4种，全0、全1、未初始化、固定值
1. 全0
```bash
arr=np.zeros((2,3))#数组形状
print(arr)
print(arr.ndim)# 2
print(arr.shape)# (2,3)
```
2. 全1
```bash
#基本同上，没啥说的
arr1=np.ones((2,3,5))
print(arr1)
print(arr1.ndim)# 3
print(arr1.shape)# (2,3,5)
#注意一下全0/1数组数据类型默认float
```
3. 未初始化，效率更快
```bash
arr2=np.empty((2,3)) #里面的值为内存残留值，在性能敏感场景是推荐用法
```

4. 固定值
```
arr3=np.full((3,5),520)#两个参数，形状和填充值
```

#### 3. 基于数值范围生成

|函数|作用|范围|特点|
|----|----|----|----|
|np.arange(start,stop,step)|按照给定步长step生成**等差数列**|[start,stop)|等差d=step|
|np.linspace(start,stop,num)|按照给定个数num生成**等间隔数列**|[start,stop]|生成的数等分区间]
|np.logspace(start,stop,num,base)|生成从base^start到base^stop的num个**指数等间隔的幂**|[start,stop]|底数默认10|
```bash
#等差数列
arr1=np.arange(1,10,2)
# [1 3 5 7 9]

#等间隔
arr2=np.linspace(1,10,5)
#[ 1. 3.25 5.5  7.75 10.  ]

#对数间隔
arr3=np.logspace(0,10,6)#base未赋值默认10
#[1.e+00 1.e+02 1.e+04 1.e+06 1.e+08 1.e+10]   
```
#### 4. 特殊矩阵生成

|函数|作用|
|----|----|
|np.eye(n)|生成n*n单位矩阵|
|np.diag(arr)|用一维数组创建对角矩阵|

```bash
arr1=np.eye(3)
a=[1,2,3]
arr2=np.diag(a)
print(arr1)
print(arr2)

# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```
#### 5. 随机数组生成

|函数|作用|生成特点|
|----|----|----|
|np.random.rand(d0,d1,...)|生成指定形状0-1随机浮点数|均匀分布|
|np.random.uniform(low,high,size)|生成指定范围区间的随机浮点数|均匀分布|
|np.random.randint(low,high,size)|生成指定区间的随机整数|均匀分布|
|np.random.randn(d0,d1,...)|生成指定形状0-1正态分布浮点数|正态分布|

### 数据类型
没什么好讲的，就是最基本的几个数据类型 int,float,bool等，**注意别超范围**
>主要用于array第二个参数dtype赋值

### 索引与切片
|类型|用法|示例|
|---|-----|-----|
|基本索引|通过整数索引直接访问元素|arr[0],arr[2,3]|
|切片索引|选择连续区域|arr[1:5],arr[:,2:4]|
|步长切片|通过步长间隔选取|arr[start:stop:step],arr[1:8:2]|
|布尔索引|通过布尔条件筛选切片规则|arr[arr>0]|
|花式索引|整数数组选元素|arr[ [0,2,4] ]|

#### 一维数组
```bash
arr=np.random.randint(1,100,15)
print(arr)
print(arr[0])
print(arr[2:10]) #半闭半开区间
print(arr[1:10:2]) #带上步长，还是半闭半开区间^^
print(arr[:])  #获取全部数据
print(arr[arr>50]) #对每个值进行筛选
print(arr[[1,3,5]]) #花式索引，筛选指定元素为数组

# [81 31 44 33 17 25 88 13 39 11 4 19 50 25 10 ]       
# 81                                                                                   
# [44 33 17 25 88 13 39 11]
# [31 33 25 13 11]                                                         
# [81 31 44 33 17 25 88 13 39 11 4 19 50 25 10]                          
# [81 88]   
# [31 33 25]
```

#### 二维数组
和一维数组大同小异，注意选清楚行列即可
```bash
arr=np.random.randint(1,100,(5,5))
print(arr)
print(arr[1,1])
print(arr[2,2:5]) #第3行第3到第5个元素
print(arr[[0,2]]) #花式索引，筛选第1、3行
print(arr[:,[0,2]]) #筛选第1、3列
print(arr[3,:]) #与arr[3]等价
# [[86 72 23 70 37]
#  [ 7 46 28 95 3 ]
#  [94 14 89 48 56]
#  [81 45 19 27 41]                                                                 
#  [15 98 89 61 24]] 

#  46
#  [89 48 56]

#  [[86 72 23 70 37]
#   [94 14 89 48 56]]

# [[86 23]
#  [7  28]
#  [94 89]
#  [81 19]
#  [15 89]]

#  [81 45 19 27 41]

```
#### 按行/列筛选


1. 根据某行筛选列
```bash
mask = arr[0, :] > 50   # 第1行中，值大于50的那些列
print(arr[:, mask])
```
2. 多条件组合筛选行
```
mask = (arr[:, 0] > 30) & (arr[:, 1] > 40) #筛选出 第一列>30且第二列>40 的行，mask在这里是一个bool类型的ndarray
print(arr[mask])
```
3. 同时筛选行列
```
row_mask = arr[:, 0] > 30 #第一列>30的行
col_mask = arr[0, :] > 50 #第一行>50的列
print(arr[row_mask][:, col_mask]) #统计输出，先筛选出第一列>30的所有行，再在这些行中筛选出第一行>50的那些列
```
4. 保留子矩阵形状
```
print(arr[np.ix_([0, 2], [1, 3])]) #花式索引的笛卡尔积，取第0、2行和第1、3列的交叉组合，结果是一个 2×2 矩阵
```

### ndarray的运算





