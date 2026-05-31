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
print(arr.adim)

arr2=np.array([[1,2],[3,4]]) #创立二维的ndarry数组
print(arr2)
print(arr.adim)
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
1. 基础构造
```bash
import numpy as np
arr1=np.array(5) #创造一个0维数组，元素只有5

a=[1,2,3]
arr2=np.array(a) #传入列表a，此时arr2是一个一维数组

arr3=np.array(a,dtype=float)
#print(arr3)结果为[1. 2. 3.]，因为我们用dtype强行指定了数据类型为float
```
在初始化ndarray时
- 第一个参数为**object：要转换成数组的数据（列表，元组等）**
- 第二个参数为**dtype：指定数据类型，和之前的属性同名**







