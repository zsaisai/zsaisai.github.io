---
title: 常见概率分布总结
date: 2020-07-19 11:43:45
tags: Machine Learning
categories: 机器学习
top:
mathjax: true
---
# 相关概念

 概率论和统计中的一些相关概念：

- 方差

   如果是在统计领域，方差就是各个样本值和全体样本值的平均数之差的平方值的平均数。如果是在概率论里，方差就是度量随机变量与其数学期望(均值)之间偏离度。 

- 标准差

  标准差是方差的平方根 ，方差虽然能很好的描述数据与均值的偏离程度，但是处理结果是不符合我们的直观思维的。举个例子：一个班级里有60个学生，平均成绩是70分，标准差是9，方差是81，假设成绩服从正态分布，那么我们通过方差不能直观的确定班级学生与均值到底具体偏离了多少分，通过标准差我们就可以直接得到学生成绩分布在[61,79]范围的概率为68%

- 数据类型

  数据类型（统计学里也叫随机变量）有两种。也对应不同的概率分布。

  **离散数据**根据名称很好理解，就是数据的取值是不连续的，有明确的间隔 。例如掷硬币就是一个典型的离散数据，因为抛硬币要么是正面，要么是反面，还有比如男女性别，二分类中的0/1关系等。

  **连续数据**。连续数据正好相反，它能取任意的数值。例如时间，它能无限分割。数据十分平滑 ，还有比如年领，温度，等。是常见的变量类型。
  

<!-- more -->

# 概率分布

数据类型有两种。也对应不同的概率分布。

常见离散概率分布有：**伯努利分布，二项分布，泊松分布，几何分布**。

常见连续概率分布有：**正态分布，拉普拉斯分布，指数分布**。

## 1. 伯努利分布

伯努利分布(两点分布/0-1分布)：伯努利试验指的是**只有两种可能结果的单次随机试验**。若随机变量X的取值为0和1两种情况，且满足概率分布 ![P(X=1)=p, P(X=0)=1-p](https://www.zhihu.com/equation?tex=P%28X%3D1%29%3Dp%2C+P%28X%3D0%29%3D1-p) ，则X服从参数为 ![p](https://www.zhihu.com/equation?tex=p) 的伯努利分布。

举例：假设有产品100件，其中正品90件，次品10件。现在随机从这100件中挑选1件，那么他挑选出正品的概率为0.9，即 ![P(X=正品)=p = 0.9](https://www.zhihu.com/equation?tex=P%28X%3D%E6%AD%A3%E5%93%81%29%3Dp+%3D+0.9) 。

**定义：**

如果[随机变量](https://baike.baidu.com/item/随机变量)X只取0和1两个值，并且相应的概率为：

![img](https://bkimg.cdn.bcebos.com/formula/94dbb6dd3fcf103a46c81205a8e46d36.svg)

则称随机变量X服从参数为p的伯努利分布，若令q=1一p，则X的概率函数可写为：

![img](https://bkimg.cdn.bcebos.com/formula/6964206b823dda83d13fd409cb9075e5.svg)

要证明该概率函数![img](https://bkimg.cdn.bcebos.com/formula/b52ed39c539199d5f4d4c046e1ffaef8.svg) 确实是公式所定义的伯努利分布，只要注意到 ![img](https://bkimg.cdn.bcebos.com/formula/09841e3efde3f52360fcf239d100e590.svg)

 ，就很容易得证。

## 2. 二项分布

现在**独立重复**的挑了n个产品(有放回的)，则他挑出的n个产品中，有k件是正品的概率。简单来说就是，n是重复的伯努利实验的次数，是一个随机变量。所以二项分布也叫n重伯努利分布。

**定义：**

若随机变量X的取值为 ![[公式]](https://www.zhihu.com/equation?tex=0%EF%BC%8C1%EF%BC%8C...%EF%BC%8Cn) ，且满足概率分布 ![[公式]](https://www.zhihu.com/equation?tex=P%28X%3Dk%29%3D%5Cbinom%7Bn%7D%7Bk%7Dp%5E%7Bk%7D%281-p%29%5E%7Bn-k%7D) ,则称X服从参数为 ![[公式]](https://www.zhihu.com/equation?tex=n%2Cp) 的**二项分布**， ![[公式]](https://www.zhihu.com/equation?tex=X+%5Csim+B%28n%2Cp%29)

## 3. 泊松分布

 泊松分布在概率统计当中非常重要，可以很方便地用来计算一些比较难以计算的概率。很多书上会说，泊松分布的**本质还是二项分布**，泊松分布只是用来简化二项分布计算的。**假设现在在一天时间中不停歇的挑选产品**，则单位时间（极小）内挑出正品零件的概率为P，一天共挑出正品k个。 （案例来自概率论书上的例题）

我们把这个p的式子带入原式，可以得到：

![[公式]](https://www.zhihu.com/equation?tex=P%28k%29+%3D+C_n%5Ek+%5Ccdot+%7B%5Cfrac%7B%5Clambda%7D%7Bn%7D%7D%5E%7Bk%7D%281-%5Cfrac%7B%5Clambda%7D%7Bn%7D%29%5E%7Bn-k%7D+%5C%5C)



为了满足二项分布，在单位时间内只发生一次挑选事件（正品或次品），我们需要让单位时间尽量小。所以这个**n应该越大越好**，根据极限，让n趋向于无穷，所以这个问题就变成了一个求极限的问题。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+P%28k%29+%3D+%5Clim_%7Bn+%5Cto+%5Cinfty%7D++C_n%5Ek+%5Ccdot%7B%5Cfrac%7B%5Clambda%7D%7Bn%7D%7D%5E%7Bk%7D%281-%5Cfrac%7B%5Clambda%7D%7Bn%7D%29%5E%7Bn-k%7D+%5Cend%7Baligned%7D+%5C%5C)



我们来算一下这个极限：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+P%28k%29+%26%3D+%5Clim_%7Bn+%5Cto+%5Cinfty%7D++C_n%5Ek+%5Ccdot%7B%5Cfrac%7B%5Clambda%7D%7Bn%7D%7D%5E%7Bk%7D%281-%5Cfrac%7B%5Clambda%7D%7Bn%7D%29%5E%7Bn-k%7D+%5C%5C+%26%3D++%5Clim_%7Bn+%5Cto+%5Cinfty%7D+%5Cfrac%7Bn%28n-1%29%28n-2%29%5Ccdots%28n-k%2B1%29%7D%7Bk%21%7D%7B%5Cfrac%7B%5Clambda%7D%7Bn%7D%7D%5Ek%281-%5Cfrac%7B%5Clambda%7D%7Bn%7D%29%5E%7Bn-k%7D+%5C%5C+%26%3D+%5Clim_%7Bn+%5Cto+%5Cinfty%7D+%5Cfrac%7B%5Clambda%5Ek%7D%7Bk%21%7D+%281-%5Cfrac%7B%5Clambda%7D%7Bn%7D%29%5En%5Ccdot+%5Cfrac%7Bn%7D%7Bn%7D+%5Ccdot+%5Cfrac%7Bn-1%7D%7Bn%7D%5Ccdots+%5Cfrac%7Bn-k%2B1%7D%7Bn%7D+%281-%5Cfrac%7B%5Clambda%7D%7Bn%7D%29%5E%7B-k%7D+%5Cend%7Baligned%7D+%5C%5C)



我们把这个极限拆分开来看，其中：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Clim_%7Bn+%5Cto+%5Cinfty%7D%5Cfrac%7Bn%7D%7Bn%7D+%5Ccdot+%5Cfrac%7Bn-1%7D%7Bn%7D%5Ccdots+%5Cfrac%7Bn-k%2B1%7D%7Bn%7D+%281-%5Cfrac%7B%5Clambda%7D%7Bn%7D%29%5E%7B-k%7D+%3D+1+%5Cend%7Baligned%7D+%5C%5C)![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Clim_%7Bn+%5Cto+%5Cinfty%7D%281-%5Cfrac%7B%5Clambda%7D%7Bn%7D%29%5En+%26%3D+%5Clim_%7Bn%5Cto+%5Cinfty%7D%5C%7B%281%2B%5Cfrac%7B1%7D%7B-%5Cfrac%7Bn%7D%7B%5Clambda%7D%7D%29%5E%7B-%5Cfrac%7Bn%7D%7B%5Clambda%7D%7D%5C%7D%5E%7B-%5Clambda%7D%5C%5C+%26%3D+e%5E%7B-%5Clambda%7D+%5Cend%7Baligned%7D+%5C%5C)



所以，我们代入，可以得到：

![[公式]](https://www.zhihu.com/equation?tex=P%28k%29+%3D+%5Cfrac%7B%5Clambda%5Ek%7D%7Bk%21%7De%5E%7B-%5Clambda%7D+%5C%5C)



这个就是泊松分布的**概率密度函数**了，也就是说在一天中挑出k个正品的概率就是![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Clambda%5Ek%7D%7Bk%21%7De%5E%7B-%5Clambda%7D)。

也就是说泊松分布是我们将时间无限切分，然后套用二项分布利用数学极限推导出来的结果。本质上来说，它的内核仍然是二项分布。使用泊松分布的原因是，当n很大，p很小的时候，我们使用二项分布计算会非常困难，因为使用乘方计算出来的值会非常巨大，这个时候，我们使用泊松分布去逼近这个概率就很方便了。

## 4. 正态分布

正态曲线呈钟型，两头低，中间高，左右对称因其曲线呈钟形，也叫高斯分布。 是一个非常重要的分布。

**一维正态分布**

若随机变量![img](https://bkimg.cdn.bcebos.com/formula/b7f43d75b3354a3bb6dccf21c32bdeff.svg)、尺度参数 ![img](https://bkimg.cdn.bcebos.com/formula/29ecac7d98ef7b8b7d4399a961ff7b42.svg) 为概率密度函数：

![img](https://bkimg.cdn.bcebos.com/formula/d8fc1a3696534a47f23d6bcb60c1212c.svg)

则这个随机变量服从正态分布。

**标准正态分布**

当

![img](https://bkimg.cdn.bcebos.com/formula/040c60274885dfce9652570e92cf8dcc.svg)

 时，正态分布就成为标准正态分布

![img](https://bkimg.cdn.bcebos.com/formula/a49f2d97f625020c180a64346e8cece7.svg)

对应图像如下。

![1629368593450](../AppData/Roaming/Typora/typora-user-images/1629368593450.png)

 重要性质：

- 密度函数关于平均值对称
- 平均值与他的众数、中位数为同一值
- 函数曲线下68.268949%的面积在平均数左右的一个标准差范围内
- 95.449974%的面积在平均数左右两个标准差2 σ 2 \sigma2*σ*的范围内
- 99.730020%的面积在平均数左右三个标准差3 σ 3 \sigma3*σ*的范围内
- 函数曲线的拐点（inflection point）为离平均数一个标准差距离的位置。

## 4. 指数分布

指数分布与其他分布的最大不同之处在于，它所针对的随机变量X是不是指独立随机事件值，而是指**不同的独立事件发生之间时间间隔值的分布,时间越长发生的概率指数型增大(减小)**。在我们日常的消费领域，通常的**目的是求出在某个时间区间内，会发生随机事件的概率有多大**。如：银行窗口服务、交通管理、火车票售票系统、消费市场研究报告中被广泛运用。

**定义**

其中λ > 0是分布的一个参数，常被称为率参数（rate parameter）。即每单位时间内发生某事件的次数。指数分布的区间是[0,∞)。 如果一个随机变量*X*呈指数分布，则可以写作：*X*~ E（λ）。θ=1/λ,因此概率密度函数：

![img](https://bkimg.cdn.bcebos.com/formula/b26bb385caa4f7ad35ce1b115141e2d2.svg)

 

其中θ>0为常数，则称X服从参数θ的指数分布。

## 5. 拉普拉斯分布

 在概率论和统计学中，拉普拉斯是一种**连续概率**分布。由于它可以看做是俩个不同位置的指数分布背靠背拼在一起，所以它也叫做双指数分布。设随机变量![img](https://bkimg.cdn.bcebos.com/formula/959ad200f9fb074d206b223d046b88ba.svg) ，具有密度函数

![img](https://bkimg.cdn.bcebos.com/formula/6f0bb2dfe2e78f1766705087eea0bb81.svg)

其中![img](https://bkimg.cdn.bcebos.com/formula/b006e7ce47334512fcd4f914ad7811bf.svg)为常数，且![img](https://bkimg.cdn.bcebos.com/formula/0054fbc91fc17f8f6e212a2c4ba69e20.svg) ，则称![img](https://bkimg.cdn.bcebos.com/formula/959ad200f9fb074d206b223d046b88ba.svg)服从参数为![img](https://bkimg.cdn.bcebos.com/formula/b006e7ce47334512fcd4f914ad7811bf.svg)的拉普拉斯分布。 与正态分布相比，正态分布是用相**对于u平均值的差的平方**来表示，而拉普拉斯概率密度用**相对于差的绝对值**来表示。因此，拉普拉斯的尾部比正态分布更加平坦。

 ![1629369138404](../AppData/Roaming/Typora/typora-user-images/1629369138404.png)

## 



