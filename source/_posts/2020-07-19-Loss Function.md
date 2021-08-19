---
title: 损失函数
date: 2020-07-19 11:43:45
tags: Machine Learning
categories: 机器学习
top:
mathjax: true
---
# 1. 概念区分
**Loss Function**
损失函数 Loss Function 

损失函数 Loss Function还分为**经验风险损失函数**和**结构风险损失函数**。经验风险损失函数指预测结果和实际结果的差别，通常是针对单个训练样本而言，给定一个模型输出  y^ 和一个真实 y ，损失函数输出一个实值损失y - y^ 。用来评价模型的**预测值**和**真实值**不一样的程度，损失函数越好，通常模型的性能越好。不同的模型用的损失函数一般也不一样。结构风险损失函数是指经验风险损失函数加上**正则项**。

**Cost Function** 

代价函数 Cost Function 通常是针对整个训练集（或者在使用 mini-batch gradient descent 时一个 mini-batch）的总损失 。

**Objective Function**

目标函数 Objective Function 是一个更通用的术语，表示任意希望被优化的函数，用于机器学习领域和非机器学习领域（比如运筹优化）

即：损失函数和代价函数只是在针对样本集上有区别。
<!-- more -->

# 2.回归常用损失函数

## 2.1 均方差损失（MSE）

- 均方差 Mean Squared Error (MSE)损失是机器学习、深度学习回归任务中最常用的一种损失函数，也称为 L2 Loss。其基本形式如下

  ![](https://www.zhihu.com/equation?tex=J_%7BMSE%7D+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28y_i+-+%5Chat%7By_i%7D%29%5E2+%5C%5C)

- 原理推导

  实际上在一定的假设下，我们可以使用最大化似然得到均方差损失的形式。假设**模型预测与真实值之间的误差服从标准高斯分布**（正态分布）（ ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu%3D0%2C+%5Csigma%3D1) ），则给定一个 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 模型输出真实值 ![[公式]](https://www.zhihu.com/equation?tex=y_i) 的概率为


  ![[公式]](https://www.zhihu.com/equation?tex=p%28y_i%7Cx_i%29+%3D+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7D%5Cmathbb%7Bexp%7D%5Cleft+%28-%5Cfrac%7B%28y_i-%5Chat%7By_i%7D%29%5E2%7D%7B2%7D%5Cright+%29+%5C%5C)

  进一步我们假设数据集中 N 个样本点之间相互独立，则给定所有 ![[公式]](https://www.zhihu.com/equation?tex=x) 输出所有真实值 ![[公式]](https://www.zhihu.com/equation?tex=y) 的概率，即似然 函数，为所有 ![[公式]](https://www.zhihu.com/equation?tex=p%28y_i+%5Cvert+x_i%29) 的累乘

  ![[公式]](https://www.zhihu.com/equation?tex=L%28x%2C+y%29+%3D+%5Cprod_%7Bi%3D1%7D%5E%7BN%7D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7D%5Cmathbb%7Bexp%7D%5Cleft+%28-%5Cfrac%7B%28y_i-%5Chat%7By_i%7D%29%5E2%7D%7B2%7D%5Cright%29+%5C%5C)

  通常为了计算方便，我们通常最大化对数似然（通过去log,将乘机转为和的形式,方便计算）

  ![[公式]](https://www.zhihu.com/equation?tex=LL%28x%2C+y%29%3D%5Cmathbb%7Blog%7D%28L%28x%2C+y%29%29%3D-%5Cfrac%7BN%7D%7B2%7D%5Cmathbb%7Blog%7D2%5Cpi+-+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5E%7BN%7D+%28y_i-%5Chat%7By_i%7D%29%5E2+%5C%5C)

  去掉与 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D) 无关的第一项，将最大化对数似然转化为最小化**负**对数似然 ，就可以使用梯度下降等方法对其求最小值了。

  ![[公式]](https://www.zhihu.com/equation?tex=NLL%28x%2C+y%29+%3D+%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28y_i+-+%5Chat%7By_i%7D%29%5E2+%5C%5C)

  可以看到这个实际上就是均方差损失的形式。也就是说**在模型输出与真实值的误差服从高斯分布的假设下，最小化均方差损失函数与极大似然估计本质上是一致的**，因此在这个假设能被满足的场景中（比如回归），均方差损失是一个很好的损失函数选择；当这个假设没能被满足的场景中（比如分类），均方差损失不是一个好的选择。

## 2.2 平均绝对损失 (MAE)

- 平均绝对误差 Mean Absolute Error (MAE)是另一类常用的损失函数，也称为 L1 Loss。其基本形式如下

![[公式]](https://www.zhihu.com/equation?tex=+J_%7BMAE%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft+%7C+y_i+-+%5Chat%7By_i%7D+%5Cright+%7C+%5C%5C)

MAE 损失的最小值为 0（当预测等于真实值时），最大值为无穷大。随着预测与真实值绝对误差 ![[公式]](https://www.zhihu.com/equation?tex=%5Clvert+y-+%5Chat%7By%7D%5Crvert) 的增加，MAE 损失呈线性增长.

- 原理推导

同样的我们可以在一定的假设下通过最大化似然得到 MAE 损失的形式，假设**模型预测与真实值之间的误差服从拉普拉斯分布 Laplace distribution**（ ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu%3D0%2C+b%3D1) ），则给定一个 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 模型输出真实值 ![[公式]](https://www.zhihu.com/equation?tex=y_i) 的概率为

![[公式]](https://www.zhihu.com/equation?tex=p%28y_i%7Cx_i%29+%3D+%5Cfrac%7B1%7D%7B2%7D%5Cmathbb%7Bexp%7D%28-%5Cleft+%7Cy_i-%5Chat%7By_i%7D%5Cright%7C%29+%5C%5C)

与上面推导 MSE 时类似，我们可以得到的负对数似然实际上就是 MAE 损失的形式

![[公式]](https://www.zhihu.com/equation?tex=L%28x%2C+y%29+%3D+%5Cprod_%7Bi%3D1%7D%5E%7BN%7D%5Cfrac%7B1%7D%7B2%7D%5Cmathbb%7Bexp%7D%28-%7Cy_i-%5Chat%7By_i%7D%7C%29%5C%5C+++LL%28x%2C+y%29+%3D+-%5Cfrac%7BN%7D%7B2%7D+-+%5Csum_%7Bi%3D1%7D%5E%7BN%7D+%7Cy_i-%5Chat%7By_i%7D%7C+%5C%5C+++NLL%28x%2C+y%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7BN%7D+%7Cy_i-%5Chat%7By_i%7D%7C++%5C%5C)

#### MAE与MSE的比较

**1. MSE 通常比 MAE 可以更快地收敛**。当使用梯度下降算法时，MSE 损失的梯度为 ![[公式]](https://www.zhihu.com/equation?tex=-%5Chat%7By_i%7D) ，而 MAE 损失的梯度为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpm1) ，即 MSE 的梯度的 scale 会随误差大小变化，而 MAE 的梯度的 scale 则一直保持为 1，即便在绝对误差 ![[公式]](https://www.zhihu.com/equation?tex=%5Clvert+y_i-%5Chat%7By_i%7D+%5Crvert) 很小的时候 MAE 的梯度 scale 也同样为 1，这实际上是非常不利于模型的训练的。当然你可以通过在训练过程中动态调整学习率缓解这个问题，但是总的来说，损失函数梯度之间的差异导致了 MSE 在大部分时候比 MAE 收敛地更快。这个也是 MSE 更为流行的原因。

**2. MAE 对于 outlier 更加 robust**。我们可以从两个角度来理解这一点：

第一个角度是直观地理解，下图是 MAE 和 MSE 损失画到同一张图里面，由于MAE 损失与绝对误差之间是线性关系，MSE 损失与误差是平方关系，当误差非常大的时候，MSE 损失会远远大于 MAE 损失。因此当数据中出现一个误差非常大的 outlier 时，MSE 会产生一个非常大的损失，对模型的训练会产生较大的影响。

![img](https://pic2.zhimg.com/80/v2-c8edffe0406dafae41a042e412cd3251_1440w.jpg)

​	第二个角度是从两个损失函数的假设出发，MSE 假设了误差服从高斯分布，MAE 假设了误差服从拉普拉斯分布。拉普拉斯分布本身对于 outlier 更加 robust，当右图右侧出现了 outliers 时，拉普拉斯分布相比高斯分布受到的影响要小很多。因此以拉普拉斯分布为假设的 MAE 对 outlier 比高斯分布为假设的 MSE 更加 robust。

![img](https://pic1.zhimg.com/80/v2-93ad65845f5b0dc0327fde4ded661804_1440w.jpg)

## 2.3 Huber Loss

上文我们分别介绍了 MSE 和 MAE 损失以及各自的优缺点，MSE 损失收敛快但容易受 outlier 影响，MAE 对 outlier 更加健壮但是收敛慢，Huber Loss则是一种将 MSE 与 MAE 结合起来，取两者优点的损失函数，也被称作 Smooth Mean Absolute Error Loss 。其原理很简单，就是在误差接近 0 时使用 MSE，误差较大时使用 MAE，公式为

![[å¬å¼]](https://www.zhihu.com/equation?tex=J_%7Bhuber%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%5Cmathbb%7BI%7D_%7B%7C+y_i+-+%5Chat%7By_i%7D%7C+%5Cleq+%5Cdelta%7D+%5Cfrac%7B%28y_i+-+%5Chat%7By_i%7D%29%5E2%7D%7B2%7D%2B+%5Cmathbb%7BI%7D_%7B%7C+y_i+-+%5Chat%7By_i%7D%7C+%3E+%5Cdelta%7D+%28%5Cdelta+%7Cy_i+-+%5Chat%7By_i%7D%7C+-+%5Cfrac%7B1%7D%7B2%7D%5Cdelta%5E2%29+%5C%5C)

上式中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta) 是 Huber Loss 的一个超参数，![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta) 的值是 MSE 和 MAE 两个损失连接的位置。上式等号右边第一项是 MSE 的部分，第二项是 MAE 部分，在 MAE 的部分公式为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta+%5Clvert+y_i+-+%5Chat%7By_i%7D%5Crvert+-+%5Cfrac%7B1%7D%7B2%7D%5Cdelta%5E2)是为了保证误差 ![[公式]](https://www.zhihu.com/equation?tex=%5Clvert+y+-+%5Chat%7By%7D%5Crvert%3D%5Cpm+%5Cdelta) 时 MAE 和 MSE 的取值一致，进而保证 Huber Loss 损失连续可导。

下图是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta%3D1.0) 时的 Huber Loss，可以看到在 ![[公式]](https://www.zhihu.com/equation?tex=%5B-%5Cdelta%2C+%5Cdelta%5D) 的区间内实际上就是 MSE 损失，在![[公式]](https://www.zhihu.com/equation?tex=%28-%5Cinfty%2C+%5Cdelta%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=%28%5Cdelta%2C+%5Cinfty%29) 区间内为 MAE损失。

![img](https://pic4.zhimg.com/80/v2-b4260d38f70dd920fa46b8717596bda7_1440w.jpg)

- Huber Loss 的特点

Huber Loss 结合了 MSE 和 MAE 损失，在误差接近 0 时使用 MSE，使损失函数可导并且梯度更加稳定；在误差较大时使用 MAE 可以降低 outlier 的影响，使训练对 outlier 更加健壮。缺点是需要额外地设置一个 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta) 超参数。

- Huber Loss python实现

```python
# huber 损失
def huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)
```



# 3.分类常用损失函数

## 3.1 交叉熵损失

### 3.1.1 二分类

- 考虑二分类，在二分类中我们通常使用Sigmoid函数将模型的输出压缩到 (0, 1) 区间内![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D+%5Cin+%280%2C+1%29) ，用来代表给定输入 ![[公式]](https://www.zhihu.com/equation?tex=x_i) ，模型判断为正类的概率。下图是对二分类的交叉熵损失函数的可视化，蓝线是目标值为 0 时输出不同输出的损失，黄线是目标值为 1 时的损失。可以看到约接近目标值损失越小，随着误差变差，损失呈指数增长。

![[公式]](https://www.zhihu.com/equation?tex=NLL%28x%2C+y%29%3DJ_%7BCE%7D%3D-%5Csum_%7Bi%3D1%7D%5EN%5Cleft+%28y_i%5Cmathbb%7Blog%28%7D%5Chat%7By_i%7D%29+%2B+%281-+y_i%29%5Cmathbb%7Blog%7D%281-%5Chat%7By_i%7D%29%5Cright%29+%5C%5C)

![img](https://pic2.zhimg.com/80/v2-7e7732b869d7334c2c960c1089b13439_1440w.jpg)



- 原理推导

由于只有分两类，因此同时也得到了正负类的概率， 则可以假设样本服从**伯努利分布（0-1分布）** 。

![[公式]](https://www.zhihu.com/equation?tex=p%28y_i%3D1%7Cx_i%29+%3D+%5Chat%7By_i%7D%5C%5C+++p%28y_i%3D0%7Cx_i%29+%3D+1-%5Chat%7By_i%7D++%5C%5C)

将两条式子合并成一条

![[公式]](https://www.zhihu.com/equation?tex=p%28y_i%7Cx_i%29+%3D+%28%5Chat%7By_i%7D%29%5E%7By_i%7D+%281-%5Chat%7By_i%7D%29%5E%7B1-y_i%7D+%5C%5C)

假设数据点之间独立同分布，则似然函数，即各样本的概率乘机可以表示为

![[公式]](https://www.zhihu.com/equation?tex=L%28x%2C+y%29%3D%5Cprod_%7Bi%3D1%7D%5EN%28%5Chat%7By_i%7D%29%5E%7By_i%7D+%281-%5Chat%7By_i%7D%29%5E%7B1-y_i%7D+%5C%5C)

对似然取对数，然后加负号变成最小化负对数似然，即为交叉熵损失函数的形式

![[公式]](https://www.zhihu.com/equation?tex=NLL%28x%2C+y%29%3DJ_%7BCE%7D%3D-%5Csum_%7Bi%3D1%7D%5EN%5Cleft+%28y_i%5Cmathbb%7Blog%28%7D%5Chat%7By_i%7D%29+%2B+%281-+y_i%29%5Cmathbb%7Blog%7D%281-%5Chat%7By_i%7D%29%5Cright%29+%5C%5C)

### 3.1.2 多分类

在多分类的任务中，交叉熵损失函数的推导思路和二分类是一样的，变化的地方有两个：

1. 真实值 ![[公式]](https://www.zhihu.com/equation?tex=y_i) 现在是一个 One-hot 向量，每个值代表每个类对应的概率，和为1

2. 模型输出的激活函数由原来的 Sigmoid 函数换成 Softmax函数

Softmax 函数将每个维度的输出范围都限定在 ![[公式]](https://www.zhihu.com/equation?tex=%280%2C+1%29) 之间，同时所有维度的输出和为 1，用于表示一个概率分布。

举例：假设一个5分类问题，然后一个样本I的标签y=[0,0,0,1,0]，也就是说样本I的真实标签是4。

假设模型预测的结果概率（softmax的输出）p=[0.1,0.15,0.05,0.6,0.1]，可以看出这个预测是对的，那么对应的损失L=-log(0.6)，也就是当这个样本经过这样的网络参数产生这样的预测p时，它的损失是-log(0.6)。

那么假设p=[0.15,0.2,0.4,0.1,0.15]，这个预测结果就很离谱了，因为真实标签是4，而你觉得这个样本是4的概率只有0.1（远不如其他概率高，如果是在测试阶段，那么模型就会预测该样本属于类别3），对应损失L=-log(0.1)。

再假设p=[0.05,0.15,0.4,0.3,0.1]，这个预测结果虽然也错了，但是没有前面那个那么离谱，对应的损失L=-log(0.3)。我们知道log函数在输入小于1的时候是个负数，而且log函数是递增函数，所以-log(0.6) < -log(0.3) < -log(0.1)。简单讲就是你预测错比预测对的损失要大，预测错得离谱比预测错得轻微的损失要大。 

- 原理推导

我们知道Softmax 函数将每个维度的输出范围都限定在 ![[公式]](https://www.zhihu.com/equation?tex=%280%2C+1%29) 之间，同时所有维度的输出和为 1，用于表示一个概率分布。那末对应类的概率可以表示为。

![[公式]](https://www.zhihu.com/equation?tex=p%28y_i%7Cx_i%29+%3D+%5Cprod_%7Bk%3D1%7D%5EK%28%5Chat%7By_i%7D%5Ek%29%5E%7By_i%5Ek%7D+%5C%5C)

假设模型输出![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D)=[0.1,0.15,0.05,0.6,0.1] ,真实样本为[0,0,0,1,0]，计算得p刚好为0.6,非1项的幂都为1了。

其中 ![[公式]](https://www.zhihu.com/equation?tex=k+%5Cin+K) 表示 K 个类别中的一类，同样的假设数据点之间独立同分布，可得到负对数似然为

![[公式]](https://www.zhihu.com/equation?tex=NLL%28x%2C+y%29+%3D+J_%7BCE%7D+%3D+-%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bk%3D1%7D%5EK+y_i%5Ek+%5Cmathbb%7Blog%7D%28%5Chat%7By_i%7D%5Ek%29+%5C%5C)

由于 ![[公式]](https://www.zhihu.com/equation?tex=y_i) 是一个 one-hot 向量，除了目标类为 1 之外其他类别上的输出都为 0，因此上式也可以写为

![[公式]](https://www.zhihu.com/equation?tex=J_%7BCE%7D+%3D+-%5Csum_%7Bi%3D1%7D%5EN+y_i%5E%7Bc_i%7D%5Cmathbb%7Blog%7D%28%5Chat%7By_i%7D%5E%7Bc_i%7D%29+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=c_i) 是样本 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 的目标类。通常这个应用于多分类的交叉熵损失函数也被称为 Softmax Loss 或者 Categorical Cross Entropy Loss。

## 分类为什么是交叉熵

分类中为什么不用均方差损失？上文在介绍均方差损失的时候讲到实际上均方差损失假设了误差服从高斯分布，在分类任务下这个假设没办法被满足，因此效果会很差。为什么是交叉熵损失呢？有两个角度可以解释这个事情。

**一个角度从最大似然的角度，也就是我们上面的推导**，**另一个角度是可以用信息论来解释交叉熵损失**

以下是信息论的角度来解释。

假设对于样本 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 存在一个最优分布 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5E%7B%5Cstar%7D) 真实地表明了这个样本属于各个类别的概率，那么我们希望模型的输出 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D) 尽可能地逼近这个最优分布，在信息论中，我们可以使用 KL 散度来衡量两个分布的相似性。给定分布 ![[公式]](https://www.zhihu.com/equation?tex=p) 和分布 ![[公式]](https://www.zhihu.com/equation?tex=q) ， 两者的 KL 散度公式如下

![[公式]](https://www.zhihu.com/equation?tex=+KL%28p%2C+q%29%3D%5Csum_%7Bk%3D1%7D%5EKp%5Ek%5Cmathbb%7Blog%7D%28p%5Ek%29+-+%5Csum_%7Bk%3D1%7D%5EKp%5Ek%5Cmathbb%7Blog%7D%28q%5Ek%29+%5C%5C)

其中第一项为分布 ![[公式]](https://www.zhihu.com/equation?tex=p) 的信息熵，第二项为分布 ![[公式]](https://www.zhihu.com/equation?tex=p) 和 ![[公式]](https://www.zhihu.com/equation?tex=q) 的交叉熵。将最优分布 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5E%7B%5Cstar%7D) 和输出分布![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D) 带入 ![[公式]](https://www.zhihu.com/equation?tex=p) 和 ![[公式]](https://www.zhihu.com/equation?tex=q) 得到

![[公式]](https://www.zhihu.com/equation?tex=KL%28y_i%5E%7B%5Cstar%7D%2C+%5Chat%7By_i%7D%29%3D%5Csum_%7Bk%3D1%7D%5EKy_i%5E%7B%5Cstar+k%7D%5Cmathbb%7Blog%7D%28y_i%5E%7B%5Cstar+k%7D%29+-+%5Csum_%7Bk%3D1%7D%5EKy_i%5E%7B%5Cstar+k%7D%5Cmathbb%7Blog%7D%28%5Chat%7By_i%7D%5Ek%29+%5C%5C)

由于我们希望两个分布尽量相近，因此我们最小化 KL 散度。同时由于上式第一项信息熵仅与最优分布本身相关，因此我们在最小化的过程中可以忽略掉，变成最小化

![[公式]](https://www.zhihu.com/equation?tex=-%5Csum_%7Bk%3D1%7D%5EKy_i%5E%7B%5Cstar+k%7D%5Cmathbb%7Blog%7D%28%5Chat%7By_i%7D%5Ek%29+%5C%5C)

我们并不知道最优分布 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5E%7B%5Cstar%7D) ，但训练数据里面的目标值 ![[公式]](https://www.zhihu.com/equation?tex=y_i) 可以看做是 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5E%7B%5Cstar%7D) 的一个近似分布

![[公式]](https://www.zhihu.com/equation?tex=-+%5Csum_%7Bk%3D1%7D%5EKy_i%5Ek%5Cmathbb%7Blog%7D%28%5Chat%7By_i%7D%5Ek%29+%5C%5C)

这个是针对单个训练样本的损失函数，如果考虑整个数据集，则

![[公式]](https://www.zhihu.com/equation?tex=J_%7BKL%7D+%3D+-%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bk%3D1%7D%5EK+y_i%5Ek+%5Cmathbb%7Blog%7D%28%5Chat%7By_i%7D%5Ek%29%3D-%5Csum_%7Bi%3D1%7D%5EN+y_i%5E%7Bc_i%7D%5Cmathbb%7Blog%7D%28%5Chat%7By_i%7D%5E%7Bc_i%7D%29+%5C%5C)

可以看到**通过最小化交叉熵的角度推导出来的结果和使用最大化似然得到的结果是一致的**。

# 总结

- 交叉熵函数与最大似然函数的联系和区别

区别：**交叉熵函数**使用来描述模型预测值和真实值的差距大小，越大代表越不相近；**似然函数**的本质就是衡量在某个参数下，整体的估计和真实的情况一样的概率，越大代表越相近。

联系：**交叉熵函数**可以由最大似然函数在**伯努利分布**的条件下推导出来，或者说**最小化交叉熵函数**的本质就是**对数似然函数的最大化**。

- 分类中的交叉熵损失相当于在数据分布为伯努利分布的情况下的log损失，均方损失相当于数据在高斯分布下的log损失，绝对值损失则是在拉普拉斯分布下的log损失。 

- 通常在损失函数中还会有正则项（L1/L2 正则），这些正则项作为损失函数的一部分，通过约束参数的绝对值大小以及增加参数稀疏性来降低模型的复杂度，防止模型过拟合。



