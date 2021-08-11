---
title:K-means clustering
date: 2021-02-27
tags: Machine Learning
categories: 机器学习
top:
---

# K-means clustering
k均值聚类算法（k-means clustering algorithm）是一种迭代求解的聚类分析算法，在这一章里，你将使用有效的数据集对k-means聚类算法进行分析，并了解到数据挖掘中的若干重要概念。

<!--more-->

## 背景介绍

k均值算法群集中的每个点都应靠近该群集的中心。要想实现kmeans算法，
首先我们选择k，即我们想要在数据中找到的簇数。然后，以某种方式初始化这k个簇的中心，称为质心。
然后，我们将数据中的每个点分配给质心最接近的点，然后将每个质心的位置重新计算为分配给其质心的所有点的均值。
下面介绍一个[网站](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210207100455218.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70#pic_center)

这个网站能把kmeans算法的聚类过程动态图画出来，如图我们点击Add Centroid，添加了四个中心点
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210207100505867.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70#pic_center)

然后点击GO，就能看到kmeans算法聚类的动画过程
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210207100514983.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70#pic_center)


**算法步骤**

1. 首先输入k的值，即我们希望将数据集经过聚类得到k个分组；

2. 从数据集中随机选择k个数据点作为初始聚类中心，对任意一个样本点，求其到K个聚类中心的距离，将样本点归类到距离最小的中心的聚类，如此迭代n次；

3. 每次迭代过程中，利用均值等方法更新各个聚类的中心点；

4. 对K个聚类中心，利用2,3步迭代更新后，如果位置点变化很小，则认为达到稳定状态，对不同的聚类块和聚类中心可选择不同的颜色标注。

**算法优缺点**
优点
1.kmeans算法原理比较简单，实现起来也很容易，收敛速度较快。

2.聚类效果相对于其他聚类算法来说较好，算法的可解释度比较强。

3.参数较少，主要需要调参的参数仅仅是簇数k。

缺点
1.字符串等非数值型数据不适用，kmeans算法基于均值计算，首先要求簇的平均值可以被定义和使用。

2.K-Means的第一步是确定k（要生成的簇的数目），对于不同的初始值K，可能会导致不同结果。

3.应用数据集存在局限性，适用于球状或集中分布数据，不适用于特殊情况数据。
## 算法定义
k均值聚类是最著名的划分聚类算法，给定一个数据点集合和需要的聚类数目k，k由用户指定，其主要目的是将n个样本点划分为k个簇，使得相似的样本尽量被分到同一个聚簇。

## K值的选取
在实际应用中，由于Kmeans一般作为数据预处理，或者用于辅助分类贴标签。所以k一般不会设置很大。对于k值的选取，一般采用以下几种方法。

**手肘法**

手肘法的核心指标是SSE(sum of the squared errors误差平方和)

聚类数k增大，数据划分就会更精细，每个簇的聚合程度也会提高，从而误差平方和SSE会逐渐变小。当k小于真实聚类数时，k的增大会增加每个簇的聚合程度，
故SSE的下降幅度会很大。当k到达真实聚类数时，再增加k所得到的聚合程度会迅速变小，所以SSE的下降幅度会骤减，
然后随着k值的继续增大而趋于平缓，由此可见SSE和k的关系图是
一个手肘的形状，而这个肘部对应的k值就是数据的真实聚类数。

具体做法是让k从1开始取值直到取到你认为合适的上限，对每一个k值进行聚类并且记下对于的SSE，然后画出k和SSE的关系图，
最后选取肘部对应的k作为我们的最佳聚类数。
下图利用了UCI上葡萄酒的数据集[wine.data](http://archive.ics.uci.edu/ml/datasets/Wine)，然后用sklearn库中自带的k-means算法对K值的选取进行了可视化操作。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021020710042291.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70#pic_center)


 **轮廓系数法**
轮廓系数（silhouette coefficient）是簇的密集与分散程度的评价指标
1. 计算样本i到同一类中其他样本的平均距离ai。如果ai越小，说明样本i与同类中其他样本的距离越近，即越相似。我们将ai称为样本i的类别内不相似度。

2. 计算样本i到其他类别的所有样本的平均距离bi，称为样本i与其他类之间的不相似度。如果bi越大，说明样本i与其他类之间距离越远，即越不相似。

轮廓系数的值在-1和1之间，该值越接近于1，簇越紧凑，聚类越好。当轮廓系数接近1时，簇内紧凑，并远离其他簇。其中，轮廓系数的计算如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021020710044364.png#pic_center)
## 案例实现（python）
### 数据集
**数据集介绍**
本章的数据样本来源于[UCI](http://archive.ics.uci.edu/ml/datasets/seeds)上的小麦种子数据集。部分数据集如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210207100817268.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70#pic_center)


其中，数据集包含210个样本。每个样品具有7个属性：面积A，周长P，紧密度C = 4piA / P ^ 2，籽粒长度，籽粒宽度，不对称系数和籽粒凹槽长度。 
所有这些参数都是实值连续的。数据集包含属于三种不同小麦品种的籽粒：卡玛，罗莎和加拿大。

| 数据集特点：| 多元 | 实例数量： | 210 | 领域： | 生活类 |
| ------| ------ | ------ | ------| ------ | ------ |
| 属性特征： | 真实 | 属性数量： | 7 | 上传日期： | 2012.9.29 |
| 相关人物： | 分类/聚类 | 缺值： | N/A | 浏览次数： | 326689 |

### 代码实现
1、导入我们需要的库

```python
import matplotlib.pyplot as plt
from numpy import *
import numpy as np  
```
2、计算欧氏距离

```python
#利用公式计算两个向量p1与p2之间的距离，返回diff
def get_distance(p1, p2):
    diff = (sum(power(p1 - p2, 2)))
    return diff
```
3、导入数据

```python
#1、根据数据文件的路径file_path打开样本数据
#2、将存储的文本信息转换成向量并传出
def load_data(file_path):
    f = open(file_path)
    data = []
    for line in f.readlines():
        row = []
        lines = line.strip().split()
        for x in lines:
            row.append(float(x))
        data.append(row)
    f.close()
    return np.mat(data)
```
4、初始化聚类中心

```python
#参数 data:样本数据点集合
#参数 k:已确定的最终聚类的类别数
#return:聚类中心集合my_cluster_center
def random_data(my_data, k):
    sum = np.shape(my_data)[1]  # 特征个数
    print(sum)
    my_cluster_center = np.mat(np.zeros((k, sum)))  # 初始化k个聚类中心
    for j in range(sum):  # 最小值+0-1之间的随机数*变化范围，即得到在最大最小值之间的随机数
        Xmin = np.min(my_data[:, j])  #得到最小值
        Xrange = np.max(my_data[:, j]) - Xmin  #得到随机数的变化范围
        my_cluster_center[:, j] = Xmin * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * Xrange
    print('初始化的聚类中心如下：')
    print(my_cluster_center)
    return my_cluster_center
```
5、k_means算法聚类分析

```python
#参数ata:样本数据点集合
#参数k:聚类中心的个数
#参数cluster_center:传进来的为初始的聚类中心集合
#首先，在数据集中选择K个点作为每个簇的初始中心，接着观察其他的数据，然后将这些数据划分到距离这K个点最近的簇中， 这个过程将数据划分成K个簇，从而完成第一次划分。但实际算法过程中，形成的新簇不一定就是最好的划分，因此在生成的新簇中， 再一次计算每个簇的中心点，然后进行划分，直到每次划分的结果保持不变为止。
def k_means(data, k, cluster_center):
    m, n = np.shape(data)  # m:样本个数；n:特征的维度
    sub_center = np.mat(np.zeros((m, 2)))  # 初始化：每个样本所属的类别（m行n列：共m个样本，第一列为所属类的标号，第二列为最小距离）
    change = 1  # 判断是否重新计算聚类中心
    while change == 1:
        change = 0
        for i in range(m):
            min_distance = np.inf  # 初始样本与聚类中心的最小值为正无穷
            Index = 0  #所属类别
            for j in range(k):
                # 分别计算i到这k个聚类中心的距离，找到距离最近的
                diff = get_distance(data[i,], cluster_center[j,])
                if diff < min_distance:
                    min_distance = diff
                    Index = j
            # 判断所属聚类中心是否发生变化（可能原本就属于这个聚类中心）
            if sub_center[i, 0] != Index:
                change = 1
                sub_center[i,] = np.mat([Index, min_distance])
        # 重新计算聚类中心
        for j in range(k):
            all = np.mat(np.zeros((1, n)))  # all记录所有属于中心j的样本，在n个维度的总和
            r = 0  # 每个类别中的样本个数
            for i in range(m):
                if sub_center[i, 0] == j:  # 属于第j个类别，计算进去
                    all = all + data[i,]
                    r = r + 1
            for a in range(n):
                try:  # sum_all除以本中心的样本个数r，即得到中心
                    cluster_center[j, a] = all[0, a] / r
                except:
                    print("没有样本属于这个聚类中心！")
        # 打印，显示聚类中心的变化过程
        print("聚类中心发生变化如下：")
        print(cluster_center)
    return cluster_center, sub_center
```
6、保存数据

```python
#把数据data写入Myfile_name文件中，并保存
def save_result(Myfile_name, data):
    m, n = np.shape(data)
    f = open(Myfile_name, "w")
    for i in range(m):
        X = []
        for j in range(n):
            X.append(str(data[i, j]))
        f.write("\t".join(X) + "\n")
    f.close()
```
7、画图

```python
#输出数据样本点的总个数，并根据数据进行聚类分析，对每一个属性进行聚类，并绘图显示
def draw(point_data, center, sub_center):
    Myfig = plt.figure()
    axes = Myfig.add_subplot(111)

    length = len(point_data)
    print(length)

    for a in range(length):
        if sub_center[a, 0] == 0:
            axes.scatter(point_data[a, 2], point_data[a, 3], color='m', alpha=0.4)
        if sub_center[a, 0] == 1:
            axes.scatter(point_data[a, 2], point_data[a, 3], color='b', alpha=0.4)
        if sub_center[a, 0] == 2:
            axes.scatter(point_data[a, 2], point_data[a, 3], color='g', alpha=0.4)
    for i in range(len(center)):
        axes.scatter(center[i, 2], center[i, 3], color='red', marker='p')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('kmeans-JYZ')
    plt.show()

```
8、主函数

```python
if __name__ == "__main__":
    K = 3  # 预设好聚类中心数k
    print("第一步：导入样本数据")
    sample_data = load_data("seeds.data")

    print("第二步：初始化k个聚类中心")
    cluster_center = random_data(sample_data, K)

    print("第三步：K_Means算法")
    cluster_center, sub_center = k_means(sample_data, K, cluster_center)

    print("第四步：保存最终的聚类中心")
    save_result("cluster_center.txt", cluster_center)

    print("第五步：保存每个样本所属类别")
    save_result("sub_center.txt", sub_center)

    # 画图
    draw(sample_data, cluster_center, sub_center)
```

### 运行结果

```python
第一步：导入样本数据
第二步：初始化k个聚类中心
8
初始化的聚类中心如下：
[[17.51686311 16.40829437  0.91619839  6.60986498  3.15821884  5.49005513
   4.80842984  1.67208591]
 [19.37741169 16.76073519  0.91290766  6.40031021  3.85596015  2.23395909
   5.15480027  2.34787189]
 [14.9196659  13.09405509  0.82642333  5.51808314  3.06647918  3.5302489
   4.7662341   2.10372636]]
第三步：K_Means算法
聚类中心发生变化如下：
[[17.81384615 15.93923077  0.87981154  6.0735      3.61865385  4.72330769
   5.90411538  1.92307692]
 [19.0005     16.40125     0.8873      6.24715     3.7551      2.935375
   6.11635     1.975     ]
 [13.15833333 13.79847222  0.86487917  5.37635417  3.05568056  3.727925
   5.12176389  2.02083333]]
聚类中心发生变化如下：
[[17.23121212 15.66787879  0.88107879  5.97942424  3.56660606  4.40127273
   5.81918182  1.84848485]
 [18.9372093  16.36790698  0.88761628  6.23623256  3.74776744  2.91974419
   6.07788372  1.90697674]
 [12.94813433 13.70589552  0.86318358  5.34711194  3.02578358  3.77799403
   5.09188806  2.06716418]]
聚类中心发生变化如下：
[[16.62634146 15.39829268  0.88052439  5.89009756  3.49668293  3.75556341
   5.67960976  1.65853659]
 [19.19767442 16.48046512  0.8879907   6.26725581  3.77960465  3.21153488
   6.12765116  2.        ]
 [12.78412698 13.63063492  0.8621      5.32544444  3.00333333  3.84895317
   5.07414286  2.11111111]]
聚类中心发生变化如下：
[[15.944      15.0836      0.880146    5.78036     3.42138     3.133462
   5.50546     1.44      ]
 [19.15104167 16.46916667  0.88708958  6.26885417  3.7729375   3.46041667
   6.12725     2.        ]
 [12.51366071 13.50669643  0.86001875  5.28633036  2.96550893  4.05597411
   5.056375    2.25      ]]
聚类中心发生变化如下：
[[15.38075758 14.81636364  0.8797803   5.68925758  3.35859091  2.94465303
   5.36257576  1.28787879]
 [19.06       16.43176471  0.88679804  6.25333333  3.76323529  3.51547059
   6.11290196  2.        ]
 [12.15903226 13.35        0.85610215  5.24280645  2.91091398  4.3377
   5.05383871  2.50537634]]
聚类中心发生变化如下：
[[15.06763889 14.66194444  0.87993194  5.63831944  3.32394444  2.84957222
   5.28854167  1.22222222]
 [18.96296296 16.39666667  0.88595185  6.24272222  3.74992593  3.54033333
   6.10077778  2.        ]
 [12.01321429 13.29011905  0.85372857  5.22530952  2.88675     4.53208333
   5.06521429  2.66666667]]
聚类中心发生变化如下：
[[14.84614286 14.54828571  0.88059429  5.59571429  3.30232857  2.74761714
   5.22582857  1.15714286]
 [18.78677966 16.32915254  0.88474746  6.22098305  3.72767797  3.58188136
   6.07911864  2.        ]
 [11.97938272 13.27962963  0.85269136  5.22535802  2.87914815  4.60960494
   5.07677778  2.72839506]]
聚类中心发生变化如下：
[[14.67805556 14.47152778  0.87971389  5.56618056  3.28319444  2.71346111
   5.18791667  1.13888889]
 [18.72180328 16.29737705  0.88508689  6.20893443  3.72267213  3.60359016
   6.06609836  1.98360656]
 [11.93675325 13.26441558  0.85168831  5.22703896  2.86797403  4.6994026
   5.09263636  2.81818182]]
聚类中心发生变化如下：
[[14.63202703 14.45324324  0.8790973   5.56178378  3.27489189  2.74404324
   5.18493243  1.13513514]
 [18.72180328 16.29737705  0.88508689  6.20893443  3.72267213  3.60359016
   6.06609836  1.98360656]
 [11.90906667 13.25026667  0.85154933  5.22233333  2.86509333  4.72218667
   5.09304     2.86666667]]
聚类中心发生变化如下：
[[14.63202703 14.45324324  0.8790973   5.56178378  3.27489189  2.74404324
   5.18493243  1.13513514]
 [18.72180328 16.29737705  0.88508689  6.20893443  3.72267213  3.60359016
   6.06609836  1.98360656]
 [11.90906667 13.25026667  0.85154933  5.22233333  2.86509333  4.72218667
   5.09304     2.86666667]]
第四步：保存最终的聚类中心
第五步：保存每个样本所属类别
210

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021020710144694.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70#pic_center)
## 总结
在这里，我们借助小麦种子seeds.data的数据集，介绍了K均值聚类算法的概念，以及利用python实现算法聚类分析以及可视化的过程， 并分析了kmeans算法的优缺点。主要思想就是把相似对象归为同一簇，把不相似对象归到不同簇。簇内的对象越相似，聚类的效果越好。
## 参考文献
1.https://blog.csdn.net/weixin_42029738/article/details/81978038

2.https://blog.csdn.net/sinat_36710456/article/details/88019323

3.Ahmad EL ALLAOUI. Clustering Kmeans with Evolutionary Strategies[J]. International Journal of Imaging and Robotics™,2017,17(3).

4.刘子熠,张喆,高天,武强. Analysis of Pump Data Based on Association and Kmeans Algorithm[J]. 数据挖掘,2020,10(02).

5.Yi Yi Aung,Myat Myat Min. An Analysis of Kmeans Algorithm Based Network Intrusion Detection System[J]. Advances in Science, Technology and Engineering Systems,2018,3(1).
