---
title: 实习结束(shareit)
date: 2021-07-24  14:46:35
tags: 
categories: 实习
top: 1
---

![](https://gitee.com/z_saisai/ware_01/raw/master/doc_image/1.jpg)

<!-- more -->

![](https://gitee.com/z_saisai/ware_01/raw/master/doc_image/2.jpg)

![](https://gitee.com/z_saisai/ware_01/raw/master/doc_image/3.jpg)

# 001

1. 学习计算广告业务相关术语
2. 了解数据表字段含义与表关系
3. 了解arflow任务执行流程(离线日志数据抽取，写到线上的sharestore(一个基于内存的数据库))
4. 了解离线训练的特征生成过程，spark session相关会话
5. 数据的特征重要性分析(互信息，卡方检验)

# 002

1. 学习特征处理（embedding,hash,特征交叉）的主要流程

2. 了解训练脚本的主要内容
3. 了解网络模型（DeepFM）的主要结构

# 003

1. 了解模型的相关路径,超参数修改，从hadoop种分割出部分数据完成一个训练过程。
2. 尝试修改特征交叉相关的c++文件，了解对应特征的proto文件的message定义,与其bazel的编译过程。
3. 分析现有特征的组合，尝试增加ctx与ad的特征组合，进行离线训练。

# 004

1. 熟悉修改离线部分的proto文件增加特征的流程。
2. 了解model server, scmp的模块执行流程。
3. 尝试修改model server部分的特征，下周进行测试。

# 005

1. 处理embedding数据流，为线上做修改准备。
2. 封装docker镜像并部署scmp，并进行了初步的上线测试。
3. 继续修改model server basic特征组合部分，例行模型的特征调整实验。

# 006

1. model_ _server_ _basic修改与各项响应时间与P99测试，失败率测试 ，GRPC高并发测试。

2. model_ _server_ _basic合并master,构建上线，后期持续观察模型LOG / ERROR。

# 007

1. 修改model server与basic，添加用户分组桶请求信息的形参，修改RPC通信PROTO，打通server上游到basic的数据流，为下周尝试添加ctr线性衰减模块做数据准备。
2. 修改model server,在分桶2进行去除优先级加减2的实验。初步观察ecpm提升效果显著，继续观察。
3. 用户手机安装的applist与usercounter的embedding模型实验进行中。

# 008

1. 发现radis请求缓存在basic前就会起效果，导致basic中的ctr线性衰减策略模块失灵，向leader反应后，ctr线性衰减实验停止。
2. 加入线上数据到离线训练中，修改上游日志ETL的spark模块，生成临时数据表，在pipeline中直接通过SQL进行在线与离线数据的整合。