---
title: 数据库原理-模式分解
date: 2020-04-24  14:46:35
tags: 数据库
categories: 数据库
top:
---

# 模式分解的要求
对存在数据冗余，插入异常与删除异常问题的关系模式，应采取将一个关系模式分为多个关系模式的方法进行处理，相应得，原来存储在一个二维表内的数据就要分散到多个二维表中，要使分解有意义，起码的要求是不能丢失前者的信息。为使分解后的模式保持原模式所满足的特性，一般要求分解具有无损连接性与保持函数依赖性。

<!--more-->

## 无损连接
无损连接是指分解后的关系实例可以通过自然连接运算恢复到原关系模式。
关系模式R(U，F)的一个分解，ρ={R1<U1,F1>,R2<U2,F2>}具有无损连接的充分必要条件是：

U1∩U2→U1-U2 €F+ 或U1∩U2→U2 -U1€F+
**无损连接的测试方法：**
ρ={R1<U1,F1>,R2<U2,F2>,...,Rk<Uk,Fk>}是关系模式R<U,F>的一个分解，U={A1,A2,...,An}，F={FD1,FD2,...,FDp}，并设F是一个最小依赖集，记FDi为Xi→Alj，其步骤如下：
1） 建立一张n列k行的表，每一列对应一个属性，每一行对应分解中的一个关系模式。若属性Aj Ui，则在j列i行上真上aj，否则填上bij；
2）对于每一个FDi做如下操作：找到Xi所对应的列中具有相同符号的那些行。考察这些行中li列的元素，若其中有aj，则全部改为aj，否则全部改为bmli，m是这些行的行号最小值。如果在某次更改后，有一行成为：a1,a2,...,an，则算法终止。且分解ρ具有无损连接性，否则不具有无损连接性。对F中p个FD逐一进行一次这样的处理，称为对F的一次扫描。
3） 比较扫描前后，表有无变化，如有变化，则返回第2 步，否则算法终止。如果发生循环，那么前次扫描至少应使该表减少一个符号，表中符号有限，因此，循环必然终止。
==注意：算法的终止条件有两个，第一个是出现某行全为a,第二个是扫描前后的表没有变化。若不满足这两点就继续循环扫描。==
**举例**：已知R<U,F>，U={A,B,C,D,E}，F={A→C,B→C,C→D,DE→C,CE→A}，R的一个分解为R1(AD)，R2(AB)，R3(BE)，R4(CDE)，R5(AE)，判断这个分解是否具有无损连接性。
① 构造一个初始的二维表，若“属性”属于“模式”中的属性，则填aj，否则填bij
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504103928551.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70)


② 根据A→C，对上表进行处理，由于属性列A上第1、2、5行相同均为a1，所以将属性列C上的b13、b23、b53改为同一个符号b13（取行号最小值）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504104001318.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70)


③ 根据B→C，对上表进行处理，由于属性列B上第2、3行相同均为a2，所以将属性列C上的b13、b33改为同一个符号b13（取行号最小值）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020050410403897.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70)



④ 根据C→D，对上表进行处理，由于属性列C上第1、2、3、5行相同均为b13，所以将属性列D上的值均改为同一个符号a4。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504104054938.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70)



⑤ 根据DE→C，对上表进行处理，由于属性列DE上第3、4、5行相同均为a4a5，所以将属性列C上的值均改为同一个符号a3。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504104114734.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70)



⑥ 根据CE→A，对上表进行处理，由于属性列CE上第3、4、5行相同均为a3a5，所以将属性列A上的值均改为同一个符号a1。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504104136784.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc0NDE5Mg==,size_16,color_FFFFFF,t_70)
⑦ 通过上述的修改，使第三行成为a1a2a3a4a5，则算法终止。且分解具有无损连接性。
## 保持函数依赖
保持函数依赖的定义:设p=(R1,R2..Rn)是R的一个分解，F是R上的FD集，如果满足πR1(F1)∪πR2(F2)∪πR3(F3).....∪πRn(Fn)与F等价，则表明p是保持函数依赖的分解。
**举例：**
给定关系模式R<U,F>,其中：U={A,B,C,D},F={A->B,B->C,C->D,D->A}
判断关系模式R的分解ρ={AB，BC,CD}是否具有依赖保持性。
解：
πAB(F)={A->B,B->A}
πBC(F)={B->C,C->B}
πCD(F)={C->D,D->C}
πAB(F)∪πBC(F)∪πCD(F)={A->B,B->A,B->C,C->B,C->D,D->C}
可以看出，A->B,B->C,C->D均已保持，且D的闭包为ABCD包含了A,即D->A也得到了保持，该分解ρ具有依赖保持性。
# 模式分解的算法
## 3NF的保持依赖性分解算法
==算法1：分解为3NF,且保持依赖保持性。==
算法输入：关系模式R与R的最小依赖集Fm;
算法输出：R的一个分解ρ=(R1,R2..Rn),Ri为3NF，且ρ具有依赖保持性。
具体步骤如下：
①如果Fm中有一个依赖X->A，满足XA=R,即输出ρ={R},转步骤④；
②如果R中某些属性与F中所有依赖的左部与右部都无关，则将他们构成一个关系模式，从R中分出去。
③对于Fm中的每一个Xi->Ai,都构成一个关系子模式Ri=XiAi;
④停止分解，输出ρ；
**举例：**
设有关系模式R<U,F>，其中U={C,T,H,R,S,G},F={CS->G,C->T,TH->R,HR->C,HS->R},保持函数依赖分解为3NF。
1）不存在有一个依赖X->A，满足XA=R，不满足算法1中条件①；
2）不存在某些属性与F中所有依赖的左部与右部都无关，不满足条件④；
3）对于F中的每一个Xi->Ai,都构成一个关系子模式，得到R1=CSG,R2=CT，R3=THR,R4=HRC,R5=HSR;
4) ρ={CSg,CT,THR,HRC,HSR};
## 3NF的无损连接与依赖保持分解
==算法2：分解为3NF,使他既具有无损连接性，又具有依赖保持性；==
输入：关系模式R，最小依赖集Fm;
输出：R的一个分解ρ=(R1,R2..Rn),Ri为3NF，具有无损连接性且ρ具有依赖保持性。
具体如下：
①根据算法1得到具有的依赖保持性分解ρ；
②检测ρ是否具有无损连接线，若是转④，否则继续③；
③求出R的候选码X,令ρ=ρ∪{X};
④停止分解，输出ρ；
**举例：**
设有关系模式R<U,F>，其中U={C,T,H,R,S,G},F={CS->G,C->T,TH->R,HR->C,HS->R},保持函数依赖分解且具有无损连接性的3NF。
1）根据算法1得到具有的依赖保持性分解ρ={CSg,CT,THR,HRC,HSR}；
2 ) 检测ρ是否具有无损连接性，如下表：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504115220804.png)
3）不执行；
4）输出ρ；
## BCNF的无损连接分解
==算法3：把关系模式分解为无损连接的BCNF==
输入：关系模式R，最小依赖集Fm;
输出：R的一个分解ρ=(R1,R2..Rn),Ri为BCNF，具有无损连接性。
具体如下：
①令ρ=（R）；
②如果ρ中所有模式都是BCNF，若是转④,否则继续③；
③若其中一个关系模式S不是BCNF,则S中必能找到一个函数依赖X->A，X不包含S的候选码，且A不属于X,设S1=XA,S2=S-A,用分解{S1,S2}代替S,转②；
④停止分解，输出ρ；
**举例：**
设有关系模式R<U,F>其中：U={C,T,H,R,S,G};F={CS->G,C->T,TH->R,HR->C,HS->R},将其分解为无损连接的BCNF.
解：求得R的候选码为HS,
1）令ρ={CTHRSG}；
2）ρ中模式不是BCNF(左部依赖并不是都包含HS)，继续③；
3）在CS->G中，CS不包含HS,不满足BCNF,将ρ分解为CSG与CTHRS，转②。
ρ中模式为{CSG，CTHRS}，关系CSG依赖为CS->G,满足bcnf，关系CTHRS依赖为C->T,TH->R,HR->C,HS->R(左部依赖并不是都包含候选码HS)，不满足bcnf，继续分解关系CTHRS，关系CTHRS中选择C->T，C不包含关系CTHRS的候选码HS,同理分解为CT与CHRS,转②。
ρ中模式为{CSG，CT，CHRS}，关系CT依赖为C->T,满足bcnf，关系CHRS依赖为CH->R,HR->C,HS->R(左部依赖并不是都包含候选码HS),不满足bcnf，继续分解关系CHRS，关系CHRS中选择CH->R，CH不包含关系CHRS的候选码HS,同理分解为CHR与CHS,转②。
ρ中模式为{CSG，CT，CHR,CHS}，关系CHR依赖为CH->R,HR->C,满足bcnf，关系CHS依赖为HS->C满足bcnf转④。
④停止分解，输出ρ={CSG，CT，CHR,CHS}；
