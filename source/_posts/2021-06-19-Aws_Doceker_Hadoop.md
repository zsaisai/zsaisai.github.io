---
title: AWS-Docker-Hadoop命令总结
date: 2021-06-19 11:43:45
tags: [AWS,Docker,Hadoop]
categories: 常用命令

top:
---

# Aws

用命令行管理aws s3
**管理存储桶**

```bash
创建桶;
$ aws s3 mb s3://bucket-name
删除桶：
$ aws s3 rb s3://bucket-name
删除非空桶：
$ aws s3 rb s3://bucket-name --force
列出存储桶
$ aws s3 ls
列出存储桶中所有的对象和文件夹
$ aws s3 ls s3://bucket-name
列出桶中  bucket-name/ MyFolder 中的对象
$ aws s3 ls s3://bucket-name/MyFolder
```
<!-- more -->

**管理对象**
命令包括 aws s3 cp、aws s3 ls、aws s3 mv、aws s3 rm 和 sync。cp、ls、mv 和 rm 命令的用法与它们在 Unix 中的对应命令相同。

```bash
// 将当前目录里的 MyFile.txt文件拷贝到 s3://my-bucket/MyFolder
$ aws s3 cp MyFile.txt s3://my-bucket/MyFolder/

// 将s3://my-bucket/MyFolder所有 .jpg 的文件移到 ./MyDirectory
$ aws s3 mv s3://my-bucket/MyFolder ./MyDirectory --exclude '*' --include '*.jpg' --recursive

// 列出  my-bucket的所有内容
$ aws s3 ls s3://my-bucket

// 列出my-bucket中MyFolder的所有内容
$ aws s3 ls s3://my-bucket/MyFolder

// 删除 s3://my-bucket/MyFolder/MyFile.txt
$ aws s3 rm s3://my-bucket/MyFolder/MyFile.txt

// 删除 s3://my-bucket/MyFolder 和它的所有内容
$ aws s3 rm s3://my-bucket/MyFolder --recursive

```
当 --recursive 选项与 cp、mv 或 rm 一起用于目录/文件夹时，命令会遍历目录树，包括所有子目录

**sync命令**
sync 命令的形式如下。可能的源-目标组合有：

本地文件系统到 Amazon S3

Amazon S3 到本地文件系统


```bash
Amazon S3 到 Amazon S3

$ aws s3 sync <source> <target> [--options]

本地文件系统到S3
 
$ aws s3 sync 本地目录/. s3://my-bucket/目录
```



# Doceker-image

```bash
在这里插入代码片
```

```bash
查看镜像列表

docker images
```
参数说明：
REPOSITORY ： 镜像名称
TAG        ： 镜像标签
IMAGE ID   ： 镜像ID（唯一）
CREATED    ： 镜像的创建日期（非获取镜像日期）
SIZE       ： 镜像大小

```bash
拉取镜像-从中央仓库中下载到本地

docker pull 镜像名称
```

```bash
删除镜像
对应镜像的容器需要停止

docker rmi 镜像ID
```

```bash
查看所有容器

docker ps -a
```

```bash
查看正在运行的容器

docker ps
```

```bash
创建容器命令

docker run

#参数 
-i          ：表示运行容器
-t          ：表示容器启动后，会进入其命令行。加入以上两个参数后，容器创建就会进入容器，即分配一个伪终端
-name       ：为创建的容器命名
-v          ：表示目录映射关系（前面是宿主机目录，后者是映射到宿主机的目录），可以使用多个 -v
              做多个目录或文件映射。（最好做目录映射，方便操作）
-d          ：在run后面加上-d 参数，则会创建一个守护式容器在后台运行（这样创建容器后不会自动登录容器，如果只加 -i -t 两个参数，创建后就会自动进入容器）
-p          ：表示端口映射，前者是宿主机端口，后者是容器内的映射端口。可以使用多个 -p 做多个端口映射
```

```bash
交互式方式创建容器（退出容器，容器自动停止）

docker run -it --name=容器名称 镜像名称:标签 /bin/bash
```

```bash
守护式方式创建容器

docker run  -di --name=容器名称 镜像名称:标签
```

```bash
登录守护式方式创建容器（退出容器，容器不会停止）

docker exec -it 容器名称（或容器ID） /bin/bash
```

```bash
停止与启动容器

docker start 容器名称（或容器ID）
docker stop  容器名称（或容器ID）
```

文件拷贝
```bash
# 将文件拷贝到容器内
docker cp 需要拷贝的文件或目录 容器名称：容器目录
# 将文件从容器内拷贝出来
docker cp 容器名称：容器目录 需要拷贝的文件或目录
```

```bash
docker备份和迁移
容器保存为镜像

docker commit 容器名称 镜像名称
```

```bash
镜像备份与恢复

# 镜像备份
docker save -o *.tar 镜像名称
 
# 镜像恢复
docker load  -i *.tar
```

```bash
#退出
exit

#关闭
docker stop mycentos

#重启
docker start mycentos
```

# Hadoop
**cat**
cat查看文件里面的内容
使用方法：hadoop fs -cat URI [URI …]
将路径指定文件的内容输出到stdout。
示例：

```bash
hadoop fs -cat hdfs://host1:port1/file1 hdfs://host2:port2/file2
hadoop fs -cat file:///file3 /user/hadoop/file4
```
**chmod**
使用方法：hadoop fs -chmod [-R] <MODE[,MODE]... | OCTALMODE> URI [URI …]
改变文件的权限。使用-R将使改变在目录结构下递归进行。命令的使用者必须是文件的所有者或者超级用户。

**chown**
使用方法：hadoop fs -chown [-R] [OWNER][:[GROUP]] URI [URI ]
改变文件的拥有者。使用-R将使改变在目录结构下递归进行。命令的使用者必须是超级用户。

**cp**
使用方法：hadoop fs -cp URI [URI …] <dest>
将文件从源路径复制到目标路径。这个命令允许有多个源路径，此时目标路径必须是一个目录。
示例：
```bash
hadoop fs -cp /user/hadoop/file1 /user/hadoop/file2
hadoop fs -cp /user/hadoop/file1 /user/hadoop/file2 /user/hadoop/dir
```
**du**
使用方法：hadoop fs -du URI [URI …]
显示目录中所有文件的大小，或者当只指定一个文件时，显示此文件的大小。
df查看磁盘空间大小：hadoop fs -df -h /
du查看文件的大小或者目录的大小（ -h -s 不同量级）
示例：
```bash
hadoop fs -du -h -s /user/hadoop/dir1 /user/hadoop/file1
```
**get**
使用方法：hadoop fs -get [-ignorecrc] [-crc] <src> <localdst>
复制文件到本地文件系统。可用-ignorecrc选项复制CRC校验失败的文件。使用-crc选项复制文件以及CRC信息。
示例：

```bash
hadoop fs -get /user/hadoop/file localfile
hadoop fs -get hdfs://host:port/user/hadoop/file localfile
```

**put**
使用方法：hadoop fs -put <localsrc> ... <dst>
从本地文件系统中复制单个或多个源路径到目标文件系统。也支持从标准输入中读取输入写入目标文件系统。

```bash
hadoop fs -put localfile hdfs://host:port/hadoop/hadoopfile
```
**getmerge**
使用方法：hadoop fs -getmerge <src> <localdst> [addnl]
接受一个源目录和一个目标文件作为输入，并且将源目录中所有的文件连接成本地目标文件。addnl是可选的，用于指定在每个文件结尾添加一个换行符。

**ls**
使用方法：hadoop fs -ls <args>
如果是文件，则按照如下格式返回文件信息：
文件名 <副本数> 文件大小 修改日期 修改时间 权限 用户ID 组ID
如果是目录，则返回它直接子文件的一个列表，就像在Unix中一样。目录返回列表的信息如下：
目录名 <dir> 修改日期 修改时间 权限 用户ID 组ID
示例：
```bash
hadoop fs -ls /user/hadoop/file1 
```
**rm**
使用方法：hadoop fs -rm URI [URI …]
删除指定的文件。
示例：
```bash
hadoop fs -rm hdfs://host:port/file /user/hadoop/emptydir
```
**rm -r**
使用方法：hadoop fs -rmr URI [URI …]
delete的递归版本。删除文件夹。
示例：

```bash
hadoop fs -rm -r /user/hadoop/dir
hadoop fs -rm -r hdfs://host:port/user/hadoop/dir
```

**mkdir**
使用方法：hadoop fs -mkdir <paths>

接受路径指定的uri作为参数，创建这些目录。其行为类似于Unix的mkdir -p，它会创建路径中的各级父目录。

示例：

```bash
hadoop fs -mkdir /user/hadoop/dir1 /user/hadoop/dir2
hadoop fs -mkdir hdfs://host1:port1/user/hadoop/dir hdfs://host2:port2/user/hadoop/dir
```
**mv**
使用方法：hadoop fs -mv URI [URI …] <dest>

将文件从源路径移动到目标路径。这个命令允许有多个源路径，此时目标路径必须是一个目录。不允许在不同的文件系统间移动文件。
示例：
```bash
hadoop fs -mv /user/hadoop/file1 /user/hadoop/file2 
```


