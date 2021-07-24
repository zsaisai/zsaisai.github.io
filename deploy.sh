#!/bin/bash
dt="Post at "`date "+%Y-%m-%d %H:%M:%S"`

hexo clean
git init

git add .

git commit -am "update blog: $dt"

git remote add origin git@github.com:zsaisai/zsaisai.github.io.git

git push -f -u origin master

