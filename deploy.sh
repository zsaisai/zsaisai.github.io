#!/bin/bash
dt="Post at "`date "+%Y-%m-%d %H:%M:%S"`

hexo clean

git add .

git commit -am "$dt"

git push -f -u origin master

