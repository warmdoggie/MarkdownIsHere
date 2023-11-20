# Git操作
## 一.命令
### 创建仓库
#### 1.git clone 'github上源码的链接'
    复制远程数据库
#### 2.git init（在任意一个文件下右键'open git bash here'
    创建全新的数据库，产生一个隐藏文件夹名为'.git',里面是仓库,剩下的是工作区
### 工作区(Workspace)和索引区(Index/stage)
#### 3.git status 
    查看当前项目状态
#### 4.git add 'file1' 'file2'...
    文件名之间用空格分开
#### 5.git add .
    将当前目录下所有的文件添加到索引区中
#### 6.git commit -m "本次提交的说明"
#### 7.git log
    查看历史提交记录
### 本地git与github进行关联
#### 1. ls -al ~/.ssh
    查看是否已经存在密钥
#### 2.ssh-keygen -t rsa -C '密钥说明'
    -t 表示密钥类型 -C密钥的描述
    (这里我在说明里写的是邮箱)
#### test a little change
#### again
试试2