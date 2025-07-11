---
title: docker面经
date: 2025-07-10 14:46:27
tags:
---
> 面试的时候问我有没有用过docker，用是用过很多具体的也不记得，自己也没有创建过docker回答的一塌糊涂（）

# Docker的常用命令


## 镜像管理（image）

| 操作           | 命令                          |
| -------------- | ----------------------------- |
| 查看已有镜像   | `docker images`             |
| 删除镜像       | `docker rmi 镜像ID或名称`   |
| 构建镜像       | `docker build -t myimage .` |
| 从远程拉取镜像 | `docker pull ubuntu:20.04`  |
| 给镜像打标签   | `docker tag old new`        |


## 容器管理（container）

| 操作                     | 命令                                  |
| ------------------------ | ------------------------------------- |
| 查看正在运行的容器       | `docker ps`                         |
| 查看所有容器（含已停止） | `docker ps -a`                      |
| 运行容器                 | `docker run -it ubuntu /bin/bash`   |
| 容器后台运行             | `docker run -d myimage`             |
| 给容器取名               | `docker run --name mycontainer ...` |
| 停止容器                 | `docker stop 容器ID或名称`          |
| 启动/重启容器            | `docker start/restart 容器ID或名称` |
| 删除容器                 | `docker rm 容器ID或名称`            |
| 查看容器日志             | `docker logs -f 容器ID`             |
| 查看容器资源占用         | `docker stats`                      |



## 容器交互与调试

| 操作             | 命令                                    |
| ---------------- | --------------------------------------- |
| 进入容器内部     | `docker exec -it 容器ID /bin/bash`    |
| 拷贝文件到容器   | `docker cp 本地路径 容器ID:/目标路径` |
| 从容器拷贝到主机 | `docker cp 容器ID:/路径 本地路径`     |


# 常见问题

## 容器和镜像的关系

> **镜像是静态的模板，容器是镜像的运行实例。**

镜像包含了

* 操作系统层（如 Ubuntu、Alpine）
* 运行环境（Python、Node.js 等）
* 应用代码
* 配置文件、依赖包等

容器 = 镜像 + 一层可写层 + 运行时状态

* 容器运行时：
  * 会从镜像加载文件系统（只读）
  * 再叠加一个 **读写层（container layer）**
  * 容器内部修改的文件不会影响镜像

## 环境a,导出为镜像b,用容器c打开,分别需要哪些命令

```bash
# 1. 查看当前容器
docker ps

# 2. 保存容器 a 为镜像 b
docker commit a b

# 3. 用镜像 b 创建并运行新容器 c
docker run -it --name c b

# 导出tar
docker save b > b.tar
```


## 裸系统环境

> 不能直接使用 `docker commit` 需要转成docker镜像

写成 `docker file`:

```bash
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 拷贝本地代码
COPY . /app

# 安装依赖（你可以从你现有环境导出 requirements.txt）
RUN pip install -r requirements.txt

CMD ["python", "main.py"]

```

然后构建镜像,运行

```bash
docker build -t myenv:latest .

docker run -it --name mycontainer myenv
```
