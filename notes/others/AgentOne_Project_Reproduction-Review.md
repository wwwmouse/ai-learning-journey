# AgentOne 本地部署完整复盘：从 Agent 理论到工程实践

## 一、任务背景

在完成 Agent 学习路线 Stage0-2 后，复现企业级 AI Agent 平台 AgentOne。

本次任务的核心目标不是完全读懂源码，而是体验：

> 从 AI Agent 理论学习，到真实工程系统部署的转变。

最终完成了：

源码 → 环境配置 → Docker 构建 → 服务启动 → 配置调试 → 完整运行

这一整套流程。

------------------------------------------------------------------------

# 二、最终运行架构

AgentOne 是一个多服务企业级应用，不是单个 Python 文件。

整体结构：

浏览器

↓

Frontend（Vue + Nginx）

↓

Backend（Spring Boot）

↓

PostgreSQL + pgvector / Redis

组件：

  服务                    技术                  作用
  ----------------------- --------------------- ----------------------
  agentone-frontend       Vue + Nginx           用户界面
  agentone-backend        Spring Boot Java      核心业务逻辑
  PostgreSQL + pgvector   数据库 + 向量数据库   数据存储、知识库检索
  Redis                   缓存                  提升访问效率

------------------------------------------------------------------------

# 三、完整部署流程

## 1. 安装 Docker 环境

最初误认为 Docker 类似 numpy。

实际：

-   numpy：Python 库
-   Docker：运行环境管理平台

Docker 解决的问题：

不同项目需要不同版本：

-   Java
-   Node
-   Maven
-   Redis
-   PostgreSQL

Docker 可以通过镜像和容器快速复现环境。

Windows 下运行 Docker：

Windows

↓

WSL2

↓

Linux 环境

↓

Docker Engine

↓

Container

------------------------------------------------------------------------

# 四、遇到的问题与解决方案

## 问题1：Docker Engine 无法启动

现象：

Docker Desktop：

Engine stopped

Virtualization support not detected

原因：

Windows 虚拟化开启，但是 WSL2 环境没有准备。

解决：

安装：

wsl --install Ubuntu

重启后确认：

wsl -l -v

确保 Linux 发行版使用 WSL2。

学习：

Docker Desktop 并不是简单软件，而是依赖 Linux 环境运行容器。

------------------------------------------------------------------------

# 问题2：docker compose 构建时无法拉取镜像

错误：

failed to fetch oauth token

原因：

Docker Hub 网络访问失败。

Docker 需要下载：

-   node
-   maven
-   java
-   redis
-   postgres

解决：

配置 Docker 镜像源。

学习：

Docker 构建依赖多个外部资源：

Docker 镜像源

npm registry

Linux apt 源

任何网络环节失败都会导致构建失败。

------------------------------------------------------------------------

# 问题3：npm install 网络错误

错误：

npm error ECONNRESET

位置：

frontend Dockerfile：

RUN npm install

原因：

前端依赖需要从 npm 下载，但网络不稳定。

解决：

调整 npm registry。

学习：

前端工程不仅包含代码，还依赖完整依赖树。

------------------------------------------------------------------------

# 问题4：apt-get 软件源失败

错误：

apt-get update

404 Not Found

unexpected EOF

位置：

backend Dockerfile：

安装 curl。

原因：

Ubuntu 软件源访问异常。

解决：

重新构建，最终成功。

注意：

这里不是修改业务代码，而是部署环境适配。

------------------------------------------------------------------------

# 问题5：后端容器启动失败

现象：

agentone-backend

Restarting

查看：

docker logs agentone-backend

发现：

JWT 密钥未配置。

原因：

docker-compose 需要环境变量：

JWT_SECRET

但是：

.env 文件不存在。

解决：

创建：

.env

添加：

JWT_SECRET=随机生成字符串

然后：

docker compose down

docker compose up -d

------------------------------------------------------------------------

# 五、最终成功状态

执行：

docker compose ps

得到：

agentone-backend Up (healthy)

agentone-frontend Up

postgres Up (healthy)

redis Up (healthy)

说明整个系统启动成功。

------------------------------------------------------------------------

# 六、从这次部署获得的工程认知

## 1. 软件不等于代码

以前：

项目 = 源代码

现在：

项目 =

代码

-   

运行环境

-   

依赖

-   

配置

-   

数据库

-   

网络

-   

部署方式

------------------------------------------------------------------------

## 2. Agent 产品不只是调用大模型

简单 AI Demo：

Vue

↓

DeepSeek API

↓

返回结果

企业 Agent 平台：

用户系统

↓

认证

↓

Agent 管理

↓

知识库

↓

向量检索

↓

工具调用

↓

模型

↓

结果管理

------------------------------------------------------------------------

## 3. Debug 能力比记命令更重要

通用排查流程：

第一步：

判断失败阶段：

-   构建失败？
-   启动失败？
-   运行失败？

第二步：

定位服务：

-   frontend
-   backend
-   database
-   cache

第三步：

查看日志：

docker logs

第四步：

区分：

-   代码问题
-   环境问题
-   网络问题
-   配置问题

------------------------------------------------------------------------

# 七、后续学习方向

推荐路线：

AgentOne 源码阅读

↓

RAG 模块

↓

Agent 执行流程

↓

Tool / MCP

↓

自己实现简化 Agent 框架

这次部署的最大价值：

从"会调用模型"

走向：

"能够构建 AI 应用系统"。
