# Claude Code + DeepSeek V4 部署笔记

部署时间：2026-05-25  
部署人：朋友协助本人部署，随后记录整理  
环境：Windows 11 + Node.js 18 + CC Switch + DeepSeek V4

---

## 1. 安装Node.js（Claude Code的运行环境）
- Windows 10/11
- 安装 Node.js 18+（Claude Code 依赖）
  - 官网：(https://nodejs.org) → 选windows安装程序

- 打开 CMD 下载claude code：
```bash
npm install -g @anthropic-ai/claude-code
```

## 2. 获取 DeepSeek V4 API Key（只做一次）
- 浏览器打开：platform.deepseek.com
- 注册/登录 → 充值（最低几块钱即可，个人推荐首次使用10-20元即可）
- 左侧点 API Keys → 创建 Key
- 名字：如 claude-code
- 生成后立即复制保存API Keys!（只显示一次）

## 3. 安装 CC Switch（只做一次）
1. 打开：github.com/farion1231/cc-switch/releases
2. 下载最新版 Windows .msi（或便携 zip）
3. 双击安装 → 一路下一步
4. 启动 CC Switch（桌面/开始菜单）

## 4. CC Switch 配置 DeepSeek V4（关键，一次配置长期用）
1. CC Switch 顶部选 Claude Code 模式
2. 右上角点加号（Add Provider）
3. 添加新供应商中选 DeepSeek

填写：
- Name：DeepSeek-V4（随便写）
- API Key：粘贴刚才在Deepseek上复制的API Keys
- 请求地址：https://api.deepseek.com/anthropic
- 模型映射中实际请求模型均填入DeepSeek-v4-pro（DeepSeek-v4-pro[1m]无法填入）
- 默认兜底模型：DeepSeek-v4-pro[1m]
- 配置json：选择最大强度思考
  
## 5. 环境验证(在CMD中输入)
- node --version      # 应显示 v18+ 或 v20+
- npm --version       # 应显示 9+
- where claude        # 应显示路径
- claude --version    # 应显示版本号

## 6. 日常使用
- 确保 CC Switch 开着并已启用 DeepSeek（最小化即可）
- 新开 CMD/PowerShell，输入：claude
- 此时 Claude Code 自动走 CC Switch 本地代理 → DeepSeek V4
- 可以正常编辑文件、跑命令、Git 操作

## 7. 关键信息备忘
| 项目 |  内容  |
|------|------|
| Node.js | 让电脑能够运行JavaScript程序（在这里是Claude Code）|
| Claude Code | 使用JavaScript/TypeScript写的命令行工具，负责交互式对话，读文件，执行命令 |
| CC Switch | 本地代理程序，把Claude Code的网络请求改地址+改模型名，转发给DeepSeek |
| DeepSeek Base URL | API的接口地址，https://api.deepseek.com/anthropic（CC Switch 预设自动填 ） |
| DeepSeek | 云端AI服务，实际运行的大模型，接受请求，生成回复 |

> 生效标志：**Claude 启动后问"你是什么模型"，回答是 DeepSeek

## 8. 问题及解决

| 问题 | 解决 |
|-----|------|
| `npm install` 下载失败 | 检查 Node.js 版本、切换网络 |
| CC Switch 未连接 | 检查网络与 API Key 有效性 |
| 模型回复质量差 | 确认启用的是 `deepseek-v4-pro[1m]` |

---

## 9. 典型案例：claude 命令突然失效

&gt; 记录时间：2026-05-28  
&gt; 现象：输入 `claude` 提示命令找不到，但之前正常使用

### 故障时间线

1. 最初： `claude.cmd` 正常，`claude` 命令可用 
2. 发现问题： `AppData\Roaming\npm\claude.cmd` 丢失（原因不明，可能被杀毒软件清理或误删）
3. 连带问题：`C:\Users\用户名\claude` 空文件出现，导致where claude指令能找到文件但无法执行（`npx` 残留或误操作生成）
4. 故障时：`where claude` 优先找到空文件，`claude` 命令失效 
5. 修复后： 删除空文件，重装生成 `claude.cmd`，恢复正常 

### 排查命令

| 命令 | 作用 | 正常与异常 |
|------|-----|-------------|
| where claude               |   查看所有claude路径 | 正常：2条（claude + claude.cmd）；异常：多条或找不到 |
| dir C:\Users\用户名\claude  |    检查用户目录下是否有异常空文件 | 正常：找不到；异常：有个0字节空文件 |
| dir "%APPDATA%\npm\claude*"|    检查 npm 全局目录下的启动脚本 | 正常：3个文件(claude,claude.cmd,claude.ps1)；异常：找不到或缺失 |

### 根因分析

> 核心原因：claude.cmd文件缺失
`where claude` 在 Windows 下按 **PATH 环境变量顺序** 查找可执行文件。  
`C:\Users\用户名\claude` 这个空文件恰好排在 `AppData\Roaming\npm\claude.cmd` 之前被找到，导致系统认为"找到了 claude"，但文件是空的，无法执行。

### 修复步骤

1. 卸载旧版
```
npm uninstall -g @anthropic-ai/claude-code
```
2. 清理残留（包括空文件和损坏的脚本）
```
del "C:\Users\www\claude" 2>nul
rmdir /s /q "%APPDATA%\npm\node_modules\@anthropic-ai" 2>nul
del "%APPDATA%\npm\claude.cmd" 2>nul
del "%APPDATA%\npm\claude" 2>nul
```
>逐条运行，没有返回是正常的
3. 重新安装 (回到第一步)
```
npm install -g @anthropic-ai/claude-code
```

###  验证
| 输入指令 | 作用 | 期望结果 |
|---------|------|---------|
| where claude   |  查找所有识别到的claude可执行文件，确认没有旧残留 | 应该只显示 AppData\Roaming\npm\claude 和 claude.cmd，没有多余路径 |
| dir "%APPDATA%\npm\claude*"    |  列出npm（Node.js自带的包管理工具）下所有claude开头的文件，确认生成启动脚本 |  显示 claude、claude.cmd、claude.ps1 三个文件 |
| claude --version   | 打印claude code版本号，确认程序正常运行 | 显示 2.1.153 (Claude Code) 或更高版本|
