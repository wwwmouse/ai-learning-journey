# Git 关联 GitHub 配置笔记

> 记录从本地仓库关联 GitHub 的全过程，以备后续换电脑或新建项目时参考。
---

## 前置条件
- 已安装 [Git](https://git-scm.com/downloads)
- 已注册 [GitHub](https://github.com) 账号
- 已安装 VS Code

---

## 一、GitHub 上已有仓库，本地从零开始

### 场景
GitHub 上已创建仓库，本地文件夹需要关联仓库

### 步骤

#### 1. 进入本地仓库根目录

```bash
cd E:\VScode
```

#### 2. 初始化本地 Git 仓库

```bash
git init
```

> 这会创建一个隐藏的 `.git` 文件夹，让 Git 开始管理这个目录。
> 放松点，文件名变绿是正常的^^

#### 3. 关联远程仓库

```bash
git remote add origin https://github.com/你的用户名/关联仓库名.git
```

> `origin` 是远程仓库的别名，可以自定义，但 `origin` 是约定俗成的名字。

#### 4. 设置分支名

```bash
git branch -M main
```
> 将默认分支命名为 `main`（GitHub 的标准）。

#### 5. 拉取远程代码

```bash
git pull origin main
```

> 把 GitHub 上的文件同步到本地，避免冲突。

---

## 二、配置 Git 身份（首次使用必做）

Git 要求每次提交前配置用户名和邮箱，用于标识提交者。

### 全局配置（所有仓库通用）

```bash
git config --global user.name "你的用户名"
git config --global user.email "你的GitHub邮箱@example.com"
```

### 当前仓库单独配置（可选）

如果某个项目想用不同的身份：

```bash
git config user.name "新用户名"
git config user.email "新GitHub邮箱@example.com"
```

### 结束后验证配置

```bash
git config user.name
git config user.email
```
>git config user.name(email)其实就是获取当前仓库的用户名（邮箱）
---

## 三、创建 .gitignore 过滤垃圾文件

### 问题
初始化后，VS Code 源代码管理面板可能显示大量无关文件（如 Python 缓存、数据集、编辑器配置等）。

### 解决
在仓库根目录创建 `.gitignore` 文件：

```gitignore
# === IDE & Editor ===
.vscode/
.idea/
*.swp
*.swo
*~

# === Python ===
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# === Jupyter ===
.ipynb_checkpoints

# === Data & Models ===
data/
datasets/
*.tar.gz
*.zip
*.pth
*.pt
*.h5
*.onnx

# === Logs & Temp ===
*.log
*.tmp
*.bak

# === OS ===
.DS_Store
Thumbs.db
```

### 提交 .gitignore

```bash
git add .gitignore
git commit -m "chore: add gitignore"
git push origin main
```

> `.gitignore` 本身需要被 Git 追踪，所以修改后也要提交。

---

## 四、日常提交流程

### 标准流程

```
修改文件 → git add → git commit → git push
```

### 具体命令

```bash
# 1. 暂存单个文件（放入提交队列）
git add notes/vscode-git-github-setup-guide.md
# 或暂存所有变更：
git add .

# 2. 提交到本地仓库（附带说明）
git commit -m "此次修改的说明"
-m就是-message的缩写，后跟此次提交的说明文字

# 3. 推送到 GitHub
git push origin main
```

### 提交信息规范

| 前缀 | 含义 | 例子 |
|------|------|------|
| `add:` | 新增内容 | `add: numpy array basics notes` |
| `update:` | 更新现有内容 | `update: README learning roadmap` |
| `fix:` | 修正错误 | `fix: typo in README dates` |
| `chore:` | 杂项/配置 | `chore: add gitignore` |

---

## 五、VS Code 图形界面操作（替代命令行）

### 查看远程仓库

```bash
git remote -v
```

输出示例：
```
origin  https://github.com/你的用户名/你的仓库名 (fetch)
origin  https://github.com/你的用户名/你的仓库名 (push)
```

---

## 六、常见问题排查

### 1. `fatal: not a git repository`

**原因**：当前目录没有 `.git` 文件夹。

**解决**：

在根目录输入git init


### 2. `fatal: pathspec 'xxx' did not match any files`

**原因**：文件名拼写错误，或文件未被 Git 追踪。

**解决**：检查文件名大小写、空格、下划线是否一致。

### 3. `Author identity unknown`

**原因**：未初始配置 `user.name` 和 `user.email`。

**解决**：
```bash
#可以选择只配置当前仓库
git config user.name "用户名"
git config user.email "你的邮箱@example.com"

#或者直接配全局
git config --global user.name "用户名"
git config --global user.email "你的邮箱@example.com"
```
### 4. 误初始化到子文件夹

**原因**：在子文件夹执行了 `git init`。

**解决**：
```bash
# 删除错误的 .git
rm -rf .git

# 回到根目录重新初始化
cd ..  #回到上级目录
git init
```

### 5. 大量无关文件显示在源代码管理

**原因**：缺少 `.gitignore` 或规则不匹配。

**解决**：检查 `.gitignore` 中的文件夹名是否与本地完全一致（注意空格、大小写）。

### 6. add文件后提示warning

```bash
warning: in the working copy of 'notes/vscode-git-github-setup-guide.md'
LF will be replaced by CRLF the next time Git touches it 
```

**原因**：Git 检测到文件是 LF 格式，提醒下次会自动转成 CRLF。

**解决**：这是 Windows 的正常现象，无须在意^^。

---

## 七、安全原则

| 危险操作 | 后果 | 建议 |
|---------|------|------|
| `git push --force` | 覆盖远程历史，可能丢失他人提交 | **永远不要使用!!** |
| `git reset --hard` | 删除未提交的本地修改 | 使用前确认已保存 |
| 在子文件夹初始化 | 仓库结构混乱 | 始终在根目录操作 |

---

## 八、验证配置成功

```bash
# 查看当前状态
git status

# 查看提交历史
git log --oneline -5

# 查看远程关联
git remote -v

# 查看当前分支
git branch
```

---


> 记录时间：2026-05-31
> 环境：Windows 11 + VS Code + Git
