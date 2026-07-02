> 面向 Git 新手，从零开始将 CuriosityPPOAgent 项目推送至 GitHub。
> 全程在 Windows 环境下操作，每步均标注 **Windows PowerShell** 命令；为兼顾跨平台复现者，同时附 **Linux Bash** 等价命令。
> 项目约定：12 次 Git 提交、MIT 协议、主分支名为 `main`。

---

## 前置准备

### 1. 安装 Git

- 访问官方下载页：`https://git-scm.com/download/win`
- 下载 64-bit Git for Windows 安装包，按默认选项安装。
- 安装完成后，验证版本。

- Windows PowerShell:
  ```powershell
  git --version
  ```

- Linux Bash:
  ```bash
  git --version
  ```

> 预期输出形如 `git version 2.45.x.windows.1`。

### 2. 配置全局用户名与邮箱

- Windows PowerShell:
  ```powershell
  git config --global user.name "你的GitHub用户名"
  git config --global user.email "你的GitHub邮箱@example.com"
  git config --global core.quotepath false   # 解决中文文件名乱码
  git config --global init.defaultBranch main # 默认分支名设为 main
  ```

- Linux Bash:
  ```bash
  git config --global user.name "你的GitHub用户名"
  git config --global user.email "你的GitHub邮箱@example.com"
  git config --global core.quotepath false
  git config --global init.defaultBranch main
  ```

### 3. 注册 GitHub 账号

- 访问 `https://github.com/signup` 注册账号（若已有账号可跳过）。
- 记住用户名与注册邮箱，后续命令中 `<你的用户名>` 替换为实际用户名。

---

## 步骤 1：初始化本地仓库

进入项目根目录 `curiosity-ppo`，初始化 Git 仓库。

- Windows PowerShell:
  ```powershell
  cd <项目根目录>
  git init
  ```

- Linux Bash:
  ```bash
  cd ~/curiosity-ppo
  git init
  ```

> 预期输出：`Initialized empty Git repository in .../curiosity-ppo/.git/`

---

## 步骤 2：添加 .gitignore 并验证

确保 `.pt`、`.onnx`、`wandb/`、`__pycache__/` 等大文件与缓存不被提交。

### 2.1 创建 .gitignore

在项目根目录创建 `.gitignore` 文件，内容至少包含：

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# 模型权重与导出文件（大文件，禁止提交）
*.pt
*.pth
*.ckpt
*.onnx
*.safetensors

# 训练日志与缓存
wandb/
runs/
logs/
*.log

# IDE
.vscode/
.idea/
*.swp

# 系统文件
.DS_Store
Thumbs.db
```

### 2.2 验证 .gitignore 生效

执行后检查「将被追踪的文件」中是否仍包含 `.pt` / `.onnx` / `wandb` 目录。

- Windows PowerShell:
  ```powershell
  git add .
  git status --short | Select-String -Pattern "\.pt|\.onnx|wandb|__pycache__"
  ```

- Linux Bash:
  ```bash
  git add .
  git status --short | grep -E "\.pt|\.onnx|wandb|__pycache__"
  ```

> 若输出为空，说明大文件已被正确忽略。若有残留，需补充 `.gitignore` 规则后执行 `git rm -r --cached <文件>` 清除缓存。

- 额外校验：查看完整待提交列表确认无误。
  - Windows PowerShell:
    ```powershell
    git status --short
    ```
  - Linux Bash:
    ```bash
    git status --short
    ```

---

## 步骤 3：首次提交

将所有文件加入暂存区并提交，作为 12 次提交链的起点（实际项目应分多次逻辑提交，此处为首次提交示例）。

- Windows PowerShell:
  ```powershell
  git add .
  git commit -m "feat: 初始化 CuriosityPPOAgent 项目结构与核心代码"
  ```

- Linux Bash:
  ```bash
  git add .
  git commit -m "feat: 初始化 CuriosityPPOAgent 项目结构与核心代码"
  ```

> 提交后用 `git log` 查看提交记录，确认 12 次提交完整。
> - Windows PowerShell:
>   ```powershell
>   git log --oneline
>   ```
> - Linux Bash:
>   ```bash
>   git log --oneline
>   ```

---

## 步骤 4：在 GitHub 创建新仓库

1. 浏览器登录 GitHub，点击右上角「+」→「New repository」。
2. Repository name 填：`curiosity-ppo`
3. Description 填（可选，见 `./assets/repo_cover_desc.txt` 中的 About 描述）。
4. 可见性选 **Public**（开源）。
5. **不要勾选**「Add a README file」、**不要勾选**「Add .gitignore」、**不要勾选**「Choose a license」——避免与本地仓库冲突。
6. 点击「Create repository」。

> 创建完成后，GitHub 会给出仓库地址，形如：
> - HTTPS: `https://github.com/<你的用户名>/curiosity-ppo.git`
> - SSH: `git@github.com:<你的用户名>/curiosity-ppo.git`

---

## 步骤 5：关联远程仓库

将本地仓库与 GitHub 远程仓库建立关联。

- Windows PowerShell（HTTPS 方式）:
  ```powershell
  git remote add origin https://github.com/<你的用户名>/curiosity-ppo.git
  git remote -v
  ```

- Linux Bash（HTTPS 方式）:
  ```bash
  git remote add origin https://github.com/<你的用户名>/curiosity-ppo.git
  git remote -v
  ```

- 若使用 SSH 方式（需先配置 SSH Key）:
  - Windows PowerShell:
    ```powershell
    git remote add origin git@github.com:<你的用户名>/curiosity-ppo.git
    ```
  - Linux Bash:
    ```bash
    git remote add origin git@github.com:<你的用户名>/curiosity-ppo.git
    ```

> 预期 `git remote -v` 输出两行（fetch 与 push），地址一致即关联成功。

---

## 步骤 6：推送到 GitHub

首次推送需用 `-u` 绑定上游分支。

### 6.1 确认当前分支名为 main

- Windows PowerShell:
  ```powershell
  git branch -M main
  ```

- Linux Bash:
  ```bash
  git branch -M main
  ```

> `git branch -M main` 将当前分支强制重命名为 `main`，避免本地为 `master` 与远程 `main` 不一致。

### 6.2 执行推送

- Windows PowerShell:
  ```powershell
  git push -u origin main
  ```

- Linux Bash:
  ```bash
  git push -u origin main
  ```

> 推送过程中会弹出 GitHub 认证窗口（或要求输入 Personal Access Token），按提示完成认证。
> 推送成功后，刷新 GitHub 仓库页面即可看到代码。

---

## 步骤 7：设置 Topics 和 About 描述

1. 进入 GitHub 仓库主页。
2. 点击右上角「About」区域齿轮图标「Edit repository details」。
3. Description：粘贴 `./assets/repo_cover_desc.txt` 中的 GitHub About 栏描述（350 字符以内）。
4. Topics：逐个输入标签（见 `./docs/repo_tags.md`），共 17 个：
   ```
   pytorch reinforcement-learning ppo intrinsic-motivation curiosity-driven icm rnd ngu sparse-reward exploration deep-reinforcement-learning game-ai python onnx vite-react research-project undergraduate
   ```
5. 点击「Save changes」。

---

## 步骤 8：添加 LICENSE 文件（如尚未添加）

本项目采用 **MIT** 协议。

### 8.1 检查是否已有 LICENSE

- Windows PowerShell:
  ```powershell
  Test-Path .\LICENSE
  ```

- Linux Bash:
  ```bash
  test -f LICENSE && echo "存在" || echo "不存在"
  ```

### 8.2 若不存在，创建 MIT LICENSE

在项目根目录创建 `LICENSE` 文件，内容为标准 MIT 协议文本，年份填当前年份，版权人填作者名。

- Windows PowerShell（创建并提交）:
  ```powershell
  # 手动创建 LICENSE 文件后执行：
  git add LICENSE
  git commit -m "docs: 添加 MIT 开源协议 LICENSE 文件"
  git push
  ```

- Linux Bash:
  ```bash
  # 手动创建 LICENSE 文件后执行：
  git add LICENSE
  git commit -m "docs: 添加 MIT 开源协议 LICENSE 文件"
  git push
  ```

> 也可通过 GitHub 网页端「Add file」→「Create new file」，文件名输入 `LICENSE`，右侧选择「MIT License」模板自动生成，再提交。

---

## 常见问题排查

### 问题 1：大文件 push 失败（超过 GitHub 100MB 限制）

**现象**：`push` 报错 `file larger than 100MB` 或 `pre-receive hook declined`。

**原因**：误提交了 `.pt` 模型权重或 `wandb/` 日志等大文件。

**解决**：

1. 确认 `.gitignore` 已包含 `*.pt`、`*.onnx`、`wandb/` 等规则。
2. 从 Git 历史中移除已追踪的大文件。
   - Windows PowerShell:
     ```powershell
     git rm -r --cached wandb
     git rm --cached *.pt
     git commit -m "chore: 移除误提交的大文件"
     git push
     ```
   - Linux Bash:
     ```bash
     git rm -r --cached wandb
     git rm --cached *.pt
     git commit -m "chore: 移除误提交的大文件"
     git push
     ```
3. 若大文件已进入历史提交，需用 `git filter-repo` 或 BFG 清理历史（高级操作，谨慎执行）。

### 问题 2：中文文件名乱码

**现象**：`git status` 中中文文件名显示为 `\345\276\247\345...` 转义形式。

**解决**：

- Windows PowerShell:
  ```powershell
  git config --global core.quotepath false
  ```
- Linux Bash:
  ```bash
  git config --global core.quotepath false
  ```

> 设置后重新执行 `git status`，中文文件名应正常显示。

### 问题 3：认证失败（Authentication failed）

**现象**：`push` 报错 `Authentication failed` 或 `support for password authentication was removed`。

**原因**：GitHub 已于 2021 年 8 月取消密码认证，需使用 **Personal Access Token (PAT)** 或 **SSH Key**。

**解决（方式 A：使用 PAT）**：

1. GitHub → Settings → Developer settings → Personal access tokens → Generate new token，勾选 `repo` 权限。
2. 推送时密码栏填入该 Token 而非账号密码。
3. 可用 Git Credential Manager 缓存凭据，避免重复输入：
   - Windows PowerShell:
     ```powershell
     git config --global credential.helper manager
     ```
   - Linux Bash:
     ```bash
     git config --global credential.helper store
     ```

**解决（方式 B：使用 SSH Key）**：

1. 生成 SSH Key。
   - Windows PowerShell:
     ```powershell
     ssh-keygen -t ed25519 -C "你的GitHub邮箱@example.com"
     ```
   - Linux Bash:
     ```bash
     ssh-keygen -t ed25519 -C "你的GitHub邮箱@example.com"
     ```
2. 将 `~/.ssh/id_ed25519.pub` 公钥内容添加到 GitHub → Settings → SSH and GPG keys。
3. 将远程地址改为 SSH：
   - Windows PowerShell:
     ```powershell
     git remote set-url origin git@github.com:<你的用户名>/curiosity-ppo.git
     ```
   - Linux Bash:
     ```bash
     git remote set-url origin git@github.com:<你的用户名>/curiosity-ppo.git
     ```

### 问题 4：分支名 main vs master 不一致

**现象**：本地分支为 `master`，远程仓库默认为 `main`，`push` 报错 `refusing to merge unrelated histories` 或推送后看不到代码。

**解决**：

1. 将本地分支重命名为 `main`：
   - Windows PowerShell:
     ```powershell
     git branch -M main
     ```
   - Linux Bash:
     ```bash
     git branch -M main
     ```
2. 确保已设置默认分支为 main：
   - Windows PowerShell:
     ```powershell
     git config --global init.defaultBranch main
     ```
   - Linux Bash:
     ```bash
     git config --global init.defaultBranch main
     ```
3. 重新推送：
   - Windows PowerShell:
     ```powershell
     git push -u origin main
     ```
   - Linux Bash:
     ```bash
     git push -u origin main
     ```

---

## 附：完整流程命令速查（一键参考）

- Windows PowerShell:
  ```powershell
  cd <项目根目录>
  git init
  git branch -M main
  # （确保 .gitignore 已就位）
  git add .
  git commit -m "feat: 初始化 CuriosityPPOAgent 项目结构与核心代码"
  git remote add origin https://github.com/<你的用户名>/curiosity-ppo.git
  git push -u origin main
  ```

- Linux Bash:
  ```bash
  cd ~/curiosity-ppo
  git init
  git branch -M main
  # （确保 .gitignore 已就位）
  git add .
  git commit -m "feat: 初始化 CuriosityPPOAgent 项目结构与核心代码"
  git remote add origin https://github.com/<你的用户名>/curiosity-ppo.git
  git push -u origin main
  ```

---

## 推送后验证清单

| 验证项 | 命令（PowerShell / Bash 通用） | 预期结果 |
|:---|:---|:---|
| 本地无未提交改动 | `git status` | `nothing to commit, working tree clean` |
| 提交历史完整 | `git log --oneline` | 显示 12 次提交 |
| 远程关联正常 | `git remote -v` | 显示 origin fetch/push 地址 |
| 远程分支同步 | `git branch -r` | 显示 `origin/main` |
| GitHub 网页可见代码 | 浏览器访问仓库 | 代码文件、docs、assets 均可见 |
