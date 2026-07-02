> 本清单用于 CuriosityPPOAgent 推送 GitHub 前的最终核验，共分 6 组、24 项检查。
> 每项均给出**合格判定条件**与**勾选框**，全部勾选后方可执行 `git push`。
> 验证命令统一标注 **Windows PowerShell** 与 **Linux Bash** 两种形式。

---

## 第一组：代码完整性

| 序号 | 检查项 | 合格判定条件 | 勾选 |
|:---:|:---|:---|:---:|
| 1.1 | Python 源码文件数 | 根目录及子目录下 `.py` 文件数等于 63 个 | [ ] |
| 1.2 | YAML 配置文件数 | 配置目录下 `.yaml` / `.yml` 文件数等于 7 个 | [ ] |
| 1.3 | 脚本完整性 | 训练 / 评估 / 导出 / 可视化入口脚本均存在且可执行，无空文件 | [ ] |
| 1.4 | Web Demo 文件数 | 前端 Demo 目录下文件数等于 22 份（含 Vite+React 组件与页面） | [ ] |
| 1.5 | 依赖清单完整 | `requirements.txt` / `pyproject.toml` 存在且版本固定，与实际 import 一致 | [ ] |

**验证命令：**

- 统计 Python 文件数
  - Windows PowerShell:
    ```powershell
    (Get-ChildItem -Recurse -Filter *.py -File | Measure-Object).Count
    ```
  - Linux Bash:
    ```bash
    find . -name "*.py" -type f | wc -l
    ```

- 统计 YAML 文件数
  - Windows PowerShell:
    ```powershell
    (Get-ChildItem -Recurse -Include *.yaml,*.yml -File | Measure-Object).Count
    ```
  - Linux Bash:
    ```bash
    find . -name "*.yaml" -o -name "*.yml" | wc -l
    ```

---

## 第二组：文档完整性

| 序号 | 检查项 | 合格判定条件 | 勾选 |
|:---:|:---|:---|:---:|
| 2.1 | README 主文档 | `README.md` 存在，含项目简介、安装、训练、评估、结果指标、引用说明 | [ ] |
| 2.2 | LICENSE 协议 | `LICENSE` 文件存在，内容为 MIT 协议全文，年份与版权人正确 | [ ] |
| 2.3 | CONTRIBUTING 贡献指南 | `CONTRIBUTING.md` 存在，说明分支规范、提交信息规范与 PR 流程 | [ ] |
| 2.4 | release_note 版本说明 | `docs/release_note.md` 或 `CHANGELOG.md` 存在，记录 12 次提交对应的功能变更 | [ ] |
| 2.5 | 文档总数 | `docs/` 下 Markdown 文档数等于 10 篇（含本清单与 git_upload_step 等） | [ ] |

**验证命令：**

- 统计 docs 目录文档数
  - Windows PowerShell:
    ```powershell
    (Get-ChildItem -Path .\docs -Recurse -Filter *.md -File | Measure-Object).Count
    ```
  - Linux Bash:
    ```bash
    find ./docs -name "*.md" -type f | wc -l
    ```

---

## 第三组：测试验证

| 序号 | 检查项 | 合格判定条件 | 勾选 |
|:---:|:---|:---|:---:|
| 3.1 | 单元测试通过 | 全量单元测试用例数等于 144 个，且全部 PASS，退出码为 0 | [ ] |
| 3.2 | 冒烟测试通过 | 冒烟测试 23 项全部通过，覆盖训练前向 / 导出 / 推理主链路 | [ ] |
| 3.3 | 性能指标达标 | Crafter 100 万步达 19.0%（基线 15.6%，+21.7%）；Montezuma 3500+（基线 120）；MiniGrid 96.8 万步（基线 242 万，2.5x 加速） | [ ] |
| 3.4 | 消融实验达标 | 4 组消融（full / no_icm / no_episodic / no_rnd）结果完整记录，full 组最优 | [ ] |
| 3.5 | 显存占用达标 | 训练时显存峰值 ≤ 2.2GB（RTX3060 Laptop 6GB 环境） | [ ] |

**验证命令：**

- 运行单元测试
  - Windows PowerShell:
    ```powershell
    python -m pytest tests/ -q
    ```
  - Linux Bash:
    ```bash
    python -m pytest tests/ -q
    ```

- 统计测试用例数
  - Windows PowerShell:
    ```powershell
    (Get-ChildItem -Recurse -Filter test_*.py | Select-String -Pattern "def test_" | Measure-Object).Count
    ```
  - Linux Bash:
    ```bash
    grep -rh "def test_" tests/ --include="test_*.py" | wc -l
    ```

---

## 第四组：敏感信息清除

| 序号 | 检查项 | 合格判定条件 | 勾选 |
|:---:|:---|:---|:---:|
| 4.1 | 无绝对路径 | 源码与配置中不存在 `C:\` / `D:\` / `/home/用户名/` 等本地绝对路径 | [ ] |
| 4.2 | 无 API Key | 代码与配置中不含 wandb API Key、GitHub Token、HuggingFace Token 等凭据 | [ ] |
| 4.3 | 无个人信息泄露 | 不含真实姓名、手机号、学号、私人邮箱等非必要个人信息 | [ ] |
| 4.4 | 无 wandb 密钥 | `wandb` 登录密钥未硬编码；`wandb` 目录已被 `.gitignore` 忽略 | [ ] |
| 4.5 | 无缓存与临时文件 | `__pycache__/`、`.pytest_cache/`、`*.egg-info/` 均未提交 | [ ] |

**验证命令：**

- 检查绝对路径残留
  - Windows PowerShell:
    ```powershell
    Get-ChildItem -Recurse -Include *.py,*.yaml,*.yml,*.md | Select-String -Pattern "C:\\|D:\\|/home/"
    ```
  - Linux Bash:
    ```bash
    grep -rn -E "C:\\|D:\\|/home/" --include="*.py" --include="*.yaml" --include="*.md" .
    ```

- 检查密钥残留
  - Windows PowerShell:
    ```powershell
    Get-ChildItem -Recurse -Include *.py,*.yaml,*.yml | Select-String -Pattern "api_key|API_KEY|token|secret|wandb.login"
    ```
  - Linux Bash:
    ```bash
    grep -rn -E "api_key|API_KEY|token|secret|wandb.login" --include="*.py" --include="*.yaml" .
    ```

> 以上命令输出应为空（或仅匹配到无关紧要的变量名注释），否则需清理后再提交。

---

## 第五组：仓库规范

| 序号 | 检查项 | 合格判定条件 | 勾选 |
|:---:|:---|:---|:---:|
| 5.1 | 开源协议 | 仓库采用 MIT 协议，`LICENSE` 文件存在且 README 中声明协议 | [ ] |
| 5.2 | .gitignore 完整 | `.gitignore` 含 `*.pt`、`*.onnx`、`wandb/`、`__pycache__/`、`.venv/` 等规则 | [ ] |
| 5.3 | GitHub Actions CI | `.github/workflows/` 下存在 CI 配置，能在 push / PR 时自动运行测试 | [ ] |
| 5.4 | Issue 模板 | `.github/ISSUE_TEMPLATE/` 下存在 Bug 报告与 Feature 请求模板 | [ ] |
| 5.5 | Topics 与 About | 已设置 17 个 Topics（见 `repo_tags.md`）与 350 字符内 About 描述 | [ ] |

**验证命令：**

- 检查 .gitignore 关键规则
  - Windows PowerShell:
    ```powershell
    Select-String -Path .\.gitignore -Pattern "\.pt|\.onnx|wandb|__pycache__"
    ```
  - Linux Bash:
    ```bash
    grep -E "\.pt|\.onnx|wandb|__pycache__" .gitignore
    ```

- 检查 CI 与模板目录
  - Windows PowerShell:
    ```powershell
    Test-Path .\.github\workflows
    Test-Path .\.github\ISSUE_TEMPLATE
    ```
  - Linux Bash:
    ```bash
    test -d .github/workflows && echo "CI OK" || echo "CI 缺失"
    test -d .github/ISSUE_TEMPLATE && echo "模板 OK" || echo "模板缺失"
    ```

---

## 第六组：最终确认

| 序号 | 检查项 | 合格判定条件 | 勾选 |
|:---:|:---|:---|:---:|
| 6.1 | git status 无未提交 | `git status` 输出 `nothing to commit, working tree clean` | [ ] |
| 6.2 | git log 12 次提交 | `git log --oneline` 显示恰好 12 次提交记录 | [ ] |
| 6.3 | 远程推送成功 | `git push` 无报错，GitHub 网页可见全部代码 | [ ] |
| 6.4 | 远程分支同步 | `git branch -r` 显示 `origin/main`，本地 main 与远程一致 | [ ] |
| 6.5 | 仓库可访问性 | 公网匿名访问仓库 URL 可正常打开，README 正常渲染 | [ ] |

**验证命令：**

- 查看工作区状态
  - Windows PowerShell:
    ```powershell
    git status
    ```
  - Linux Bash:
    ```bash
    git status
    ```

- 查看提交历史
  - Windows PowerShell:
    ```powershell
    git log --oneline
    ```
  - Linux Bash:
    ```bash
    git log --oneline
    ```

- 统计提交次数
  - Windows PowerShell:
    ```powershell
    (git log --oneline | Measure-Object -Line).Lines
    ```
  - Linux Bash:
    ```bash
    git log --oneline | wc -l
    ```

- 查看远程分支
  - Windows PowerShell:
    ```powershell
    git branch -r
    ```
  - Linux Bash:
    ```bash
    git branch -r
    ```

---

## 自检结果汇总

| 组别 | 检查项数 | 已通过数 | 状态 |
|:---|:---:|:---:|:---:|
| 第一组 代码完整性 | 5 | /5 | 待确认 |
| 第二组 文档完整性 | 5 | /5 | 待确认 |
| 第三组 测试验证 | 5 | /5 | 待确认 |
| 第四组 敏感信息清除 | 5 | /5 | 待确认 |
| 第五组 仓库规范 | 5 | /5 | 待确认 |
| 第六组 最终确认 | 5 | /5 | 待确认 |
| **合计** | **24** | **/24** | **待确认** |

> 全部 24 项勾选完毕后，方可确认仓库达到开源发布标准，执行最终 `git push`。
