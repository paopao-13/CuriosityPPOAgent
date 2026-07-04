# 贡献指南

欢迎提交 Issue 和 PR。

## 开发环境

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

```bash
# Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 提交规范

遵循 Conventional Commits：

```
<type>(<scope>): <description>
```

type: `feat` / `fix` / `docs` / `refactor` / `test` / `chore`

## 测试

提交前确保测试通过：

```powershell
python -m pytest tests/ -v
```

## PR 流程

1. Fork 仓库
2. 创建分支 `feature/xxx` 或 `fix/xxx`
3. 提交改动
4. 创建 PR，描述变更内容
