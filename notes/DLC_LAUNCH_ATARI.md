# DLC / DSW 一键启动命令(可直接粘贴)

本文件给出把 Atari Montezuma 10M 步重训任务提交到阿里云 PAI 的**完整可执行命令块**。
仓库已备:`scripts/cloud_train_atari.sh`(自愈 watch-dog)、`requirements_atari.txt`(精简依赖)。

---

## A. DSW 交互式(推荐先验证)

在 DSW 实例的 Terminal 里逐段执行:

```bash
# 1) 拉代码(替换成你自己的仓库地址)
git clone <你的仓库地址> curiosity-ppo
cd curiosity-ppo

# 2) 建隔离环境 + 装精简依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_atari.txt

# 3) 验证环境(GPU 可见 + Atari ROM 授权)
python -c "import ale_py, gymnasium, torch; print('env ok, cuda=', torch.cuda.is_available())"

# 4) 启动训练(自愈:崩溃自动续训,checkpoint 落 OSS)
CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh
```

---

## B. DLC 无人值守任务(提交后关电脑也继续跑)

在 DLC 建任务时按下面填写,**启动命令**框直接粘最后那段:

| 控制台字段 | 填什么 |
|---|---|
| 任务类型 | 自定义训练(PyTorch) |
| 镜像 | PyTorch 2.1 + Python 3.9 + CUDA 12(公共镜像) |
| 资源规格 | **A10(gn7i)** 免费额度内,或 A100 抢占式;内存务必 ≥30GB |
| 代码源 | Git 仓库 URL(填你的 curiosity-ppo 仓库) |
| 输出通道 | 指向 OSS Bucket(容器内对应 `/mnt/oss`) |
| 启动命令 | 见下框 |

```bash
set -e
# 进入代码目录(DLC git 挂载路径通常为 /ml/code 或同名目录,下面自动查找兜底)
REPO=$(find / -maxdepth 5 -name cloud_train_atari.sh 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
cd "$REPO/.."
pip install -r requirements_atari.txt
mkdir -p /mnt/oss/atari_seed42
CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh
```

---

## C. 常用运维命令

```bash
# 看进度(DSW 终端 或 DLC 日志页等同)
tail -f /mnt/oss/atari_seed42/train.log

# 小内存实例兜底(若规格内存 <32GB)
VEC_ENV=dummy CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh

# 关 torch.compile(极少数 A10/A100 上编译报错时)
USE_COMPILE=0 CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh
```

> 脚本行为:崩溃自动从最新 `step_*.pt` 续跑(最多 20 次);日志出现 `nan/inf/Traceback/MemoryError` 立即退出避免空转;日志 `tee` 同时写文件与 stdout(DLC 日志服务捕获 stdout)。
