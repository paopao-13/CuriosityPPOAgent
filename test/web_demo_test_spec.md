# Web 前端推理 Demo 专项完整测试规范

> **项目**：CuriosityPPOAgent — ICM+RND 分层新颖好奇心 PPO 智能体
> **测试对象**：Web Demo（Vite + React + TypeScript + ONNX Runtime Web）
> **硬件基准**：AMD R7 6800H + RTX3060 Laptop 6GB
> **文档版本**：v1.0
> **适用范围**：`web/` 目录下的前端推理 Demo 全流程测试

---

## 目录

1. [测试环境与前置条件](#1-测试环境与前置条件)
2. [ONNX 模型导出全流程测试](#2-onnx-模型导出全流程测试)
3. [本地 Vite 前端项目启动测试](#3-本地-vite-前端项目启动测试)
4. [网页可视化功能验收测试](#4-网页可视化功能验收测试)
5. [静态前端资源打包与 IGA Pages 部署验证](#5-静态前端资源打包与-iga-pages-部署验证)
6. [轻量化网页推理延迟测试](#6-轻量化网页推理延迟测试)
7. [测试结果汇总报告模板](#7-测试结果汇总报告模板)

---

## 1. 测试环境与前置条件

### 1.1 硬件环境

| 组件 | 规格 |
|------|------|
| CPU | AMD Ryzen 7 6800H（8核16线程） |
| GPU | NVIDIA RTX 3060 Laptop（6GB VRAM） |
| 内存 | ≥ 16GB |
| 存储 | SSD，剩余空间 ≥ 5GB |

### 1.2 软件环境

| 组件 | 版本要求 |
|------|----------|
| 操作系统 | Windows 10/11 64-bit |
| Python | ≥ 3.10 |
| Node.js | ≥ 18.0（推荐 20.x LTS） |
| npm | ≥ 9.0 |
| 浏览器 | Chrome ≥ 110 / Edge ≥ 110（支持 WebGL2） |
| CUDA（导出用） | ≥ 11.8（配合 PyTorch） |

### 1.3 项目依赖版本

```json
{
  "dependencies": {
    "react": "18.3.x",
    "react-dom": "18.3.x",
    "onnxruntime-web": "1.18.x"
  },
  "devDependencies": {
    "vite": "5.4.x",
    "typescript": "5.5.x",
    "@types/react": "18.3.x",
    "@types/react-dom": "18.3.x"
  }
}
```

### 1.4 项目结构概览

```text
project-root/
├── scripts/
│   └── export_onnx.py              # ONNX 模型导出脚本
├── results/
│   └── onnx/
│       └── model.onnx              # 导出的 ONNX 模型产物
├── web/                            # Web Demo 根目录
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── public/
│   │   └── model.onnx              # 前端加载的模型文件
│   └── src/
│       ├── App.tsx                 # 主页面
│       ├── game/
│       │   ├── env.ts              # MiniGrid 环境模拟器
│       │   ├── obs.ts              # 观测预处理
│       │   ├── actions.ts          # 动作定义
│       │   └── render.ts           # 渲染逻辑
│       ├── hooks/
│       │   └── useAgent.ts         # ONNX 模型加载与推理 Hook
│       ├── components/
│       │   ├── GridCanvas.tsx      # 全局地图渲染（16×16）
│       │   ├── AgentView.tsx       # Agent 视野渲染（7×7）
│       │   ├── ControlPanel.tsx    # 控制面板
│       │   └── StatsPanel.tsx      # 统计面板
│       └── styles/
│           └── app.css             # 样式文件
└── test/
    └── web_demo_test_spec.md       # 本测试规范
```

### 1.5 前置准备检查清单

| 序号 | 检查项 | 通过条件 |
|------|--------|----------|
| P-01 | Python 环境可用 | `python --version` 输出 ≥ 3.10 |
| P-02 | PyTorch 已安装且支持 CUDA | `python -c "import torch; print(torch.cuda.is_available())"` 输出 `True` |
| P-03 | 训练 checkpoint 文件存在 | 指定路径下存在 `.pt`/`.pth` 文件 |
| P-04 | Node.js 环境可用 | `node --version` 输出 ≥ 18.0 |
| P-05 | npm 环境可用 | `npm --version` 输出 ≥ 9.0 |
| P-06 | 浏览器支持 WebGL2 | 浏览器控制台执行 `document.createElement('canvas').getContext('webgl2')` 返回非 null |
| P-07 | `web/` 目录结构完整 | 上述项目结构中所有文件均存在 |

---

## 2. ONNX 模型导出全流程测试

### 2.1 测试用例 ONNX-01：导出脚本执行与模型文件生成

**测试目的**

验证 `scripts/export_onnx.py` 能够正确读取训练 checkpoint，导出完整的 PolicyNet（encoder + actor 头 → logits）ONNX 模型文件，且模型文件可被 ONNX Runtime 正常加载。

**分步执行操作**

1. 确认训练 checkpoint 文件路径，假设为 `checkpoints/best_model.pt`。
2. 创建导出输出目录：

   ```bash
   mkdir -p results/onnx
   ```

3. 执行导出命令：

   ```bash
   python scripts/export_onnx.py \
       --checkpoint checkpoints/best_model.pt \
       --output results/onnx/model.onnx \
       --env minigrid
   ```

4. 等待导出脚本执行完成，观察终端输出日志。
5. 检查 `results/onnx/model.onnx` 文件是否已生成：

   ```bash
   ls -la results/onnx/model.onnx
   ```

6. 使用以下 Python 脚本验证 ONNX 模型可加载且结构正确：

   ```python
   import onnxruntime as ort
   import onnx

   # 加载 ONNX 模型结构
   model = onnx.load("results/onnx/model.onnx")
   onnx.checker.check_model(model)

   # 创建推理会话
   session = ort.InferenceSession("results/onnx/model.onnx")

   # 打印输入/输出信息
   print("=== Inputs ===")
   for inp in session.get_inputs():
       print(f"  name={inp.name}, shape={inp.shape}, type={inp.type}")
   print("=== Outputs ===")
   for out in session.get_outputs():
       print(f"  name={out.name}, shape={out.shape}, type={out.type}")
   ```

**预期合格输出**

- 导出脚本终端输出无报错，包含类似 `ONNX model exported successfully` 的成功日志。
- `results/onnx/model.onnx` 文件已生成，文件大小 > 0（通常为数百 KB 至数 MB）。
- `onnx.checker.check_model(model)` 通过，无异常抛出。
- 输入节点信息显示动态 batch 轴（如 `shape=['batch', ...]` 或 `shape=[None, ...]`）。
- 输出节点为 logits，维度与动作空间一致（MiniGrid 环境下通常为 `[batch, num_actions]`）。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 脚本报 `FileNotFoundError` | checkpoint 路径错误 | 核对 `--checkpoint` 路径，使用绝对路径重试 |
| 脚本报 `KeyError: 'state_dict'` | checkpoint 格式不匹配 | 检查 checkpoint 加载方式，确认是否需要提取 `model_state_dict` 键 |
| `onnx.checker.check_model` 抛出异常 | ONNX 图结构不合法 | 检查 `torch.onnx.export` 的 `dynamic_axes` 参数配置；确认 `opset_version` 兼容 |
| 输出 logits 维度错误 | actor 头输出与动作空间不匹配 | 检查导出时 `num_actions` 参数是否与训练一致 |
| 模型文件大小为 0 | 导出过程中断 | 检查磁盘空间；重新执行导出命令 |

---

### 2.2 测试用例 ONNX-02：动态 Batch 轴验证

**测试目的**

验证导出的 ONNX 模型支持动态 batch 轴，能够正确处理 batch=1/4/8/16 的输入张量，输出维度正确对应。

**分步执行操作**

1. 编写以下 Python 验证脚本并保存为 `test/scripts/verify_dynamic_batch.py`：

   ```python
   import numpy as np
   import onnxruntime as ort

   session = ort.InferenceSession("results/onnx/model.onnx")
   input_name = session.get_inputs()[0].name

   # 观测预处理后输入形状: [batch, 3, 64, 64]
   for batch_size in [1, 4, 8, 16]:
       dummy_input = np.random.randn(batch_size, 3, 64, 64).astype(np.float32)
       outputs = session.run(None, {input_name: dummy_input})
       logits = outputs[0]
       print(f"batch={batch_size}: input={dummy_input.shape}, output={logits.shape}")
       assert logits.shape[0] == batch_size, \
           f"batch dimension mismatch: expected {batch_size}, got {logits.shape[0]}"
       print(f"  -> PASS")
   ```

2. 执行脚本：

   ```bash
   python test/scripts/verify_dynamic_batch.py
   ```

**预期合格输出**

- 四个 batch size（1/4/8/16）均成功推理，无报错。
- 每个 batch 的输出 shape 第一维与输入 batch 一致：
  - batch=1 → output shape `[1, num_actions]`
  - batch=4 → output shape `[4, num_actions]`
  - batch=8 → output shape `[8, num_actions]`
  - batch=16 → output shape `[16, num_actions]`
- 所有断言通过，输出 `-> PASS`。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 报 `INVALID_ARGUMENT` 或维度错误 | 导出时未设置动态 batch 轴 | 在 `torch.onnx.export` 中配置 `dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}` |
| 仅 batch=1 通过，其他失败 | 模型中存在硬编码 batch 维度 | 检查模型中是否有 `torch.Tensor([1, ...])` 硬编码；使用 `torch.randn(batch, ...)` 重导出 |
| 输入形状不匹配 | 预处理维度与模型期望不一致 | 确认输入为 `[batch, 3, 64, 64]`（NCHW 格式），核对 `obs.ts` 中 `getModelInput` 的输出维度 |

---

### 2.3 测试用例 ONNX-03：ONNX 与 PyTorch 推理一致性校验

**测试目的**

验证 ONNX Runtime 推理结果与原始 PyTorch 模型推理结果一致，最大绝对误差 < 1e-4，确保导出过程未引入精度损失。

**分步执行操作**

1. 编写一致性校验脚本并保存为 `test/scripts/verify_consistency.py`：

   ```python
   import numpy as np
   import torch
   import onnxruntime as ort
   from models.policy_net import PolicyNet  # 根据实际模块路径调整

   # 加载 PyTorch 模型
   checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
   pytorch_model = PolicyNet(...)  # 按实际构造参数初始化
   pytorch_model.load_state_dict(checkpoint["model_state_dict"])
   pytorch_model.eval()

   # 加载 ONNX 模型
   onnx_session = ort.InferenceSession("results/onnx/model.onnx")
   input_name = onnx_session.get_inputs()[0].name

   # 生成多组随机测试输入
   np.random.seed(42)
   max_abs_error = 0.0
   num_test_cases = 20

   for i in range(num_test_cases):
       batch_size = np.random.choice([1, 4, 8])
       dummy_input = np.random.randn(batch_size, 3, 64, 64).astype(np.float32)

       # PyTorch 推理
       with torch.no_grad():
           pt_input = torch.from_numpy(dummy_input)
           pt_output = pytorch_model(pt_input).numpy()

       # ONNX 推理
       onnx_output = onnx_session.run(None, {input_name: dummy_input})[0]

       # 计算最大绝对误差
       abs_error = np.max(np.abs(pt_output - onnx_output))
       max_abs_error = max(max_abs_error, abs_error)
       print(f"Case {i+1}: batch={batch_size}, max_abs_error={abs_error:.2e}")

   print(f"\n=== Overall Max Absolute Error: {max_abs_error:.2e} ===")
   print(f"=== Threshold: 1e-4 ===")
   if max_abs_error < 1e-4:
       print("=== RESULT: PASS ===")
   else:
       print("=== RESULT: FAIL ===")
   ```

2. 执行脚本：

   ```bash
   python test/scripts/verify_consistency.py
   ```

**预期合格输出**

- 20 组测试用例全部成功推理，无报错。
- 每组测试的最大绝对误差均 < 1e-4（通常在 1e-6 ~ 1e-5 量级）。
- 总体最大绝对误差 < 1e-4。
- 最终输出 `RESULT: PASS`。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 最大绝对误差 ≥ 1e-4 | 浮点精度差异或算子实现差异 | 尝试将 `torch.onnx.export` 的 `opset_version` 提升至 17+；检查是否有自定义算子未正确转换 |
| 误差在 1e-3 ~ 1e-2 量级 | 模型参数未正确加载或 BN 层模式错误 | 确认 `model.eval()` 已调用；检查 checkpoint 加载后参数是否一致 |
| 误差极大（> 0.1） | 模型结构导出错误 | 对比 PyTorch 与 ONNX 的中间层输出，定位差异层；检查 `forward` 方法中是否有条件分支 |
| ONNX 推理报形状错误 | 动态轴配置不当 | 参考 ONNX-02 修复方案 |

---

## 3. 本地 Vite 前端项目启动测试

### 3.1 测试用例 DEV-01：依赖安装

**测试目的**

验证 `web/` 目录下的前端项目能够成功安装所有 npm 依赖，无版本冲突或安装失败。

**分步执行操作**

1. 进入 Web Demo 目录：

   ```bash
   cd web
   ```

2. 清理可能存在的旧依赖缓存（首次安装可跳过）：

   ```bash
   rm -rf node_modules package-lock.json
   ```

3. 执行依赖安装：

   ```bash
   npm install
   ```

4. 观察安装过程输出，等待完成。
5. 验证关键依赖版本：

   ```bash
   npm ls react react-dom onnxruntime-web vite typescript
   ```

**预期合格输出**

- `npm install` 执行完成，无 `ERR!` 级别错误。
- 可能存在 `WARN`（如 peer dependency 警告），但不影响安装。
- `node_modules/` 目录已生成。
- `npm ls` 输出显示各依赖版本符合要求：
  - react@18.3.x
  - react-dom@18.3.x
  - onnxruntime-web@1.18.x
  - vite@5.4.x
  - typescript@5.5.x
- 无版本冲突（`UNMET DEPENDENCY` 或 `PEER DEP` 致命错误）。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| `npm install` 报 `ERESOLVE` | 依赖版本冲突 | 检查 `package.json` 中版本范围；尝试 `npm install --legacy-peer-deps` |
| 网络超时 | npm 源问题 | 切换 npm 源：`npm config set registry https://registry.npmmirror.com` |
| `onnxruntime-web` 安装失败 | 平台兼容性问题 | 确认 Node.js 版本 ≥ 18；检查 `onnxruntime-web` 是否有对应平台的预编译二进制 |
| `npm ls` 显示版本不符 | `package.json` 版本锁定不正确 | 更新 `package.json` 中对应依赖版本号后重新安装 |

---

### 3.2 测试用例 DEV-02：开发服务器启动

**测试目的**

验证 Vite 开发服务器能够正常启动，应用页面可访问，无编译错误。

**分步执行操作**

1. 确保 `web/` 目录下依赖已安装（DEV-01 通过）。
2. 将导出的 ONNX 模型复制到前端可访问目录：

   ```bash
   cp results/onnx/model.onnx web/public/model.onnx
   ```

3. 启动开发服务器：

   ```bash
   cd web
   npm run dev
   ```

4. 观察终端输出，等待 Vite 启动完成。
5. 在浏览器中打开终端显示的本地地址（通常为 `http://localhost:5173`）。
6. 打开浏览器开发者工具（F12），检查 Console 面板是否有报错。
7. 检查 Network 面板中 `model.onnx` 是否成功加载（HTTP 200）。

**预期合格输出**

- 终端输出类似：

  ```text
  VITE v5.4.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
  ```

- 浏览器页面正常加载，显示 Demo 主界面：
  - 左侧：16×16 全局地图画布
  - 右侧上方：7×7 Agent 视野画布
  - 右侧中部：控制面板（单步/自动播放/速度控制按钮）
  - 右侧下方：统计面板（步数/奖励/推理后端）
- Console 面板无红色错误（`Error`/`Uncaught`）。
- Network 面板中 `model.onnx` 请求状态为 200，文件大小与导出文件一致。
- 可能有黄色警告（如 React StrictMode 双渲染提示），属正常现象。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| `npm run dev` 报 TypeScript 编译错误 | 类型定义缺失或不匹配 | 检查 `tsconfig.json` 配置；根据报错信息修正 `.tsx`/`.ts` 文件中的类型错误 |
| Vite 启动报端口占用 | 5173 端口被占用 | 修改 `vite.config.ts` 中 `server.port`，或终止占用进程 |
| 页面白屏 | React 渲染异常 | 查看 Console 错误；检查 `App.tsx` 中组件导入路径是否正确 |
| `model.onnx` 加载 404 | 模型文件未放置到正确位置 | 确认 `web/public/model.onnx` 存在；Vite 中 `public/` 目录文件通过根路径访问 |
| Console 报 ONNX 加载错误 | onnxruntime-web WASM 文件缺失 | 检查 `vite.config.ts` 中是否正确配置 `optimizeDeps` 或 WASM 文件路径 |

---

### 3.3 测试用例 DEV-03：ONNX 模型浏览器端加载验证

**测试目的**

验证 `useAgent.ts` Hook 能够在浏览器中成功加载 ONNX 模型，初始化 ONNX Runtime Web 推理会话，并正确识别推理后端（WebGL/WASM）。

**分步执行操作**

1. 确保开发服务器已启动（DEV-02 通过）。
2. 在浏览器中打开 Demo 页面。
3. 打开开发者工具 Console 面板。
4. 观察页面加载后的 Console 日志，寻找模型加载相关日志。
5. 检查统计面板中"推理后端"字段显示的值。
6. 在 Console 中手动检查 ONNX Runtime 状态：

   ```javascript
   // 查看 ort 全局对象是否可用
   console.log(typeof ort);
   ```

7. 若模型存在，验证会话已创建；若模型不存在（删除 `public/model.onnx` 后刷新），验证随机策略回退。

**预期合格输出**

- 模型存在时：
  - Console 输出模型加载成功日志（如 `ONNX model loaded successfully`）。
  - 统计面板"推理后端"字段显示 `webgl` 或 `wasm`。
  - 无加载错误日志。
- 模型缺失时（回退场景）：
  - Console 输出模型缺失警告（如 `Model not found, falling back to random policy`）。
  - 统计面板"推理后端"字段显示 `random` 或 `fallback`。
  - 页面功能仍然可用（可单步/自动播放，使用随机策略）。
  - 无致命错误阻断页面交互。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| Console 报 `Failed to fetch model.onnx` | 模型文件路径错误或未部署 | 确认 `useAgent.ts` 中模型路径配置为 `/model.onnx`（对应 `public/` 目录） |
| 报 `WebGL is not supported` | 浏览器/显卡不支持 WebGL2 | 更新浏览器版本；更新显卡驱动；回退至 WASM 后端 |
| 报 `wasm streaming compile failed` | WASM 文件加载失败 | 检查 `vite.config.ts` 中 `optimizeDeps.exclude` 是否包含 `onnxruntime-web`；配置 `ORT_WASM_PATHS` |
| 推理后端字段为空/undefined | `useAgent.ts` 未正确返回后端信息 | 检查 Hook 中 `session.handler` 或 backend 检测逻辑 |
| 模型缺失时页面崩溃 | 回退逻辑未实现或异常未捕获 | 在 `useAgent.ts` 中添加 try-catch；确保 `loadModel` 失败时设置 fallback 标志 |

---

## 4. 网页可视化功能验收测试

### 4.1 测试用例 VIS-01：全局地图实时渲染验证

**测试目的**

验证 `GridCanvas` 组件能够正确渲染 16×16 的 MiniGrid 全局地图，Agent 位置、障碍物、目标等元素实时更新。

**分步执行操作**

1. 确保开发服务器已启动且页面正常加载（DEV-02/DEV-03 通过）。
2. 观察页面左侧的全局地图画布区域。
3. 视觉检查以下要素：
   - 画布尺寸为 16×16 网格。
   - 网格边界清晰可辨。
   - Agent（通常用特殊颜色/图标标记）可见。
   - 障碍物（墙壁）与可通行区域有视觉区分。
   - 目标位置（如有）有特殊标记。
4. 点击控制面板"单步执行"按钮 1 次。
5. 观察 Agent 在地图上的位置是否发生变化。
6. 连续点击"单步执行"5 次，观察每次 Agent 位置更新。
7. 使用浏览器截图对比初始状态与执行后状态。

**预期合格输出**

- 16×16 网格完整渲染，无缺失或溢出。
- Agent、障碍物、目标等元素颜色/图标清晰可辨，与 `render.ts` 中定义一致。
- 每次单步执行后，Agent 在全局地图上的位置实时更新，无延迟。
- 渲染无闪烁、撕裂或画面残留（前一帧状态不残留）。
- Agent 不会移出地图边界或穿越墙壁。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 画布空白/不渲染 | Canvas 绑定失败 | 检查 `GridCanvas.tsx` 中 `useRef` 与 `canvas` 绑定；确认 `useEffect` 中调用了渲染函数 |
| 网格尺寸错误（非 16×16） | 渲染参数硬编码错误 | 检查 `render.ts` 中网格维度常量；确认 `env.ts` 中 `grid_size=16` |
| Agent 位置不更新 | 状态未传递到渲染层 | 检查 `App.tsx` 中 state 管理与 props 传递；确认 `useEffect` 依赖数组包含 agent 状态 |
| 画面闪烁/残留 | Canvas 未清除前一帧 | 在 `render.ts` 渲染函数开头添加 `ctx.clearRect()` 或 `ctx.fillRect()` 清屏 |
| Agent 穿墙/越界 | 环境模拟器碰撞检测异常 | 检查 `env.ts` 中 `step()` 方法的边界与碰撞逻辑 |

---

### 4.2 测试用例 VIS-02：Agent 视野 7×7 实时渲染验证

**测试目的**

验证 `AgentView` 组件能够正确渲染 Agent 当前视野范围内的 7×7 局部观测，并随 Agent 移动实时更新。

**分步执行操作**

1. 在 Demo 页面正常加载状态下，观察右侧上方的 Agent 视野画布区域。
2. 视觉检查以下要素：
   - 画布尺寸为 7×7 网格。
   - Agent 位于视野中心（第 4 行第 4 列，即索引 [3,3]）。
   - 视野范围内的障碍物、目标等元素可见。
   - 视野外的区域（如有遮挡）有视觉区分（如半透明/灰色）。
3. 点击"单步执行"按钮 1 次，观察视野内容是否随 Agent 移动而变化。
4. 连续单步执行 5 次，每次观察视野内容更新。
5. 将 Agent 视野与全局地图中 Agent 周围 7×7 区域对比，验证一致性。

**预期合格输出**

- 7×7 视野网格完整渲染。
- Agent 始终位于视野中心位置。
- 视野内容与全局地图中 Agent 周围对应区域一致（考虑朝向/旋转因素）。
- 每次 Agent 移动后，视野内容实时更新，无延迟。
- 视野边缘（Agent 靠近地图边界时）正确处理越界区域（显示为墙壁或空白）。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 视野画布空白 | `AgentView` 组件未接收视野数据 | 检查 `App.tsx` 中 `agentView` prop 传递；确认 `env.ts` 中 `get_agent_view()` 返回有效数据 |
| 视野尺寸错误 | 视野维度参数错误 | 确认 `obs.ts` 或 `env.ts` 中 `view_size=7` 配置 |
| 视野内容与全局地图不一致 | 视野提取逻辑错误 | 检查 `env.ts` 中视野裁剪与朝向变换逻辑；对比 `get_agent_view()` 输出与全局地图 |
| Agent 不在视野中心 | 视野中心偏移计算错误 | 确认视野中心索引为 `(view_size - 1) / 2 = 3`；检查 `render.ts` 中 Agent 在视野画布的绘制坐标 |
| 视野不随移动更新 | 视野状态未更新 | 检查 `App.tsx` 中 `step()` 调用后是否更新 `agentView` 状态 |

---

### 4.3 测试用例 VIS-03：控制面板单步执行功能

**测试目的**

验证 `ControlPanel` 组件的"单步执行"按钮能够触发一次完整的推理-动作-环境更新-渲染流程。

**分步执行操作**

1. 确保 Demo 页面正常加载且模型已加载（或随机策略回退已生效）。
2. 记录当前统计面板中的步数值（初始应为 0）。
3. 点击控制面板"单步执行"按钮。
4. 观察以下变化：
   - 全局地图中 Agent 位置是否变化。
   - Agent 视野是否更新。
   - 统计面板步数是否 +1。
   - 统计面板奖励是否更新（如有奖励变化）。
5. 重复步骤 3-4 共 10 次。
6. 验证步数最终为 10，每次点击均有且仅有一次动作执行。

**预期合格输出**

- 每次点击"单步执行"后：
  - Agent 执行一个动作（移动/转向等），全局地图与视野同步更新。
  - 统计面板步数精确 +1。
  - 奖励值根据环境反馈更新（可能为 0 或非零）。
- 10 次点击后步数精确为 10。
- 按钮响应无延迟（点击后 < 200ms 内画面更新）。
- 无重复触发（一次点击仅执行一步）。
- 按钮在推理过程中不会被禁用过久导致无法连续点击（除非设计为防抖）。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 点击按钮无反应 | 事件绑定失败或推理报错 | 检查 `ControlPanel.tsx` 中 `onClick` 处理函数；查看 Console 是否有推理错误 |
| 步数不增加或增加多次 | 状态更新逻辑错误 | 检查 `App.tsx` 中 `handleStep` 函数；确认 `setStepCount(prev => prev + 1)` 使用函数式更新 |
| Agent 位置不变 | 动作执行或环境更新失败 | 检查 `env.ts` 中 `step(action)` 返回值；确认动作索引有效 |
| 画面更新延迟过大 | 渲染阻塞或推理耗时过长 | 参考 LAT-01 延迟测试；检查是否同步阻塞了主线程 |
| 奖励不更新 | 奖励计算逻辑缺失 | 检查 `env.ts` 中 `step()` 返回的 reward 是否传递到统计面板 |

---

### 4.4 测试用例 VIS-04：控制面板自动播放功能

**测试目的**

验证"自动播放"按钮能够启动连续推理循环，Agent 持续执行动作，画面实时更新，且可通过按钮停止。

**分步执行操作**

1. 确保 Demo 页面正常加载。
2. 点击控制面板"自动播放"按钮。
3. 观察 Agent 是否开始连续自动执行动作：
   - 全局地图中 Agent 持续移动。
   - Agent 视野持续更新。
   - 统计面板步数持续递增。
4. 等待约 10 秒（或步数达到 50+），观察播放是否流畅。
5. 再次点击按钮（此时应变为"暂停"或"停止"），观察是否停止。
6. 停止后记录最终步数。
7. 再次点击"自动播放"，验证可恢复播放。
8. 暂停后点击"单步执行"，验证单步功能仍正常。

**预期合格输出**

- 点击"自动播放"后，Agent 立即开始连续执行动作。
- 全局地图与视野画面持续实时更新，无卡顿（视觉流畅度 ≥ 10 FPS）。
- 统计面板步数持续递增，奖励值实时更新。
- 再次点击按钮可停止自动播放，步数不再增加。
- 停止后可重新启动自动播放。
- 自动播放停止后，单步执行功能仍正常工作。
- 自动播放期间浏览器无冻结、无崩溃、Console 无错误堆积。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 点击自动播放无反应 | 事件处理或定时器未启动 | 检查 `ControlPanel.tsx` / `App.tsx` 中 `setInterval`/`requestAnimationFrame` 启动逻辑 |
| 播放卡顿/掉帧 | 推理阻塞主线程 | 使用 `requestAnimationFrame` 替代 `setInterval`；考虑将推理放入 Web Worker |
| 无法停止 | 停止逻辑未正确清除定时器 | 确认 `clearInterval`/`cancelAnimationFrame` 被正确调用；检查状态标志 |
| 步数增长但画面不更新 | 渲染未跟上推理速度 | 降低自动播放频率；在每次推理后同步触发渲染 |
| 播放一段时间后浏览器崩溃 | 内存泄漏 | 检查 `useEffect` 清理函数；确认 ONNX 推理张量被正确释放 |
| 暂停后单步失效 | 状态冲突 | 检查 `isPlaying` 状态标志；确保暂停时单步逻辑不被阻塞 |

---

### 4.5 测试用例 VIS-05：速度控制功能

**测试目的**

验证速度控制（滑块/按钮）能够调节自动播放的执行频率，且不同速度档位下画面更新均流畅。

**分步执行操作**

1. 启动自动播放（VIS-04 通过）。
2. 将速度控制调至最低档（如 1 步/秒或最慢档位）。
3. 观察 Agent 执行频率，使用计时器估算每步间隔。
4. 将速度控制调至中档。
5. 观察执行频率变化。
6. 将速度控制调至最高档（如 10+ 步/秒或最快档位）。
7. 观察执行频率与画面流畅度。
8. 在不同速度档位下观察统计面板步数增长速率。

**预期合格输出**

- 最低速度档：每步间隔明显较长（约 500ms-1000ms），画面更新从容可辨。
- 中档速度：每步间隔适中（约 100ms-300ms），画面更新流畅。
- 最高速度档：每步间隔极短（< 100ms），画面快速更新但仍可观察 Agent 运动。
- 速度变化即时生效，无需暂停重启。
- 所有速度档位下画面无撕裂、无冻结。
- 统计面板步数增长速率与速度档位成正比。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 速度滑块无效果 | 速度值未传递到定时器 | 检查 `App.tsx` 中速度状态与 `setInterval` delay 的绑定；使用 `useEffect` 监听速度变化重建定时器 |
| 最高速度时画面冻结 | 推理耗时超过间隔 | 确认最高速度间隔 > 单步推理延迟；使用 `requestAnimationFrame` 自适应节奏 |
| 速度变化需暂停才生效 | 定时器未动态更新 | 在 `useEffect` 依赖数组中加入速度变量，自动清除并重建定时器 |
| 速度档位刻度不合理 | 配置缺失 | 在 `ControlPanel.tsx` 中定义合理的速度档位映射（如 1x/2x/5x/10x） |

---

### 4.6 测试用例 VIS-06：统计面板信息显示

**测试目的**

验证 `StatsPanel` 组件能够正确显示步数、累计奖励、推理后端类型，且数值实时更新。

**分步执行操作**

1. 在 Demo 页面加载后，观察右侧下方的统计面板。
2. 记录初始显示值：
   - 步数：应为 0
   - 奖励：应为 0（或初始值）
   - 推理后端：应显示 `webgl`/`wasm`/`random`
3. 执行单步操作 1 次，观察统计面板数值变化。
4. 执行自动播放 20 步，观察：
   - 步数是否精确递增至 20。
   - 奖励值是否根据环境反馈累计更新。
   - 推理后端显示是否保持稳定。
5. 在 Console 中手动对比统计面板显示值与实际状态值：

   ```javascript
   // 假设可通过 React DevTools 或全局变量访问状态
   ```

6. 调整浏览器窗口大小，验证统计面板布局自适应。

**预期合格输出**

- 统计面板包含至少三项信息：步数、奖励、推理后端。
- 初始值正确（步数=0，奖励=0 或初始值）。
- 每次单步/自动播放后，步数精确递增。
- 奖励值实时累计，数值与 `env.ts` 返回的 reward 一致。
- 推理后端稳定显示，不随步数变化而闪烁（除非后端确实切换）。
- 窗口缩放时面板布局不错乱、不遮挡。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 统计面板空白 | 组件未渲染或 props 未传递 | 检查 `App.tsx` 中 `StatsPanel` 的 props 传递 |
| 步数不更新 | 状态未绑定 | 检查 `stepCount` state 是否在 `handleStep` 中更新并传递给 `StatsPanel` |
| 奖励显示错误 | 奖励计算或累加逻辑错误 | 检查奖励累加使用函数式更新 `setReward(prev => prev + stepReward)` |
| 推理后端显示 undefined | 后端检测逻辑缺失 | 在 `useAgent.ts` 中通过 `session.handler` 或配置项检测并返回后端类型 |
| 布局错乱 | CSS 样式问题 | 检查 `app.css` 中统计面板的 flexbox/grid 布局；添加 `min-width`/`overflow` 控制 |

---

### 4.7 测试用例 VIS-07：推理流畅度与无卡顿验证

**测试目的**

验证在自动播放模式下，推理与渲染流程不阻塞主线程，页面交互流畅，无明显卡顿。

**分步执行操作**

1. 启动 Demo 页面，加载模型。
2. 启动自动播放，速度设为中高档。
3. 在自动播放期间执行以下交互操作，观察响应速度：
   - 拖动速度控制滑块。
   - 滚动页面（如有滚动条）。
   - 点击暂停按钮。
   - 鼠标悬停在按钮上观察 hover 效果。
4. 打开开发者工具 Performance 面板，录制 5 秒自动播放过程。
5. 分析性能火焰图：
   - 检查主线程长任务（Long Task > 50ms）数量。
   - 检查 FPS 是否稳定。
6. 检查 Console 是否有 `onnxruntime` 相关的性能警告。

**预期合格输出**

- 自动播放期间所有交互操作响应迅速（< 100ms 响应）。
- 速度滑块拖动即时生效。
- 暂停按钮点击后立即停止。
- Performance 面板录制中：
  - FPS 稳定在 ≥ 30（理想 ≥ 60）。
  - 长任务（> 50ms）数量极少（5 秒内 ≤ 3 个）。
  - 主线程无持续阻塞（绿色空闲帧存在）。
- Console 无性能相关错误或警告堆积。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 交互严重卡顿 | 推理阻塞主线程 | 将 ONNX 推理迁移至 Web Worker；使用异步推理 `session.run()` 配合 `await` |
| FPS 大幅下降 | 渲染频率过高或 Canvas 重绘开销大 | 降低渲染频率（推理可高频但渲染限频）；使用 `requestAnimationFrame` 控制渲染 |
| 长任务频繁 | 同步推理或大数据拷贝 | 确认 ONNX Runtime Web 使用异步 API；减少 `Float32Array` 频繁创建 |
| 内存持续增长 | 张量/会话未释放 | 检查 `useAgent.ts` 中 `ort.Tensor` 是否在推理后被 GC；避免闭包持有大对象 |

---

## 5. 静态前端资源打包与 IGA Pages 部署验证

### 5.1 测试用例 BUILD-01：生产环境打包

**测试目的**

验证 `npm run build` 能够成功执行 TypeScript 类型检查与 Vite 生产打包，生成可部署的静态资源。

**分步执行操作**

1. 确保 `web/` 目录下依赖已安装（DEV-01 通过）。
2. 执行打包命令：

   ```bash
   cd web
   npm run build
   ```

3. 观察终端输出，等待打包完成。
4. 检查 `web/dist/` 目录是否生成：

   ```bash
   ls -la dist/
   ls -la dist/assets/
   ```

5. 检查生成文件列表，确认包含：
   - `index.html`
   - `assets/*.js`（打包后的 JS）
   - `assets/*.css`（打包后的 CSS）
   - `model.onnx`（若 `public/` 中存在）
6. 检查 `dist/index.html` 中资源引用路径是否正确（相对路径或绝对路径）。
7. 验证打包产物无 TypeScript 类型错误（`tsc` 阶段通过）。

**预期合格输出**

- 终端输出类似：

  ```text
  > curiosity-ppo-web@1.0.0 build
  > tsc && vite build

  vite v5.4.x building for production...
  ✓ N modules transformed.
  dist/index.html                  0.46 kB
  dist/assets/index-xxxxx.css     xx.xx kB
  dist/assets/index-xxxxx.js      xxx.xx kB
  dist/model.onnx                  xxx.xx kB
  ✓ built in xxx ms
  ```

- `tsc` 阶段无类型错误（`error TSxxxx` 不出现）。
- `dist/` 目录生成完整，包含 HTML/JS/CSS/ONNX 文件。
- JS 文件大小合理（通常 < 2MB，含 onnxruntime-web）。
- `dist/index.html` 中资源引用为 hashed 文件名（如 `/assets/index-abc123.js`）。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| `tsc` 报类型错误 | TypeScript 类型不匹配 | 根据报错信息修正 `.ts`/`.tsx` 文件；检查 `tsconfig.json` 中 `strict` 配置 |
| Vite 打包报 `Could not resolve` 依赖 | 导入路径错误或依赖缺失 | 检查 import 路径；确认 `node_modules` 完整安装 |
| 打包产物体积过大 | onnxruntime-web 全量打包 | 配置 Vite manualChunks 分包；确认未引入冗余依赖 |
| `dist/` 中缺少 `model.onnx` | `public/` 目录中无模型文件 | 将 ONNX 模型复制到 `web/public/model.onnx` 后重新打包 |
| `index.html` 引用路径错误 | base 配置不当 | 若部署到子路径，在 `vite.config.ts` 中设置 `base: './'` 或对应路径 |

---

### 5.2 测试用例 BUILD-02：打包产物本地预览

**测试目的**

验证打包产物能够通过 `npm run preview` 正常启动预览服务器，页面功能与开发模式一致。

**分步执行操作**

1. 确保 BUILD-01 打包成功。
2. 启动预览服务器：

   ```bash
   cd web
   npm run preview
   ```

3. 在浏览器中打开终端显示的预览地址（通常为 `http://localhost:4173`）。
4. 执行以下功能验证（与开发模式对比）：
   - 页面是否正常加载，布局完整。
   - 模型是否成功加载（Console 无错误）。
   - 单步执行功能正常。
   - 自动播放功能正常。
   - 统计面板显示正常。
5. 打开开发者工具 Network 面板，确认所有资源（JS/CSS/ONNX/WASM）加载成功（HTTP 200）。

**预期合格输出**

- 预览服务器启动，终端显示 `http://localhost:4173`。
- 页面加载完整，布局与开发模式一致。
- 所有功能（模型加载/单步/自动播放/统计面板）正常工作。
- Network 面板中所有资源状态为 200，无 404。
- Console 无错误。
- onnxruntime-web 的 WASM 文件成功加载（Network 中可见 `.wasm` 请求）。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 页面白屏 | JS/CSS 路径错误 | 检查 `vite.config.ts` 中 `base` 配置；若部署在子路径需设置正确 base |
| WASM 文件 404 | onnxruntime-web WASM 未正确打包 | 在 `vite.config.ts` 中配置 `optimizeDeps`；使用 `copy-webpack-plugin` 或 Vite 的 `publicDir` 复制 WASM 文件 |
| 模型加载失败 | `model.onnx` 未包含在打包产物中 | 确认 `web/public/model.onnx` 存在；Vite 会自动将 `public/` 内容复制到 `dist/` |
| 功能异常但开发模式正常 | 环境差异 | 检查是否有 `import.meta.env.DEV` 条件分支；确认生产模式下的代码路径正确 |

---

### 5.3 测试用例 BUILD-03：IGA Pages 静态部署可行性验证

**测试目的**

验证 `web/dist/` 打包产物具备部署到 IGA Pages 静态托管服务的技术可行性，资源路径、CORS、WASM 加载等在静态托管环境下可正常工作。

**分步执行操作**

1. 确保 BUILD-01 打包成功，`dist/` 目录完整。
2. 检查 `vite.config.ts` 中的 `base` 配置：
   - 若 IGA Pages 部署在根路径，`base` 应为 `'/'` 或默认。
   - 若部署在子路径（如 `https://user.iga.pages/curiosity-ppo/`），`base` 应为 `'/curiosity-ppo/'`。
3. 若需调整 base，修改后重新打包：

   ```bash
   cd web
   npm run build
   ```

4. 检查 `dist/index.html` 中的资源引用路径是否与 base 配置一致。
5. 验证 ONNX 模型与 WASM 文件的加载路径：
   - 确认 `useAgent.ts` 中模型路径使用相对路径或与 base 一致的路径。
   - 确认 onnxruntime-web 的 WASM 路径配置（`ortEnv.wasm.wasmPaths`）指向正确。
6. 模拟静态部署验证（使用任意静态文件服务器）：

   ```bash
   # 使用 npx http-server 模拟静态托管
   npx http-server dist -p 8080 --cors
   ```

7. 在浏览器中打开 `http://localhost:8080`，验证全部功能。
8. 检查 Console 中是否有 CORS 或 MIME 类型相关错误。
9. （可选）实际部署到 IGA Pages，验证线上访问。

**预期合格输出**

- `dist/` 产物在静态文件服务器下正常加载。
- 所有资源（HTML/JS/CSS/ONNX/WASM）HTTP 200，无 404。
- 模型加载成功，推理功能正常。
- Console 无 CORS 错误、无 MIME 类型错误。
- WASM 文件的 `Content-Type` 为 `application/wasm`（静态服务器正确配置）。
- 页面功能与本地预览一致。
- 若部署到 IGA Pages，线上 URL 可正常访问，功能完整。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 资源 404 | base 路径不匹配 | 修改 `vite.config.ts` 中 `base` 配置；确保与 IGA Pages 部署路径一致 |
| WASM MIME 类型错误 | 静态服务器未配置 `.wasm` 的 Content-Type | 在 IGA Pages 配置中添加 MIME 映射；或使用 onnxruntime-web 的 JS fallback 模式 |
| CORS 错误 | 跨域资源加载受限 | 确保所有资源同源部署；或将 ONNX 模型通过 CDN 配置正确的 CORS 头 |
| 模型加载失败 | 模型路径在部署后变化 | 在 `useAgent.ts` 中使用 `import.meta.env.BASE_URL` 拼接模型路径 |
| SPA 路由 404 | 刷新页面时服务器未回退到 index.html | IGA Pages 通常默认支持 SPA 回退；若不支持需配置 `_redirects` 文件（`/* /index.html 200`） |
| HTTPS 混合内容警告 | HTTP 资源在 HTTPS 页面加载 | 确保所有资源使用 HTTPS；IGA Pages 默认提供 HTTPS |

---

## 6. 轻量化网页推理延迟测试

### 6.1 测试用例 LAT-01：单步推理延迟基准测试

**测试目的**

测量浏览器端 ONNX Runtime Web 单步推理延迟，验证单步推理时间 < 100ms 的合格标准。

**分步执行操作**

1. 确保 Demo 页面正常加载且模型已成功加载（推理后端为 WebGL 或 WASM）。
2. 打开浏览器开发者工具 Console 面板。
3. 在 `useAgent.ts` 中确认推理函数已包含性能计时（或手动注入）：

   ```typescript
   // useAgent.ts 中的推理函数应包含性能计时
   const predict = async (obs: Float32Array): Promise<number> => {
     const start = performance.now();
     const tensor = new ort.Tensor("float32", obs, [1, 3, 64, 64]);
     const results = await session.run({ [inputName]: tensor });
     const logits = results[outputName].data as Float32Array;
     const action = argmax(logits);
     const elapsed = performance.now() - start;
     console.log(`[Inference] ${elapsed.toFixed(2)} ms`);
     return action;
   };
   ```

4. 若代码中无计时，在 Console 中手动注入测试脚本：

   ```javascript
   // 假设可通过全局访问或 React DevTools 获取推理函数
   // 执行 20 次推理并记录延迟
   const latencies = [];
   const numTrials = 20;

   async function benchmarkInference() {
     // 预热推理（首次推理包含编译开销，不计入统计）
     console.log("Warming up...");

     for (let i = 0; i < 5; i++) {
       // 触发一次预热推理（通过点击单步或调用推理函数）
     }

     console.log("Starting benchmark...");
     for (let i = 0; i < numTrials; i++) {
       const start = performance.now();
       // 触发一次单步推理（点击单步按钮或调用推理函数）
       // 注意：需等待推理完成
       // latencies.push(performance.now() - start);
     }

     // 计算统计量
     const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
     const max = Math.max(...latencies);
     const min = Math.min(...latencies);
     const p95 = latencies.sort((a, b) => a - b)[Math.floor(numTrials * 0.95)];

     console.log(`=== Inference Latency (${numTrials} trials) ===`);
     console.log(`  Average: ${avg.toFixed(2)} ms`);
     console.log(`  Min:     ${min.toFixed(2)} ms`);
     console.log(`  Max:     ${max.toFixed(2)} ms`);
     console.log(`  P95:     ${p95.toFixed(2)} ms`);
     console.log(`  Threshold: 100 ms`);
     console.log(`  Result: ${avg < 100 ? "PASS" : "FAIL"}`);
   }

   benchmarkInference();
   ```

5. 执行单步操作 20 次（含 5 次预热），记录每次推理延迟日志。
6. 收集 Console 中 `[Inference] xxx ms` 日志，计算统计量。
7. 分别在以下条件下重复测试：
   - WebGL 后端
   - WASM 后端（可通过 `ortEnv.wasm.proxy` 或配置强制指定）
   - 不同浏览器（Chrome / Edge）

**预期合格输出**

| 指标 | 合格标准 |
|------|----------|
| 平均推理延迟 | < 100ms |
| P95 推理延迟 | < 150ms（参考） |
| 最大推理延迟（稳态） | < 200ms（参考） |
| 首次推理延迟（含编译） | 不作合格判定（允许较长，预热后稳定） |

- 排除前 5 次预热推理后，稳态平均推理延迟 < 100ms。
- WebGL 后端通常比 WASM 更快（WebGL 可达 10-50ms，WASM 可达 30-90ms）。
- 延迟波动小（max - min < 50ms 为佳，非硬性要求）。
- 输出示例：

  ```text
  === Inference Latency (20 trials) ===
    Average: 35.42 ms
    Min:     28.11 ms
    Max:     52.67 ms
    P95:     48.93 ms
    Threshold: 100 ms
    Result: PASS
  ```

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 平均延迟 ≥ 100ms | 推理后端效率低或输入预处理耗时 | 优先使用 WebGL 后端；优化 `obs.ts` 中 `getModelInput` 的预处理逻辑；减少不必要的张量拷贝 |
| WebGL 后端延迟 > WASM | WebGL 不可用或回退至 CPU | 检查浏览器 WebGL2 支持；更新显卡驱动；确认 `ortEnv` 中 `webgl` 后端已启用 |
| 延迟波动大（max - min > 100ms） | GC 压力或主线程竞争 | 复用 `ort.Tensor` 对象；将推理迁移至 Web Worker 隔离主线程 |
| 首次推理极慢（> 500ms） | 模型编译/预热开销 | 属正常现象；在页面加载时执行预热推理（隐藏的 dummy 推理） |
| 所有后端均 > 100ms | 模型过大或硬件性能不足 | 考虑模型量化（INT8）；减小 encoder 输入分辨率；使用 ONNX Simplifier 精简模型图 |

---

### 6.2 测试用例 LAT-02：连续推理延迟稳定性测试

**测试目的**

验证在自动播放模式下连续推理的延迟稳定性，确保长时间运行不会出现延迟退化或内存增长导致的性能下降。

**分步执行操作**

1. 确保 LAT-01 通过（单步延迟合格）。
2. 在 `useAgent.ts` 中添加延迟记录逻辑，将每步推理延迟存入数组（或通过 Console 日志收集）。
3. 启动自动播放，速度设为最高档（推理间隔最小）。
4. 持续运行 200 步（或 2 分钟，以先到者为准）。
5. 在运行过程中观察：
   - 画面是否持续流畅。
   - Console 中推理延迟日志是否有明显上升趋势。
   - 浏览器任务管理器中该页面的内存使用是否持续增长。
6. 运行结束后，收集全部推理延迟数据，计算：
   - 前 50 步平均延迟
   - 后 50 步平均延迟
   - 全程最大延迟
   - 延迟标准差
7. 检查浏览器内存使用变化（初始 vs 结束）。

**预期合格输出**

- 全程 200 步推理均成功完成，无报错。
- 前 50 步平均延迟与后 50 步平均延迟差异 < 20ms（无显著退化）。
- 全程最大延迟 < 200ms（偶发尖峰允许，但不应持续）。
- 延迟标准差 < 30ms（稳定性良好）。
- 浏览器内存使用增长 < 50MB（无明显内存泄漏）。
- 画面全程流畅，无明显卡顿。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 后 50 步延迟显著高于前 50 步 | 内存泄漏导致 GC 频繁 | 检查 `ort.Tensor` 是否被释放；避免在推理循环中累积引用；使用 Chrome Memory Profiler 定位泄漏 |
| 延迟出现周期性尖峰 | GC 周期触发 | 减少 Float32Array 创建频率；复用输入输出缓冲区 |
| 内存持续增长 | ONNX 会话或张量泄漏 | 确认每次推理后释放 `ort.Tensor`；检查 `useAgent.ts` 中是否有闭包持有旧张量 |
| 200 步后浏览器崩溃 | 严重内存泄漏 | 使用 Chrome DevTools Memory 面板做 Heap Snapshot 对比；修复泄漏引用 |
| 延迟逐渐升高但内存稳定 | 浏览器内部缓存膨胀 | 定期刷新页面；考虑在每 N 步后重建推理会话 |

---

### 6.3 测试用例 LAT-03：不同 Batch 推理延迟对比（可选）

**测试目的**

（可选）验证浏览器端 ONNX Runtime Web 在不同 batch size 下的推理延迟，评估批量推理的性能收益。

**分步执行操作**

1. 在 Console 中手动构造不同 batch size 的输入张量：

   ```javascript
   // 假设 session 和 inputName 可通过全局或 DevTools 访问
   async function benchmarkBatch(batchSize) {
     const dummyInput = new Float32Array(batchSize * 3 * 64 * 64);
     for (let i = 0; i < dummyInput.length; i++) {
       dummyInput[i] = Math.random();
     }
     const tensor = new ort.Tensor("float32", dummyInput, [batchSize, 3, 64, 64]);

     // 预热
     await session.run({ [inputName]: tensor });

     // 基准测试
     const trials = 10;
     const latencies = [];
     for (let i = 0; i < trials; i++) {
       const start = performance.now();
       await session.run({ [inputName]: tensor });
       latencies.push(performance.now() - start);
     }
     const avg = latencies.reduce((a, b) => a + b) / trials;
     console.log(`batch=${batchSize}: avg=${avg.toFixed(2)} ms, per-sample=${(avg/batchSize).toFixed(2)} ms`);
   }

   for (const bs of [1, 4, 8, 16]) {
     await benchmarkBatch(bs);
   }
   ```

2. 执行脚本，记录各 batch size 的平均延迟与单样本延迟。

**预期合格输出**

- 所有 batch size（1/4/8/16）均成功推理。
- 单样本延迟（avg / batchSize）随 batch 增大而降低或持平（体现批量推理效率）。
- batch=1 单步延迟 < 100ms（与 LAT-01 一致）。
- batch=16 总延迟不超过 batch=1 延迟的 10 倍（即有并行效率）。

**失败判定 + 修复排查方向**

| 失败现象 | 可能原因 | 修复排查方向 |
|----------|----------|--------------|
| 大 batch 推理报内存错误 | GPU 显存或浏览器内存不足 | 降低 batch size；减小输入分辨率 |
| 单样本延迟不随 batch 降低 | 后端无并行能力 | WASM 后端无 GPU 并行；切换至 WebGL 后端测试 |
| batch=16 延迟为 batch=1 的 16 倍 | 无批量优化 | 属 WASM 后端正常现象；WebGL 后端应有更好批量效率 |

---

## 7. 测试结果汇总报告模板

### 7.1 测试执行记录表

| 用例编号 | 用例名称 | 执行日期 | 执行人 | 结果 | 备注 |
|----------|----------|----------|--------|------|------|
| ONNX-01 | 导出脚本执行与模型文件生成 | | | PASS/FAIL | |
| ONNX-02 | 动态 Batch 轴验证 | | | PASS/FAIL | |
| ONNX-03 | ONNX 与 PyTorch 推理一致性校验 | | | PASS/FAIL | 最大绝对误差: |
| DEV-01 | 依赖安装 | | | PASS/FAIL | |
| DEV-02 | 开发服务器启动 | | | PASS/FAIL | |
| DEV-03 | ONNX 模型浏览器端加载验证 | | | PASS/FAIL | 推理后端: |
| VIS-01 | 全局地图实时渲染验证 | | | PASS/FAIL | |
| VIS-02 | Agent 视野 7×7 实时渲染验证 | | | PASS/FAIL | |
| VIS-03 | 控制面板单步执行功能 | | | PASS/FAIL | |
| VIS-04 | 控制面板自动播放功能 | | | PASS/FAIL | |
| VIS-05 | 速度控制功能 | | | PASS/FAIL | |
| VIS-06 | 统计面板信息显示 | | | PASS/FAIL | |
| VIS-07 | 推理流畅度与无卡顿验证 | | | PASS/FAIL | FPS: |
| BUILD-01 | 生产环境打包 | | | PASS/FAIL | 产物大小: |
| BUILD-02 | 打包产物本地预览 | | | PASS/FAIL | |
| BUILD-03 | IGA Pages 静态部署可行性验证 | | | PASS/FAIL | 部署 URL: |
| LAT-01 | 单步推理延迟基准测试 | | | PASS/FAIL | 平均延迟: ms |
| LAT-02 | 连续推理延迟稳定性测试 | | | PASS/FAIL | 退化: ms |
| LAT-03 | 不同 Batch 推理延迟对比（可选） | | | PASS/FAIL/N/A | |

### 7.2 关键指标汇总

| 指标 | 合格标准 | 实测值 | 判定 |
|------|----------|--------|------|
| ONNX-PyTorch 最大绝对误差 | < 1e-4 | | PASS/FAIL |
| 动态 Batch 支持 | batch=1/4/8/16 均通过 | | PASS/FAIL |
| 单步推理平均延迟 | < 100ms | | PASS/FAIL |
| 连续推理延迟退化（后50步 - 前50步） | < 20ms | | PASS/FAIL |
| 自动播放 FPS | ≥ 30 | | PASS/FAIL |
| 打包产物大小 | < 5MB（含模型） | | PASS/FAIL |
| IGA Pages 部署可行性 | 全功能可用 | | PASS/FAIL |

### 7.3 测试结论模板

```text
测试结论：[全部通过 / 部分通过 / 不通过]

通过用例数：XX / XX
失败用例数：XX

关键发现：
1. [发现1]
2. [发现2]
3. [发现3]

遗留问题：
1. [问题1 - 严重程度 - 建议修复方案]
2. [问题2 - 严重程度 - 建议修复方案]

测试人签名：____________    日期：____________
审核人签名：____________    日期：____________
```

### 7.4 失败用例详细记录模板

```text
用例编号：[XXX-XX]
用例名称：[名称]
执行日期：[日期]
测试环境：[浏览器/后端/模型状态]

失败现象描述：
[详细描述观察到的失败现象]

错误日志/截图：
[粘贴 Console 错误日志或附截图引用]

根因分析：
[分析失败的根本原因]

修复建议：
[给出具体修复方向和建议]

修复后复测结果：
[复测日期 / 结果 / 备注]
```

---

## 附录 A：常用排查命令速查

```bash
# 检查 Node.js / npm 版本
node --version
npm --version

# 检查 Python / PyTorch 环境
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ONNX 模型结构检查
python -c "import onnx; m=onnx.load('results/onnx/model.onnx'); onnx.checker.check_model(m); print('OK')"

# ONNX 模型输入输出检查
python -c "import onnxruntime as ort; s=ort.InferenceSession('results/onnx/model.onnx'); print([(i.name,i.shape) for i in s.get_inputs()]); print([(o.name,o.shape) for o in s.get_outputs()])"

# 前端依赖检查
cd web && npm ls react react-dom onnxruntime-web vite typescript

# 开发服务器启动
cd web && npm run dev

# 生产打包
cd web && npm run build

# 预览打包产物
cd web && npm run preview

# 静态文件服务器模拟部署
npx http-server web/dist -p 8080 --cors
```

## 附录 B：浏览器端推理后端切换配置参考

```typescript
// useAgent.ts 中配置 ONNX Runtime Web 后端
import * as ort from "onnxruntime-web";

// 方式1：优先 WebGL，回退 WASM
ort.env.wasm.wasmPaths = "/";  // WASM 文件路径
const session = await ort.InferenceSession.create("/model.onnx", {
  executionProviders: ["webgl", "wasm"],
});

// 方式2：强制 WASM 后端
const session = await ort.InferenceSession.create("/model.onnx", {
  executionProviders: ["wasm"],
});

// 检查实际使用的后端
console.log("Backend:", session.handler.constructor.name);
```

## 附录 C：推理延迟注入代码参考

```typescript
// 在 useAgent.ts 的 predict 函数中注入性能计时
const predict = async (obs: Float32Array): Promise<number> => {
  const t0 = performance.now();

  // 构造输入张量
  const tensor = new ort.Tensor("float32", obs, [1, 3, 64, 64]);

  const t1 = performance.now();

  // 执行推理
  const results = await session.run({ [inputName]: tensor });

  const t2 = performance.now();

  // 提取动作
  const logits = results[outputName].data as Float32Array;
  const action = logits.indexOf(Math.max(...logits));

  const t3 = performance.now();

  console.log(
    `[Inference] prep=${(t1 - t0).toFixed(1)}ms, ` +
    `infer=${(t2 - t1).toFixed(1)}ms, ` +
    `post=${(t3 - t2).toFixed(1)}ms, ` +
    `total=${(t3 - t0).toFixed(1)}ms`
  );

  return action;
};
```

---

> **文档结束**
> 本测试规范覆盖 ONNX 模型导出、本地开发环境启动、网页可视化功能验收、静态打包与部署、推理延迟基准五大维度，共 19 项测试用例。所有测试用例均包含测试目的、分步执行操作、预期合格输出、失败判定与修复排查方向四要素，确保测试过程可执行、可追溯、可复现。
