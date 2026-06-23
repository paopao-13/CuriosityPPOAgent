/**
 * Serverless API: POST /api/predict
 *
 * 接收观测张量, 返回动作 (服务端 ONNX 推理, 备用方案).
 *
 * 请求体:
 *   { "observation": number[] }  — 形状 (3, 64, 64) 的扁平数组, float32, 已归一化
 *
 * 响应:
 *   { "action": number, "logits": number[] }
 *
 * 环境变量:
 *   ONNX_MODEL_PATH — ONNX 模型文件路径 (默认: ./models/policy.onnx)
 *
 * 依赖: onnxruntime-node (服务端安装, 不在浏览器端 bundle 中)
 *
 * 部署:
 *   iga pages deploy --name curiosity-demo
 *   api/ 目录下的文件自动部署为 serverless functions.
 */

// ---------------------------------------------------------------------------
// 类型定义
// ---------------------------------------------------------------------------

interface PredictRequest {
  observation: number[];
}

interface PredictResponse {
  action: number;
  logits: number[];
}

interface ErrorResponse {
  error: string;
  detail?: string;
}

// ---------------------------------------------------------------------------
// ONNX Session 懒加载 (单例)
// ---------------------------------------------------------------------------

let session: any = null;

async function getSession(): Promise<any> {
  if (session) return session;

  // 动态导入 onnxruntime-node (仅在服务端可用)
  const ort = await import('onnxruntime-node');

  const modelPath = process.env.ONNX_MODEL_PATH || './models/policy.onnx';
  console.info(`[api/predict] 加载 ONNX 模型: ${modelPath}`);

  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['cpu'],
  });

  console.info(
    `[api/predict] 模型已加载, 输入: ${session.inputNames}, 输出: ${session.outputNames}`,
  );

  return session;
}

// ---------------------------------------------------------------------------
// Argmax 工具函数
// ---------------------------------------------------------------------------

function argmax(arr: Float32Array | number[]): number {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

// ---------------------------------------------------------------------------
// 请求处理 (兼容 IGA Pages / Vercel / Cloudflare Workers 格式)
// ---------------------------------------------------------------------------

export default async function handler(
  req: { method?: string; body?: any },
  res: {
    status: (code: number) => { json: (data: any) => void };
    json: (data: any) => void;
  },
): Promise<void> {
  // --- HTTP 方法检查 ---
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed, use POST' });
    return;
  }

  // --- 解析请求体 ---
  let body: PredictRequest;
  try {
    body =
      typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
  } catch {
    res.status(400).json({ error: 'Invalid JSON body' });
    return;
  }

  if (!body?.observation || !Array.isArray(body.observation)) {
    res.status(400).json({
      error: 'Missing or invalid "observation" field',
      detail: 'Expected: { "observation": number[] } with shape (3, 64, 64)',
    });
    return;
  }

  const expectedSize = 3 * 64 * 64; // 12288
  if (body.observation.length !== expectedSize) {
    res.status(400).json({
      error: `Observation size mismatch`,
      detail: `Expected ${expectedSize} elements, got ${body.observation.length}`,
    });
    return;
  }

  // --- ONNX 推理 ---
  try {
    const sess = await getSession();
    const inputName = sess.inputNames[0];

    // 创建输入张量: shape (1, 3, 64, 64), float32
    const inputData = Float32Array.from(body.observation);
    const ort = await import('onnxruntime-node');
    const tensor = new ort.Tensor('float32', inputData, [1, 3, 64, 64]);

    const output = await sess.run({ [inputName]: tensor });

    // 获取 logits
    const outputName = sess.outputNames[0];
    const logits = Array.from(output[outputName].data as Float32Array);

    const action = argmax(logits);

    const response: PredictResponse = { action, logits };
    res.status(200).json(response);
  } catch (error) {
    console.error('[api/predict] 推理失败:', error);
    const msg = error instanceof Error ? error.message : String(error);
    res.status(500).json({
      error: 'Inference failed',
      detail: msg,
    });
  }
}
