/**
 * useAgent — ONNX 浏览器端推理 Hook
 *
 * 功能:
 *   - 加载 ONNX 模型 (onnxruntime-web, WebGL backend 优先)
 *   - 接收观测 Float32Array, 返回动作 (argmax)
 *   - 模型文件不存在时自动回退到随机策略
 *
 * 模型放置位置: public/models/policy.onnx
 * (由 scripts/export_onnx.py 导出, 复制到 web/public/models/ 即可)
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { randomAction } from '../game/actions';

// 动态导入 onnxruntime-web, 避免在 SSR 或无 WebGL 环境下崩溃
let ort: typeof import('onnxruntime-web') | null = null;
let ortLoadPromise: Promise<typeof import('onnxruntime-web')> | null = null;

async function loadOrt(): Promise<typeof import('onnxruntime-web')> {
  if (ort) return ort;
  if (ortLoadPromise) return ortLoadPromise;

  ortLoadPromise = (async () => {
    const mod = await import('onnxruntime-web');
    // 配置 WASM 路径 (使用 CDN, 确保 dev 和 production 都可用)
    try {
      mod.env.wasm.wasmPaths =
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    } catch {
      // 某些版本可能不支持 wasmPaths, 忽略错误
    }
    ort = mod;
    return mod;
  })();

  return ortLoadPromise;
}

export interface UseAgentResult {
  /** 推理函数: 传入观测张量, 返回动作编号 */
  predict: (obs: Float32Array) => Promise<number>;
  /** 模型是否正在加载 */
  loading: boolean;
  /** 是否使用随机策略 (模型未加载或加载失败) */
  useRandom: boolean;
  /** 模型加载错误信息 */
  error: string | null;
  /** 当前使用的后端名称 */
  backend: string;
}

export function useAgent(modelUrl?: string): UseAgentResult {
  const [loading, setLoading] = useState(true);
  const [useRandom, setUseRandom] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backend, setBackend] = useState('none');
  const sessionRef = useRef<any>(null);

  // 默认模型路径
  const url = modelUrl ?? '/models/policy.onnx';

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const ortMod = await loadOrt();

        // 尝试创建推理 session, 优先 WebGL, 回退到 WASM
        let sess: any = null;
        let usedBackend = 'webgl';

        try {
          sess = await ortMod.InferenceSession.create(url, {
            executionProviders: ['webgl', 'wasm'],
          });
        } catch (err1) {
          console.warn('[useAgent] WebGL 创建失败, 尝试纯 WASM:', err1);
          usedBackend = 'wasm';
          try {
            sess = await ortMod.InferenceSession.create(url, {
              executionProviders: ['wasm'],
            });
          } catch (err2) {
            throw err2;
          }
        }

        if (cancelled) return;

        sessionRef.current = sess;
        setBackend(usedBackend);
        setLoading(false);
        console.info(
          `[useAgent] 模型加载成功 (${usedBackend}), 输入: ${sess.inputNames}, 输出: ${sess.outputNames}`,
        );
      } catch (err) {
        if (cancelled) return;
        const msg =
          err instanceof Error ? err.message : '未知错误 (模型文件可能不存在)';
        console.warn('[useAgent] 模型加载失败, 回退到随机策略:', msg);
        setError(msg);
        setUseRandom(true);
        setBackend('random');
        setLoading(false);
      }
    }

    init();

    return () => {
      cancelled = true;
    };
  }, [url]);

  const predict = useCallback(
    async (obs: Float32Array): Promise<number> => {
      // 随机策略回退
      if (useRandom || !sessionRef.current) {
        return randomAction();
      }

      try {
        const ortMod = ort!;
        const session = sessionRef.current;
        const inputName = session.inputNames[0];

        // 创建输入张量: shape (1, 3, 64, 64)
        const tensor = new ortMod.Tensor('float32', obs, [1, 3, 64, 64]);
        const output = await session.run({ [inputName]: tensor });

        // 获取 logits
        const outputName = session.outputNames[0];
        const logits = output[outputName].data as Float32Array;

        // Argmax
        let maxIdx = 0;
        let maxVal = logits[0];
        for (let i = 1; i < logits.length; i++) {
          if (logits[i] > maxVal) {
            maxVal = logits[i];
            maxIdx = i;
          }
        }

        return maxIdx;
      } catch (err) {
        console.warn('[useAgent] 推理失败, 使用随机动作:', err);
        return randomAction();
      }
    },
    [useRandom],
  );

  return { predict, loading, useRandom, error, backend };
}
