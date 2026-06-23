/**
 * 观测构建: 从环境状态生成 ONNX 模型输入张量
 *
 * 管线 (与训练侧一致):
 *   1. 从完整网格提取 agent 视野 7×7 窗口
 *   2. 根据朝向旋转 (agent 始终朝 "上")
 *   3. 转换为 RGB (7×7×3)
 *   4. 最近邻缩放到 64×64×3
 *   5. 归一化到 [0, 1] (除以 255)
 *   6. 转置 HWC → CHW (用于 ONNX 输入)
 */

import { GRID_SIZE, VIEW_SIZE, type EnvState, type Cell } from './env';
import { COLORS } from './render';

/** 模型输入尺寸 */
export const OBS_SIZE = 64;

// ---------------------------------------------------------------------------
// 颜色映射: CellType → RGB
// ---------------------------------------------------------------------------

function cellToRgb(cell: Cell): [number, number, number] {
  switch (cell.type) {
    case 'wall':
      return COLORS.wall;
    case 'door':
      return cell.doorOpen ? COLORS.doorOpen : COLORS.doorClosed;
    case 'key':
      return COLORS.key;
    case 'goal':
      return COLORS.goal;
    case 'empty':
    default:
      return COLORS.empty;
  }
}

// ---------------------------------------------------------------------------
// 视野提取 + 旋转 (与 render.ts 中逻辑一致, 独立实现以解耦)
// ---------------------------------------------------------------------------

/**
 * 提取 7×7 旋转视野.
 *
 * 旋转规则 (使 agent 朝向变为 "上"):
 *   dir 3 (up):    不旋转
 *   dir 0 (right): 逆时针 90°
 *   dir 1 (down):  旋转 180°
 *   dir 2 (left):  顺时针 90°
 */
export function extractView(state: EnvState): Cell[][] {
  const { grid, agentRow, agentCol, agentDir } = state;
  const N = VIEW_SIZE;

  // 1. 提取 7×7 窗口 (agent 在中心 (3,3))
  const raw: Cell[][] = [];
  for (let dr = -3; dr <= 3; dr++) {
    const row: Cell[] = [];
    for (let dc = -3; dc <= 3; dc++) {
      const r = agentRow + dr;
      const c = agentCol + dc;
      if (r < 0 || r >= GRID_SIZE || c < 0 || c >= GRID_SIZE) {
        row.push({ type: 'wall', doorOpen: false });
      } else {
        row.push({ ...grid[r][c] });
      }
    }
    raw.push(row);
  }

  // 2. 旋转
  const rotated: Cell[][] = Array.from({ length: N }, () =>
    Array.from({ length: N }, () => ({ type: 'empty' as const, doorOpen: false })),
  );

  switch (agentDir) {
    case 3:
      return raw;
    case 0: // CCW 90°
      for (let r = 0; r < N; r++)
        for (let c = 0; c < N; c++)
          rotated[r][c] = raw[c][N - 1 - r];
      return rotated;
    case 1: // 180°
      for (let r = 0; r < N; r++)
        for (let c = 0; c < N; c++)
          rotated[r][c] = raw[N - 1 - r][N - 1 - c];
      return rotated;
    case 2: // CW 90°
      for (let r = 0; r < N; r++)
        for (let c = 0; c < N; c++)
          rotated[r][c] = raw[N - 1 - c][r];
      return rotated;
    default:
      return raw;
  }
}

// ---------------------------------------------------------------------------
// 观测张量构建
// ---------------------------------------------------------------------------

/**
 * 生成 7×7×3 RGB 观测 (uint8).
 *
 * Agent 位于中心 (3,3), 用红色绘制.
 */
export function getRgbObservation(state: EnvState): Uint8Array {
  const view = extractView(state);
  const obs = new Uint8Array(VIEW_SIZE * VIEW_SIZE * 3);

  for (let r = 0; r < VIEW_SIZE; r++) {
    for (let c = 0; c < VIEW_SIZE; c++) {
      const idx = (r * VIEW_SIZE + c) * 3;
      if (r === 3 && c === 3) {
        // Agent: 红色
        [obs[idx], obs[idx + 1], obs[idx + 2]] = COLORS.agent;
      } else {
        const [R, G, B] = cellToRgb(view[r][c]);
        obs[idx] = R;
        obs[idx + 1] = G;
        obs[idx + 2] = B;
      }
    }
  }

  return obs;
}

/**
 * 最近邻缩放: 将 7×7×3 图像缩放到 64×64×3.
 */
function nearestNeighborResize(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8Array {
  const dst = new Uint8Array(dstW * dstH * 3);
  for (let dy = 0; dy < dstH; dy++) {
    const sy = Math.min(srcH - 1, Math.floor((dy * srcH) / dstH));
    for (let dx = 0; dx < dstW; dx++) {
      const sx = Math.min(srcW - 1, Math.floor((dx * srcW) / dstW));
      const sIdx = (sy * srcW + sx) * 3;
      const dIdx = (dy * dstW + dx) * 3;
      dst[dIdx] = src[sIdx];
      dst[dIdx + 1] = src[sIdx + 1];
      dst[dIdx + 2] = src[sIdx + 2];
    }
  }
  return dst;
}

/**
 * 构建完整的 ONNX 模型输入张量.
 *
 * 输出 Float32Array, 形状 (1, 3, 64, 64), 已归一化到 [0, 1].
 *
 * 与训练侧管线对齐:
 *   ImgObsWrapper (7×7×3) → ResizeObs(64×64) → ObsToFloat32 → HWC→CHW
 */
export function getModelInput(state: EnvState): Float32Array {
  // 1. 7×7×3 RGB
  const rgb7 = getRgbObservation(state);

  // 2. 最近邻缩放到 64×64×3
  const rgb64 = nearestNeighborResize(rgb7, VIEW_SIZE, VIEW_SIZE, OBS_SIZE, OBS_SIZE);

  // 3. 归一化 + HWC → CHW
  const tensor = new Float32Array(3 * OBS_SIZE * OBS_SIZE);
  const planeSize = OBS_SIZE * OBS_SIZE;
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < planeSize; i++) {
      tensor[c * planeSize + i] = rgb64[i * 3 + c] / 255.0;
    }
  }

  return tensor;
}

/**
 * 获取 64×64 RGB 图像数据 (用于 Canvas 可视化).
 */
export function getResizedRgb(state: EnvState): Uint8ClampedArray {
  const rgb7 = getRgbObservation(state);
  const rgb64 = nearestNeighborResize(rgb7, VIEW_SIZE, VIEW_SIZE, OBS_SIZE, OBS_SIZE);

  // 转换为 ImageData 格式 (RGBA)
  const imageData = new Uint8ClampedArray(OBS_SIZE * OBS_SIZE * 4);
  for (let i = 0; i < OBS_SIZE * OBS_SIZE; i++) {
    imageData[i * 4] = rgb64[i * 3];
    imageData[i * 4 + 1] = rgb64[i * 3 + 1];
    imageData[i * 4 + 2] = rgb64[i * 3 + 2];
    imageData[i * 4 + 3] = 255;
  }
  return imageData;
}
