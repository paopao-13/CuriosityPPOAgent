/**
 * Canvas 渲染: MiniGrid DoorKey 全局地图 + Agent 视野
 *
 * 全局地图: 16×16 网格 × 24px = 384×384 Canvas
 * Agent 视野: 7×7 部分观测 × cellSize → 缩放显示
 */

import { GRID_SIZE, VIEW_SIZE, type EnvState, type Cell } from './env';

// ---------------------------------------------------------------------------
// 颜色映射 (RGB)
// ---------------------------------------------------------------------------

export const COLORS = {
  wall:     [150, 150, 150] as [number, number, number],
  doorClosed: [160, 82, 45] as [number, number, number],
  doorOpen: [210, 180, 140] as [number, number, number],
  key:      [255, 215, 0] as [number, number, number],
  agent:    [220, 60, 60] as [number, number, number],
  goal:     [60, 180, 75] as [number, number, number],
  empty:    [255, 255, 255] as [number, number, number],
  gridLine: [230, 230, 232] as [number, number, number],
};

function rgb([r, g, b]: [number, number, number]): string {
  return `rgb(${r}, ${g}, ${b})`;
}

function rgba([r, g, b]: [number, number, number], a: number): string {
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

// ---------------------------------------------------------------------------
// 全局地图渲染
// ---------------------------------------------------------------------------

export const CELL_SIZE = 24;
export const CANVAS_SIZE = GRID_SIZE * CELL_SIZE; // 384

/**
 * 渲染完整 16×16 网格到 Canvas (含 agent 三角形)
 */
export function renderGrid(
  ctx: CanvasRenderingContext2D,
  state: EnvState,
): void {
  const { grid, agentRow, agentCol, agentDir, carrying } = state;

  // 清空背景
  ctx.fillStyle = rgb(COLORS.empty);
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // 绘制每个格子
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const cell = grid[r][c];
      const x = c * CELL_SIZE;
      const y = r * CELL_SIZE;

      drawCell(ctx, cell, x, y, CELL_SIZE);
    }
  }

  // 绘制网格线
  ctx.strokeStyle = rgb(COLORS.gridLine);
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= GRID_SIZE; i++) {
    ctx.beginPath();
    ctx.moveTo(i * CELL_SIZE, 0);
    ctx.lineTo(i * CELL_SIZE, CANVAS_SIZE);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * CELL_SIZE);
    ctx.lineTo(CANVAS_SIZE, i * CELL_SIZE);
    ctx.stroke();
  }

  // 绘制 agent (三角形)
  drawAgent(ctx, agentCol * CELL_SIZE, agentRow * CELL_SIZE, CELL_SIZE, agentDir);

  // 如果 agent 携带钥匙, 在 agent 旁绘制小圆点
  if (carrying) {
    ctx.fillStyle = rgb(COLORS.key);
    ctx.beginPath();
    ctx.arc(
      agentCol * CELL_SIZE + CELL_SIZE * 0.8,
      agentRow * CELL_SIZE + CELL_SIZE * 0.2,
      CELL_SIZE * 0.12,
      0, Math.PI * 2,
    );
    ctx.fill();
  }

  // 如果 episode 结束, 绘制半透明遮罩
  if (state.done) {
    ctx.fillStyle = state.success
      ? rgba(COLORS.goal, 0.15)
      : rgba([0, 0, 0], 0.15);
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }
}

/**
 * 绘制单个格子
 */
function drawCell(
  ctx: CanvasRenderingContext2D,
  cell: Cell,
  x: number,
  y: number,
  size: number,
): void {
  const pad = 2;

  switch (cell.type) {
    case 'wall': {
      ctx.fillStyle = rgb(COLORS.wall);
      ctx.fillRect(x, y, size, size);
      // 墙壁纹理: 内部略深的小矩形
      ctx.fillStyle = rgba([120, 120, 120], 0.3);
      ctx.fillRect(x + pad, y + pad, size - 2 * pad, size - 2 * pad);
      break;
    }

    case 'door': {
      if (cell.doorOpen) {
        // 开门: 浅色框架
        ctx.fillStyle = rgb(COLORS.doorOpen);
        ctx.fillRect(x + 2, y, size - 4, size);
        ctx.fillStyle = rgba([0, 0, 0], 0.08);
        ctx.fillRect(x + size * 0.4, y, size * 0.2, size);
      } else {
        // 关门: 棕色
        ctx.fillStyle = rgb(COLORS.doorClosed);
        ctx.fillRect(x + 1, y + 1, size - 2, size - 2);
        // 门把手
        ctx.fillStyle = rgb([255, 215, 0]);
        ctx.beginPath();
        ctx.arc(x + size * 0.7, y + size * 0.5, size * 0.06, 0, Math.PI * 2);
        ctx.fill();
      }
      break;
    }

    case 'key': {
      // 钥匙: 黄色圆形
      ctx.fillStyle = rgb(COLORS.key);
      ctx.beginPath();
      ctx.arc(
        x + size / 2, y + size / 2,
        size * 0.28,
        0, Math.PI * 2,
      );
      ctx.fill();
      // 钥匙柄
      ctx.strokeStyle = rgb([200, 160, 0]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + size * 0.5, y + size * 0.5);
      ctx.lineTo(x + size * 0.75, y + size * 0.25);
      ctx.stroke();
      break;
    }

    case 'goal': {
      // 目标: 绿色方块 + 内圈
      ctx.fillStyle = rgb(COLORS.goal);
      ctx.fillRect(x + pad, y + pad, size - 2 * pad, size - 2 * pad);
      ctx.fillStyle = rgba([255, 255, 255], 0.3);
      ctx.beginPath();
      ctx.arc(x + size / 2, y + size / 2, size * 0.2, 0, Math.PI * 2);
      ctx.fill();
      break;
    }

    case 'empty':
    default: {
      // 空地: 白色 (已由背景填充, 这里不额外绘制)
      break;
    }
  }
}

/**
 * 绘制 agent 三角形 (朝向用三角形顶点指示)
 *
 * 方向: 0=right, 1=down, 2=left, 3=up
 */
function drawAgent(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  dir: number,
): void {
  const cx = x + size / 2;
  const cy = y + size / 2;
  const r = size * 0.35;

  // 三角形顶点 (相对中心)
  // 方向 0=right: 顶点向右
  // 方向 1=down:  顶点向下
  // 方向 2=left:  顶点向左
  // 方向 3=up:    顶点向上
  const angle = (dir * Math.PI) / 2; // 0, π/2, π, 3π/2

  // 顶点在 angle 方向, 底边两点在 angle ± 2π/3 方向
  const tip = { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  const base1 = {
    x: cx + r * Math.cos(angle + (2 * Math.PI) / 3),
    y: cy + r * Math.sin(angle + (2 * Math.PI) / 3),
  };
  const base2 = {
    x: cx + r * Math.cos(angle - (2 * Math.PI) / 3),
    y: cy + r * Math.sin(angle - (2 * Math.PI) / 3),
  };

  // 绘制三角形
  ctx.fillStyle = rgb(COLORS.agent);
  ctx.beginPath();
  ctx.moveTo(tip.x, tip.y);
  ctx.lineTo(base1.x, base1.y);
  ctx.lineTo(base2.x, base2.y);
  ctx.closePath();
  ctx.fill();

  // 边框
  ctx.strokeStyle = rgba([150, 30, 30], 0.5);
  ctx.lineWidth = 1;
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Agent 视野渲染
// ---------------------------------------------------------------------------

/**
 * 渲染 agent 的 7×7 部分观测视野到指定 Canvas.
 *
 * 视野已旋转: agent 始终在视野中心 (3,3), 朝向 "上".
 *
 * @param ctx 目标 Canvas 2D 上下文
 * @param state 当前环境状态
 * @param canvasSize 目标 Canvas 尺寸 (像素)
 */
export function renderAgentView(
  ctx: CanvasRenderingContext2D,
  state: EnvState,
  canvasSize: number,
): void {
  const { grid, agentRow, agentCol, agentDir, carrying } = state;
  const cellSize = canvasSize / VIEW_SIZE;

  // 提取 7×7 视野 (旋转后)
  const view = extractRotatedView(grid, agentRow, agentCol, agentDir);

  // 清空
  ctx.fillStyle = rgb(COLORS.empty);
  ctx.fillRect(0, 0, canvasSize, canvasSize);

  // 绘制视野中的每个格子
  for (let r = 0; r < VIEW_SIZE; r++) {
    for (let c = 0; c < VIEW_SIZE; c++) {
      const x = c * cellSize;
      const y = r * cellSize;

      if (r === 3 && c === 3) {
        // agent 位置: 先绘制地面, 再绘制 agent
        drawCell(ctx, { type: 'empty', doorOpen: false }, x, y, cellSize);
        // 绘制 agent 三角形 (视野中始终朝上)
        drawAgent(ctx, x, y, cellSize, 3); // dir=3 即 up
        // 携带钥匙标记
        if (carrying) {
          ctx.fillStyle = rgb(COLORS.key);
          ctx.beginPath();
          ctx.arc(x + cellSize * 0.8, y + cellSize * 0.2, cellSize * 0.12, 0, Math.PI * 2);
          ctx.fill();
        }
      } else {
        drawCell(ctx, view[r][c], x, y, cellSize);
      }
    }
  }

  // 网格线
  ctx.strokeStyle = rgba(COLORS.gridLine, 0.5);
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= VIEW_SIZE; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cellSize, 0);
    ctx.lineTo(i * cellSize, canvasSize);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * cellSize);
    ctx.lineTo(canvasSize, i * cellSize);
    ctx.stroke();
  }
}

/**
 * 从完整网格提取 agent 视野并旋转.
 *
 * 旋转规则 (使 agent 朝向变为 "上"):
 *   dir 3 (up):    不旋转
 *   dir 0 (right): 逆时针 90°
 *   dir 1 (down):  旋转 180°
 *   dir 2 (left):  顺时针 90°
 */
function extractRotatedView(
  grid: EnvState['grid'],
  agentRow: number,
  agentCol: number,
  agentDir: number,
): Cell[][] {
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
  const N = VIEW_SIZE;
  const rotated: Cell[][] = Array.from({ length: N }, () =>
    Array.from({ length: N }, () => ({ type: 'empty' as const, doorOpen: false })),
  );

  switch (agentDir) {
    case 3: // up: 不旋转
      return raw;

    case 0: // right: 逆时针 90°  →  new[r][c] = raw[c][N-1-r]
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          rotated[r][c] = raw[c][N - 1 - r];
        }
      }
      return rotated;

    case 1: // down: 180°  →  new[r][c] = raw[N-1-r][N-1-c]
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          rotated[r][c] = raw[N - 1 - r][N - 1 - c];
        }
      }
      return rotated;

    case 2: // left: 顺时针 90°  →  new[r][c] = raw[N-1-c][r]
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          rotated[r][c] = raw[N - 1 - c][r];
        }
      }
      return rotated;

    default:
      return raw;
  }
}
