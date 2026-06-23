/**
 * MiniGrid-DoorKey-16x16 环境模拟器 (TypeScript 复刻)
 *
 * 完整复刻 MiniGrid DoorKey 环境:
 *   - 16×16 网格, 墙壁包围
 *   - 中间垂直墙将网格分为两个房间, 墙上有门
 *   - 钥匙在左房间, 目标在右房间
 *   - 7 种动作: turn-left, turn-right, forward, pickup, drop, toggle, done
 *   - 碰撞检测: 不能穿墙 / 不能穿过关闭的门
 *   - 成功条件: 拿钥匙 → 开门 → 到达目标
 *
 * 坐标系: grid[row][col], row 向下增大, col 向右增大
 * 方向:    0=right, 1=down, 2=left, 3=up
 */

// ---------------------------------------------------------------------------
// 类型定义
// ---------------------------------------------------------------------------

export type CellType = 'empty' | 'wall' | 'door' | 'key' | 'goal';

export interface Cell {
  type: CellType;
  /** 门是否打开 (仅对 type='door' 有效) */
  doorOpen: boolean;
}

export type Grid = Cell[][];

export interface EnvState {
  /** 静态网格 (不含 agent, agent 位置由 agentRow/Col 单独跟踪) */
  grid: Grid;
  /** agent 当前行 */
  agentRow: number;
  /** agent 当前列 */
  agentCol: number;
  /** agent 朝向: 0=right, 1=down, 2=left, 3=up */
  agentDir: number;
  /** 是否携带钥匙 */
  carrying: boolean;
  /** episode 是否结束 */
  done: boolean;
  /** 是否成功到达目标 */
  success: boolean;
  /** 已执行步数 */
  stepCount: number;
  /** 累计奖励 */
  totalReward: number;
  /** 钥匙位置 (已被拾取则为 null) */
  keyPos: [number, number] | null;
  /** 目标位置 */
  goalPos: [number, number];
  /** 门位置 */
  doorPos: [number, number];
  /** 最近一步的动作名称 */
  lastAction: string;
  /** 最近一步的提示消息 */
  lastMessage: string;
}

export interface StepResult {
  state: EnvState;
  reward: number;
  done: boolean;
  info: { success: boolean; message: string };
}

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

export const GRID_SIZE = 16;
export const VIEW_SIZE = 7;
export const MAX_STEPS = 256;

/**
 * 方向向量: [drow, dcol]
 *   0 = right → (0, +1)
 *   1 = down  → (+1, 0)
 *   2 = left  → (0, -1)
 *   3 = up    → (-1, 0)
 */
const DIR_VECTORS: [number, number][] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
];

// ---------------------------------------------------------------------------
// 随机数 (可种子化)
// ---------------------------------------------------------------------------

/** mulberry32 PRNG — 轻量可种子化伪随机数生成器 */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// 网格构建
// ---------------------------------------------------------------------------

function createEmptyGrid(): Grid {
  const grid: Grid = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    const row: Cell[] = [];
    for (let c = 0; c < GRID_SIZE; c++) {
      const isBorder =
        r === 0 || r === GRID_SIZE - 1 || c === 0 || c === GRID_SIZE - 1;
      row.push({ type: isBorder ? 'wall' : 'empty', doorOpen: false });
    }
    grid.push(row);
  }
  return grid;
}

/** 在指定区域内随机选一个空格子 */
function randomEmptyCell(
  grid: Grid,
  rng: () => number,
  rowMin: number,
  rowMax: number,
  colMin: number,
  colMax: number,
): [number, number] {
  while (true) {
    const r = rowMin + Math.floor(rng() * (rowMax - rowMin + 1));
    const c = colMin + Math.floor(rng() * (colMax - colMin + 1));
    if (grid[r][c].type === 'empty') return [r, c];
  }
}

// ---------------------------------------------------------------------------
// 环境接口
// ---------------------------------------------------------------------------

/**
 * 重置环境, 生成新的 DoorKey 关卡.
 *
 * @param seed 随机种子 (可选, 不传则使用 Math.random)
 */
export function reset(seed?: number): EnvState {
  const rng = seed !== undefined ? makeRng(seed) : Math.random;

  const grid = createEmptyGrid();

  // --- 中间垂直墙 ---
  const wallCol = Math.floor(GRID_SIZE / 2); // 8
  for (let r = 1; r < GRID_SIZE - 1; r++) {
    grid[r][wallCol] = { type: 'wall', doorOpen: false };
  }

  // --- 门: 在墙上随机位置开口 ---
  const doorRow = 1 + Math.floor(rng() * (GRID_SIZE - 2));
  grid[doorRow][wallCol] = { type: 'door', doorOpen: false };

  // --- 钥匙: 左房间随机空格 ---
  const [keyRow, keyCol] = randomEmptyCell(
    grid, rng, 1, GRID_SIZE - 2, 1, wallCol - 1,
  );
  grid[keyRow][keyCol] = { type: 'key', doorOpen: false };

  // --- 目标: 右房间随机空格 ---
  const [goalRow, goalCol] = randomEmptyCell(
    grid, rng, 1, GRID_SIZE - 2, wallCol + 1, GRID_SIZE - 2,
  );
  grid[goalRow][goalCol] = { type: 'goal', doorOpen: false };

  // --- agent: 左房间随机空格 ---
  const [agentRow, agentCol] = randomEmptyCell(
    grid, rng, 1, GRID_SIZE - 2, 1, wallCol - 1,
  );

  return {
    grid,
    agentRow,
    agentCol,
    agentDir: Math.floor(rng() * 4),
    carrying: false,
    done: false,
    success: false,
    stepCount: 0,
    totalReward: 0,
    keyPos: [keyRow, keyCol],
    goalPos: [goalRow, goalCol],
    doorPos: [doorRow, wallCol],
    lastAction: '',
    lastMessage: '环境已重置',
  };
}

/**
 * 深拷贝状态 (含网格)
 */
function cloneState(state: EnvState): EnvState {
  return {
    ...state,
    grid: state.grid.map((row) => row.map((cell) => ({ ...cell }))),
  };
}

/** 获取 agent 前方格子坐标 */
function frontCell(state: EnvState): [number, number] {
  const [dr, dc] = DIR_VECTORS[state.agentDir];
  return [state.agentRow + dr, state.agentCol + dc];
}

/** 判断坐标是否在网格内 */
function inBounds(r: number, c: number): boolean {
  return r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE;
}

/**
 * 执行一步动作.
 *
 * @param state 当前环境状态 (不会被修改)
 * @param action 动作编号 0-6
 * @returns 新状态 + 奖励 + 是否结束 + 信息
 */
export function step(state: EnvState, action: number): StepResult {
  if (state.done) {
    return {
      state,
      reward: 0,
      done: true,
      info: { success: false, message: 'Episode 已结束' },
    };
  }

  const s = cloneState(state);
  let reward = -0.01; // 步惩罚
  let message = '';

  switch (action) {
    // ----- 左转 (逆时针) -----
    case 0: {
      s.agentDir = (s.agentDir + 3) % 4;
      message = '左转';
      break;
    }

    // ----- 右转 (顺时针) -----
    case 1: {
      s.agentDir = (s.agentDir + 1) % 4;
      message = '右转';
      break;
    }

    // ----- 前进 -----
    case 2: {
      const [dr, dc] = DIR_VECTORS[s.agentDir];
      const nr = s.agentRow + dr;
      const nc = s.agentCol + dc;

      if (!inBounds(nr, nc)) {
        message = '撞墙';
        break;
      }
      const target = s.grid[nr][nc];
      if (target.type === 'wall') {
        message = '撞墙';
        break;
      }
      if (target.type === 'door' && !target.doorOpen) {
        message = '门未打开';
        break;
      }

      s.agentRow = nr;
      s.agentCol = nc;
      message = '前进';

      // 到达目标
      if (nr === s.goalPos[0] && nc === s.goalPos[1]) {
        s.done = true;
        s.success = true;
        reward = 1.0;
        message = '成功到达目标!';
      }
      break;
    }

    // ----- 拾取 -----
    case 3: {
      const [fr, fc] = frontCell(s);
      if (inBounds(fr, fc) && s.grid[fr][fc].type === 'key' && !s.carrying) {
        s.carrying = true;
        s.grid[fr][fc] = { type: 'empty', doorOpen: false };
        s.keyPos = null;
        message = '拾取钥匙';
      } else {
        message = '前方无可拾取物品';
      }
      break;
    }

    // ----- 放下 -----
    case 4: {
      if (!s.carrying) {
        message = '未携带任何物品';
        break;
      }
      const [fr, fc] = frontCell(s);
      if (inBounds(fr, fc) && s.grid[fr][fc].type === 'empty') {
        s.carrying = false;
        s.grid[fr][fc] = { type: 'key', doorOpen: false };
        s.keyPos = [fr, fc];
        message = '放下钥匙';
      } else {
        message = '前方无法放置';
      }
      break;
    }

    // ----- 开关门 -----
    case 5: {
      const [fr, fc] = frontCell(s);
      if (!inBounds(fr, fc) || s.grid[fr][fc].type !== 'door') {
        message = '前方无门';
        break;
      }
      const door = s.grid[fr][fc];
      if (!door.doorOpen) {
        if (s.carrying) {
          door.doorOpen = true;
          message = '门已打开';
        } else {
          message = '需要钥匙才能开门';
        }
      } else {
        door.doorOpen = false;
        message = '门已关闭';
      }
      break;
    }

    // ----- 结束 episode -----
    case 6: {
      s.done = true;
      message = '主动结束';
      break;
    }

    default: {
      message = `未知动作: ${action}`;
    }
  }

  s.stepCount = state.stepCount + 1;
  s.totalReward = state.totalReward + reward;
  s.lastAction = ['', '左转', '右转', '前进', '拾取', '放下', '开关门', '结束'][action] ?? '';
  s.lastMessage = message;

  // 达到最大步数
  if (s.stepCount >= MAX_STEPS && !s.done) {
    s.done = true;
    s.lastMessage = '达到最大步数';
  }

  return {
    state: s,
    reward,
    done: s.done,
    info: { success: s.success, message },
  };
}

/**
 * 获取 agent 视野中心坐标 (用于调试)
 */
export function getAgentFront(state: EnvState): [number, number] {
  return frontCell(state);
}

/**
 * 方向名称
 */
export function dirName(dir: number): string {
  return ['右', '下', '左', '上'][dir] ?? '?';
}
