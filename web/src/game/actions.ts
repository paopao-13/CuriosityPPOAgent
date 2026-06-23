/**
 * MiniGrid 动作枚举和映射
 *
 * 与 MiniGrid / gymnasium 的 Actions 枚举保持一致:
 *   0 = turn left, 1 = turn right, 2 = forward,
 *   3 = pickup,   4 = drop,       5 = toggle, 6 = done
 */
export enum Action {
  LEFT = 0,    // 逆时针旋转 90°
  RIGHT = 1,   // 顺时针旋转 90°
  FORWARD = 2, // 朝当前方向前进一步
  PICKUP = 3,  // 拾取前方格子的物品
  DROP = 4,    // 在前方格子放置携带的物品
  TOGGLE = 5,  // 切换门/箱子状态 (开门/关门)
  DONE = 6,    // 主动结束当前 episode
}

/** 动作总数 */
export const NUM_ACTIONS = 7;

/** 动作中文名称映射 */
export const ACTION_NAMES: Record<number, string> = {
  [Action.LEFT]: '左转',
  [Action.RIGHT]: '右转',
  [Action.FORWARD]: '前进',
  [Action.PICKUP]: '拾取',
  [Action.DROP]: '放下',
  [Action.TOGGLE]: '开关门',
  [Action.DONE]: '完成',
};

/** 动作英文短名 (用于 UI 标签) */
export const ACTION_SHORT: Record<number, string> = {
  [Action.LEFT]: 'turn-left',
  [Action.RIGHT]: 'turn-right',
  [Action.FORWARD]: 'forward',
  [Action.PICKUP]: 'pickup',
  [Action.DROP]: 'drop',
  [Action.TOGGLE]: 'toggle',
  [Action.DONE]: 'done',
};

/** 随机采样一个动作 */
export function randomAction(): number {
  return Math.floor(Math.random() * NUM_ACTIONS);
}
