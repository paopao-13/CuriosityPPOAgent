/**
 * StatsPanel — 统计面板组件
 *
 * 显示:
 *   - 步数
 *   - 累计奖励
 *   - 是否成功
 *   - 钥匙获取状态
 *   - 门开启状态
 *   - 最近动作 / 消息
 *   - Agent 朝向
 */

import { type EnvState, MAX_STEPS, dirName } from '../game/env';

interface StatsPanelProps {
  state: EnvState;
  /** 当前使用的推理后端 */
  backend: string;
  /** 是否使用随机策略 */
  useRandom: boolean;
}

export function StatsPanel({ state, backend, useRandom }: StatsPanelProps) {
  const stats: { label: string; value: string; highlight?: boolean }[] = [
    {
      label: '步数',
      value: `${state.stepCount} / ${MAX_STEPS}`,
    },
    {
      label: '累计奖励',
      value: state.totalReward.toFixed(2),
      highlight: state.totalReward > 0,
    },
    {
      label: '状态',
      value: state.done
        ? state.success
          ? '成功!'
          : '失败'
        : '进行中',
      highlight: state.success,
    },
    {
      label: '钥匙',
      value: state.carrying ? '已获取' : state.keyPos ? '未获取' : '已获取',
      highlight: state.carrying,
    },
    {
      label: '门',
      value: state.grid[state.doorPos[0]][state.doorPos[1]].doorOpen
        ? '已打开'
        : '关闭',
      highlight: state.grid[state.doorPos[0]][state.doorPos[1]].doorOpen,
    },
    {
      label: '朝向',
      value: dirName(state.agentDir),
    },
  ];

  return (
    <div className="stats-panel">
      <div className="stats-grid">
        {stats.map((s) => (
          <div
            key={s.label}
            className={`stat-item ${s.highlight ? 'stat-highlight' : ''}`}
          >
            <span className="stat-label">{s.label}</span>
            <span className="stat-value">{s.value}</span>
          </div>
        ))}
      </div>

      {state.lastAction && (
        <div className="last-action">
          <span className="action-label">最近动作:</span>
          <span className="action-value">{state.lastAction}</span>
          <span className="action-message">{state.lastMessage}</span>
        </div>
      )}

      <div className="backend-info">
        <span className="backend-label">推理后端:</span>
        <span className={`backend-value ${useRandom ? 'backend-random' : 'backend-model'}`}>
          {useRandom ? '随机策略' : `ONNX (${backend})`}
        </span>
      </div>
    </div>
  );
}
