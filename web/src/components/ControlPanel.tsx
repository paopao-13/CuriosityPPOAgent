/**
 * ControlPanel — 控制面板组件
 *
 * 包含:
 *   - Step 按钮: 单步执行
 *   - Auto Play 按钮: 自动运行 / 暂停
 *   - Reset 按钮: 重置环境
 *   - Speed 滑块: 调整自动播放速度 (50ms - 1000ms)
 */

interface ControlPanelProps {
  onStep: () => void;
  onToggleAuto: () => void;
  onReset: () => void;
  isPlaying: boolean;
  isDone: boolean;
  speed: number;
  onSpeedChange: (speed: number) => void;
}

export function ControlPanel({
  onStep,
  onToggleAuto,
  onReset,
  isPlaying,
  isDone,
  speed,
  onSpeedChange,
}: ControlPanelProps) {
  return (
    <div className="control-panel">
      <div className="control-buttons">
        <button
          className="btn btn-primary"
          onClick={onStep}
          disabled={isPlaying || isDone}
        >
          单步执行
        </button>
        <button
          className="btn btn-primary"
          onClick={onToggleAuto}
          disabled={isDone && !isPlaying}
        >
          {isPlaying ? '暂停' : '自动播放'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={onReset}
        >
          重置
        </button>
      </div>

      <div className="speed-control">
        <label className="speed-label">
          速度
          <span className="speed-value">{speed}ms</span>
        </label>
        <input
          type="range"
          min={50}
          max={1000}
          step={50}
          value={speed}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
          className="speed-slider"
        />
        <div className="speed-marks">
          <span>快</span>
          <span>慢</span>
        </div>
      </div>
    </div>
  );
}
