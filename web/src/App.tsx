/**
 * App — 主页面组件
 *
 * 布局:
 *   ┌──────────────────────────────────────────┐
 *   │               标题 + 描述                  │
 *   ├──────────────────┬───────────────────────┤
 *   │                  │   Agent 视野 (7×7)    │
 *   │   全局地图        ├───────────────────────┤
 *   │   (16×16)        │   控制面板             │
 *   │                  ├───────────────────────┤
 *   │                  │   统计面板             │
 *   └──────────────────┴───────────────────────┘
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { type EnvState, reset, step } from './game/env';
import { getModelInput } from './game/obs';
import { useAgent } from './hooks/useAgent';
import { GridCanvas } from './components/GridCanvas';
import { AgentView } from './components/AgentView';
import { ControlPanel } from './components/ControlPanel';
import { StatsPanel } from './components/StatsPanel';

export default function App() {
  // --- 环境状态 ---
  const [envState, setEnvState] = useState<EnvState>(() => reset(42));
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(200);

  // --- ONNX 推理 Hook ---
  const { predict, loading, useRandom, backend } = useAgent();

  // --- Refs (避免 auto-play 闭包中拿到旧值) ---
  const stateRef = useRef(envState);
  const playingRef = useRef(isPlaying);
  const predictRef = useRef(predict);

  useEffect(() => { stateRef.current = envState; }, [envState]);
  useEffect(() => { playingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { predictRef.current = predict; }, [predict]);

  // --- 单步执行 ---
  const handleStep = useCallback(async () => {
    const current = stateRef.current;
    if (current.done) return;

    // 构建观测 → 模型推理 → 执行动作
    const obs = getModelInput(current);
    const action = await predictRef.current(obs);
    const result = step(current, action);
    setEnvState(result.state);

    if (result.done) {
      playingRef.current = false;
      setIsPlaying(false);
    }
  }, []);

  // --- 自动播放 ---
  useEffect(() => {
    if (!isPlaying) return;

    let timer: ReturnType<typeof setTimeout>;

    const runStep = async () => {
      const current = stateRef.current;
      if (current.done) {
        playingRef.current = false;
        setIsPlaying(false);
        return;
      }

      await handleStep();

      // 如果还在播放, 安排下一步
      if (playingRef.current && !stateRef.current.done) {
        timer = setTimeout(runStep, speed);
      }
    };

    // 首次延迟启动
    timer = setTimeout(runStep, speed);

    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [isPlaying, speed, handleStep]);

  // --- 切换自动播放 ---
  const handleToggleAuto = useCallback(() => {
    if (stateRef.current.done && !isPlaying) return;
    setIsPlaying((prev) => !prev);
  }, [isPlaying]);

  // --- 重置环境 ---
  const handleReset = useCallback(() => {
    setIsPlaying(false);
    playingRef.current = false;
    setEnvState(reset(Date.now() % 100000));
  }, []);

  // --- 速度变化 ---
  const handleSpeedChange = useCallback((newSpeed: number) => {
    setSpeed(newSpeed);
  }, []);

  return (
    <div className="app">
      {/* 标题区 */}
      <header className="app-header">
        <h1 className="app-title">Curiosity PPO</h1>
        <p className="app-subtitle">
          ICM + RND 好奇心驱动 PPO — MiniGrid DoorKey 智能体演示
        </p>
      </header>

      {/* 主内容区 */}
      <main className="app-main">
        {/* 左侧: 全局地图 */}
        <section className="card map-card">
          <div className="card-header">
            <h2 className="card-title">全局地图</h2>
            <span className="card-subtitle">16 × 16 网格 (上帝视角)</span>
          </div>
          <div className="canvas-wrapper">
            <GridCanvas state={envState} />
          </div>
          <div className="legend">
            <div className="legend-item">
              <span className="legend-color" style={{ background: 'rgb(220, 60, 60)' }} />
              <span>Agent</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ background: 'rgb(160, 82, 45)' }} />
              <span>门</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ background: 'rgb(255, 215, 0)' }} />
              <span>钥匙</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ background: 'rgb(60, 180, 75)' }} />
              <span>目标</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ background: 'rgb(150, 150, 150)' }} />
              <span>墙壁</span>
            </div>
          </div>
        </section>

        {/* 右侧: 控制面板 + 视野 + 统计 */}
        <aside className="side-panel">
          {/* Agent 视野 */}
          <section className="card view-card">
            <div className="card-header">
              <h2 className="card-title">Agent 视野</h2>
              <span className="card-subtitle">7 × 7 部分观测</span>
            </div>
            <div className="agent-view-wrapper">
              <AgentView state={envState} size={128} />
            </div>
            <p className="view-hint">
              视野已旋转, Agent 始终朝上. 模型输入: 64×64×3 归一化张量.
            </p>
          </section>

          {/* 控制面板 */}
          <section className="card control-card">
            <div className="card-header">
              <h2 className="card-title">控制面板</h2>
            </div>
            {loading && (
              <div className="loading-hint">正在加载 ONNX 模型...</div>
            )}
            <ControlPanel
              onStep={handleStep}
              onToggleAuto={handleToggleAuto}
              onReset={handleReset}
              isPlaying={isPlaying}
              isDone={envState.done}
              speed={speed}
              onSpeedChange={handleSpeedChange}
            />
          </section>

          {/* 统计面板 */}
          <section className="card stats-card">
            <div className="card-header">
              <h2 className="card-title">运行统计</h2>
            </div>
            <StatsPanel
              state={envState}
              backend={backend}
              useRandom={useRandom}
            />
          </section>
        </aside>
      </main>

      {/* 底部说明 */}
      <footer className="app-footer">
        <p>
          任务: 拾取钥匙 → 用钥匙开门 → 到达绿色目标.
          模型缺失时自动回退到随机策略, 可体验游戏机制.
        </p>
        <p className="footer-tech">
          Vite + React + TypeScript + ONNX Runtime Web | ICM+RND Curiosity PPO
        </p>
      </footer>
    </div>
  );
}
