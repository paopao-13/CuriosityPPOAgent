/**
 * AgentView — Agent 视野 Canvas 组件
 *
 * 显示 7×7 部分观测窗口 (放大到 128×128).
 * 视野已旋转: agent 始终在中心, 朝向 "上".
 */

import { useRef, useEffect } from 'react';
import { type EnvState } from '../game/env';
import { renderAgentView } from '../game/render';

interface AgentViewProps {
  state: EnvState;
  /** 显示尺寸 (像素), 默认 128 */
  size?: number;
}

export function AgentView({ state, size = 128 }: AgentViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 高 DPI 渲染
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    renderAgentView(ctx, state, size);
  }, [state, size]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: size,
        height: size,
        borderRadius: 8,
        display: 'block',
        imageRendering: 'pixelated',
      }}
    />
  );
}
