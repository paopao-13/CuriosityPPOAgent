/**
 * GridCanvas — 全局地图 Canvas 组件
 *
 * 接收环境状态, 渲染 16×16 完整网格到 384×384 Canvas.
 */

import { useRef, useEffect } from 'react';
import { type EnvState } from '../game/env';
import { renderGrid, CANVAS_SIZE } from '../game/render';

interface GridCanvasProps {
  state: EnvState;
}

export function GridCanvas({ state }: GridCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 启用高 DPI 渲染
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_SIZE * dpr;
    canvas.height = CANVAS_SIZE * dpr;
    ctx.scale(dpr, dpr);

    renderGrid(ctx, state);
  }, [state]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: CANVAS_SIZE,
        height: CANVAS_SIZE,
        borderRadius: 12,
        display: 'block',
      }}
    />
  );
}
