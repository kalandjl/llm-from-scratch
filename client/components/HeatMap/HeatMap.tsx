"use client"
import React, { FC, useEffect, useRef, useState } from 'react';

interface Props {
  heatmapData: number[][]
  tokens: string[]
  loading: boolean
}

const formatToken = (token: string | undefined) => {
    if (token === ' ') return '␣';
    if (token === '\n') return '↵';
    return token ?? '';
};

const HeatMap: FC<Props> = ({ heatmapData, tokens, loading }) => {
    const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const n = heatmapData.length;

    // Draw each attention score as one cell (blue-500, alpha = score), same encoding as before.
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || n <= 1) return;
        const cell = Math.max(1, Math.floor(800 / n));
        canvas.width = canvas.height = n * cell;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.fillStyle = '#09090b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        heatmapData.forEach((row, r) => row.forEach((score, c) => {
            const alpha = Math.max(0, Math.min(1, score));
            if (alpha === 0) return;
            ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
            ctx.fillRect(c * cell, r * cell, cell, cell);
        }));
    }, [heatmapData, n]);

    if (loading) {
        return (
            <div className="flex items-center justify-center rounded-2xl border border-white/[0.08] bg-zinc-900/60 py-16">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-400 border-t-transparent" />
            </div>
        )
    }

    if (n <= 1) return (<></>)

    const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const col = Math.floor(((e.clientX - rect.left) / rect.width) * n);
        const row = Math.floor(((e.clientY - rect.top) / rect.height) * n);
        if (row >= 0 && col >= 0 && row < n && col < n) setHoveredCell({ row, col });
    };

    const score = hoveredCell ? heatmapData[hoveredCell.row]?.[hoveredCell.col] : undefined;

    return (
        <div className="rounded-2xl border border-white/[0.08] bg-zinc-900/60 p-5 sm:p-6">
            <h2 className="text-sm font-medium text-zinc-300">Attention Heatmap</h2>

            <div className="mt-3 flex h-8 items-center font-mono text-xs text-zinc-500">
                {hoveredCell ? (
                    <div className="rounded-md bg-white/[0.04] px-2.5 py-1">
                        <span className="text-blue-400">&apos;{formatToken(tokens[hoveredCell.row])}&apos;</span>
                        <span> (row {hoveredCell.row})</span>
                        <span className="mx-2">→</span>
                        <span className="text-emerald-400">&apos;{formatToken(tokens[hoveredCell.col])}&apos;</span>
                        <span> (col {hoveredCell.col})</span>
                        {score !== undefined && <span className="ml-3 text-zinc-300">{score.toFixed(3)}</span>}
                    </div>
                ) : (
                    <span>Hover over a cell to see details</span>
                )}
            </div>

            <canvas
                ref={canvasRef}
                className="mt-2 aspect-square w-full cursor-crosshair rounded-lg border border-white/[0.08] [image-rendering:pixelated]"
                onMouseMove={onMove}
                onMouseLeave={() => setHoveredCell(null)}
            />
        </div>
    );
};

export default HeatMap
