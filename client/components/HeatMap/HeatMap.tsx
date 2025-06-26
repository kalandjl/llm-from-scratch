"use client"
import React, { FC, useEffect, useState } from 'react';

interface Props {
  heatmapData: number[][]
  tokens: string[]
  loading: boolean
}

const HeatMap: FC<Props> = ({ heatmapData, tokens, loading }) => {
    const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

    const formatToken = (token: string) => {
        if (token === ' ') return '␣'; // Show space as a visible character
        if (token === '\n') return '↵'; // Show newline as a visible character
        return token;
    };



    if (heatmapData.length === 1) return (<></>)

    if (loading) {
        return (
            <div className="flex items-center justify-center py-16">
                <div className="h-6 w-6 border-4 border-t-transparent border-white rounded-full animate-spin" />
            </div>
        )
    }

    return (
        <div className="p-4 sm:p-6 bg-gray-900 text-white rounded-xl shadow-lg w-full overflow-x-auto">
        <h2 className="text-xl font-bold mb-2 text-center">Attention Heatmap</h2>
        
        {/* --- Coordinate Display Area --- */}
        <div className="h-8 mb-4 text-center text-sm text-gray-400 font-mono flex items-center justify-center">
            {hoveredCell ? (
            <div className="bg-gray-800 px-3 py-1 rounded-md">
                <span className="text-blue-400 font-bold">'{formatToken(tokens[hoveredCell.row])}'</span>
                <span className="text-gray-500"> (row {hoveredCell.row})</span>
                <span className="text-gray-500 mx-2">→</span>
                <span className="text-green-400 font-bold">'{formatToken(tokens[hoveredCell.col])}'</span>
                <span className="text-gray-500"> (col {hoveredCell.col})</span>
            </div>
            ) : (
            <span>Hover over a cell to see details</span>
            )}
        </div>

        <div className="relative w-full aspect-square overflow-y-auto border border-gray-800 rounded-lg">
            <table 
            className="min-w-full border-collapse"
            onMouseLeave={() => setHoveredCell(null)} // Clear when mouse leaves the table
            >
            {/* Table Header (Column Tokens) */}
            <thead>
                <tr>
                <th className="sticky top-0 z-10 p-2 border border-gray-700 bg-gray-800 min-w-[40px]"></th>
                {tokens.map((token, index) => (
                    <th key={`col-${index}`} className="sticky top-0 z-10 p-2 text-sm border border-gray-700 bg-gray-800 min-w-[40px]">
                    {formatToken(token)}
                    </th>
                ))}
                </tr>
            </thead>
            
            {/* Table Body (Rows and Heatmap Cells) */}
            <tbody>
                {heatmapData.map((row, rowIndex) => (
                <tr key={`row-${rowIndex}`}>
                    {/* Row Header (Row Token) */}
                    <th className="p-2 text-sm font-bold border border-gray-700 bg-gray-800">
                    {formatToken(tokens[rowIndex])}
                    </th>
                    
                    {/* Heatmap Cells */}
                    {row.map((score, colIndex) => {
                    const alpha = Math.max(0, Math.min(1, score));
                    const backgroundColor = `rgba(59, 130, 246, ${alpha})`; // Tailwind's blue-500

                    return (
                        <td
                        key={`cell-${rowIndex}-${colIndex}`}
                        className="relative p-0 border border-gray-700 h-10 w-10 text-center transition-transform duration-150 ease-in-out hover:scale-110"
                        style={{ backgroundColor }}
                        onMouseEnter={() => setHoveredCell({ row: rowIndex, col: colIndex })}
                        title={`'${formatToken(tokens[rowIndex])}' paying attention to '${formatToken(tokens[colIndex])}': ${score.toFixed(3)}`}
                        >
                        <span
                            className={`absolute inset-0 flex items-center justify-center text-xs font-mono transition-opacity duration-150 ${alpha > 0.6 ? 'text-white' : 'text-gray-300'}`}
                        >
                            {score.toFixed(2)}
                        </span>
                        </td>
                    );
                    })}
                </tr>
                ))}
            </tbody>
            </table>
        </div>
        </div>
    );
};


export default HeatMap