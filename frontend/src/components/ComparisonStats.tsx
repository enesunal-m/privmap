"use client";

import { cn } from "@/lib/utils";
import { formatNumber, formatEpsilon } from "@/lib/utils";
import {
  Layers,
  TreeDeciduous,
  Grid3X3,
  Hash,
  Activity,
  Sigma,
  Sparkles,
  ArrowRight,
} from "lucide-react";
import type {
  DecompositionStatistics,
  AdaptiveGridStatistics,
} from "@/lib/api";

interface ComparisonStatsProps {
  privtreeStats: DecompositionStatistics | null;
  adaptiveGridStats: AdaptiveGridStatistics | null;
  isLoading?: boolean;
  className?: string;
}

interface CompareRowProps {
  icon: React.ReactNode;
  label: string;
  privtreeValue: string | number;
  adaptiveGridValue: string | number;
  privtreeSubtext?: string;
  adaptiveGridSubtext?: string;
}

function CompareRow({
  icon,
  label,
  privtreeValue,
  adaptiveGridValue,
  privtreeSubtext,
  adaptiveGridSubtext,
}: CompareRowProps) {
  return (
    <div className="flex items-center gap-2 p-2 bg-surface-900/50 rounded-lg">
      <div className="w-7 h-7 rounded-lg bg-surface-700 flex items-center justify-center text-primary-400 flex-shrink-0">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-xs text-surface-400">{label}</div>
      </div>
      {/* PrivTree Value */}
      <div className="w-20 text-right">
        <div className="font-mono text-sm font-medium text-emerald-400">
          {privtreeValue}
        </div>
        {privtreeSubtext && (
          <div className="text-[10px] text-surface-500">{privtreeSubtext}</div>
        )}
      </div>
      <ArrowRight className="w-3 h-3 text-surface-600" />
      {/* Adaptive Grid Value */}
      <div className="w-20 text-right">
        <div className="font-mono text-sm font-medium text-blue-400">
          {adaptiveGridValue}
        </div>
        {adaptiveGridSubtext && (
          <div className="text-[10px] text-surface-500">
            {adaptiveGridSubtext}
          </div>
        )}
      </div>
    </div>
  );
}

export function ComparisonStats({
  privtreeStats,
  adaptiveGridStats,
  isLoading = false,
  className,
}: ComparisonStatsProps) {
  if (isLoading) {
    return (
      <div className={cn("space-y-2", className)}>
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            className="h-12 rounded-lg loading-shimmer"
            style={{ animationDelay: `${i * 100}ms` }}
          />
        ))}
      </div>
    );
  }

  if (!privtreeStats || !adaptiveGridStats) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center py-8 text-surface-400",
          className
        )}
      >
        <Sparkles className="w-8 h-8 mb-2 opacity-50" />
        <p className="text-sm">Run comparison to see statistics</p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1">
        <div className="text-xs text-surface-500">Metric</div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-emerald-400 w-20 text-right">
            PrivTree
          </span>
          <div className="w-3" />
          <span className="text-xs font-medium text-blue-400 w-20 text-right">
            AG
          </span>
        </div>
      </div>

      <CompareRow
        icon={<Layers className="w-3.5 h-3.5" />}
        label="Total Cells"
        privtreeValue={formatNumber(privtreeStats.total_leaves, 0)}
        adaptiveGridValue={formatNumber(adaptiveGridStats.total_cells, 0)}
      />

      <CompareRow
        icon={<TreeDeciduous className="w-3.5 h-3.5" />}
        label="Depth Range"
        privtreeValue={`${privtreeStats.min_depth}-${privtreeStats.max_depth}`}
        adaptiveGridValue={`${adaptiveGridStats.min_depth}-${adaptiveGridStats.max_depth}`}
        privtreeSubtext={`avg: ${privtreeStats.avg_depth.toFixed(1)}`}
        adaptiveGridSubtext={`avg: ${adaptiveGridStats.avg_depth.toFixed(1)}`}
      />

      <CompareRow
        icon={<Grid3X3 className="w-3.5 h-3.5" />}
        label="Grid Size"
        privtreeValue="Adaptive"
        adaptiveGridValue={`${adaptiveGridStats.m1}×${adaptiveGridStats.m1}`}
        privtreeSubtext="quadtree"
        adaptiveGridSubtext={`→ ${adaptiveGridStats.m2}×${adaptiveGridStats.m2}`}
      />

      <CompareRow
        icon={<Hash className="w-3.5 h-3.5" />}
        label="Total Count"
        privtreeValue={formatNumber(privtreeStats.total_noisy_count, 0)}
        adaptiveGridValue={formatNumber(adaptiveGridStats.total_noisy_count, 0)}
      />

      <CompareRow
        icon={<Activity className="w-3.5 h-3.5" />}
        label="Privacy (ε)"
        privtreeValue={formatEpsilon(privtreeStats.epsilon_used)}
        adaptiveGridValue={formatEpsilon(adaptiveGridStats.epsilon_used)}
      />

      <CompareRow
        icon={<Sigma className="w-3.5 h-3.5" />}
        label="Noise Scale"
        privtreeValue={`λ=${privtreeStats.noise_scale.toFixed(2)}`}
        adaptiveGridValue={`λ=${adaptiveGridStats.noise_scale_l2.toFixed(2)}`}
        privtreeSubtext={`δ=${privtreeStats.delta.toFixed(2)}`}
        adaptiveGridSubtext={`L1: ${adaptiveGridStats.noise_scale_l1.toFixed(2)}`}
      />

      {/* Algorithm Characteristics */}
      <div className="mt-4 pt-4 border-t border-surface-800">
        <div className="text-xs text-surface-500 mb-2">Characteristics</div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 bg-emerald-950/30 border border-emerald-800/30 rounded-lg">
            <div className="font-medium text-emerald-400 mb-1">PrivTree</div>
            <ul className="text-surface-400 space-y-0.5">
              <li>• Organic, deep structure</li>
              <li>• Constant noise per level</li>
              <li>• Adapts to data density</li>
            </ul>
          </div>
          <div className="p-2 bg-blue-950/30 border border-blue-800/30 rounded-lg">
            <div className="font-medium text-blue-400 mb-1">Adaptive Grid</div>
            <ul className="text-surface-400 space-y-0.5">
              <li>• Rigid 2-level structure</li>
              <li>• Split budget (ε₁/ε₂)</li>
              <li>• Predictable partitions</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
