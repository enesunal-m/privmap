"use client";

import { cn } from "@/lib/utils";
import { formatNumber, formatEpsilon } from "@/lib/utils";
import {
  Layers,
  TreeDeciduous,
  Hash,
  Sigma,
  Activity,
  Sparkles,
} from "lucide-react";

interface Statistics {
  total_leaves: number;
  max_depth: number;
  min_depth: number;
  avg_depth: number;
  total_noisy_count: number;
  epsilon_used: number;
  noise_scale: number;
  delta: number;
}

interface StatisticsPanelProps {
  statistics: Statistics | null;
  isLoading?: boolean;
  className?: string;
}

interface StatItemProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subtext?: string;
}

function StatItem({ icon, label, value, subtext }: StatItemProps) {
  return (
    <div className="flex items-center gap-3 p-3 bg-surface-900/50 rounded-lg">
      <div className="w-8 h-8 rounded-lg bg-surface-700 flex items-center justify-center text-primary-400">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-xs text-surface-400 truncate">{label}</div>
        <div className="font-mono font-medium text-surface-100">{value}</div>
        {subtext && <div className="text-xs text-surface-500">{subtext}</div>}
      </div>
    </div>
  );
}

export function StatisticsPanel({
  statistics,
  isLoading = false,
  className,
}: StatisticsPanelProps) {
  if (isLoading) {
    return (
      <div className={cn("space-y-2", className)}>
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="h-16 rounded-lg loading-shimmer"
            style={{ animationDelay: `${i * 100}ms` }}
          />
        ))}
      </div>
    );
  }

  if (!statistics) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center py-8 text-surface-400",
          className
        )}
      >
        <Sparkles className="w-8 h-8 mb-2 opacity-50" />
        <p className="text-sm">Run a query to see statistics</p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      <StatItem
        icon={<Layers className="w-4 h-4" />}
        label="Total Cells"
        value={formatNumber(statistics.total_leaves, 0)}
        subtext="Leaf nodes in quadtree"
      />

      <StatItem
        icon={<TreeDeciduous className="w-4 h-4" />}
        label="Tree Depth"
        value={`${statistics.min_depth} - ${statistics.max_depth}`}
        subtext={`Average: ${statistics.avg_depth.toFixed(1)}`}
      />

      <StatItem
        icon={<Hash className="w-4 h-4" />}
        label="Total Count"
        value={formatNumber(statistics.total_noisy_count, 0)}
        subtext="Sum of noisy counts"
      />

      <StatItem
        icon={<Activity className="w-4 h-4" />}
        label="Privacy Used"
        value={`ε = ${formatEpsilon(statistics.epsilon_used)}`}
      />

      <StatItem
        icon={<Sigma className="w-4 h-4" />}
        label="Noise Scale"
        value={`λ = ${statistics.noise_scale.toFixed(3)}`}
        subtext={`δ = ${statistics.delta.toFixed(3)}`}
      />
    </div>
  );
}
