"use client";

import { cn } from "@/lib/utils";
import { TreeDeciduous, Grid3X3, GitCompare } from "lucide-react";

export type Algorithm = "privtree" | "adaptive_grid" | "comparison";

interface AlgorithmSelectorProps {
  value: Algorithm;
  onChange: (algorithm: Algorithm) => void;
  disabled?: boolean;
  className?: string;
}

const algorithms = [
  {
    id: "privtree" as Algorithm,
    label: "PrivTree",
    shortLabel: "PT",
    icon: TreeDeciduous,
    description: "Recursive adaptive quadtree",
    color: "emerald",
  },
  {
    id: "adaptive_grid" as Algorithm,
    label: "Adaptive Grid",
    shortLabel: "AG",
    icon: Grid3X3,
    description: "Two-level uniform grid",
    color: "blue",
  },
  {
    id: "comparison" as Algorithm,
    label: "Compare",
    shortLabel: "VS",
    icon: GitCompare,
    description: "Side-by-side comparison",
    color: "purple",
  },
];

export function AlgorithmSelector({
  value,
  onChange,
  disabled = false,
  className,
}: AlgorithmSelectorProps) {
  return (
    <div className={cn("space-y-2", className)}>
      <label className="text-xs font-medium text-surface-400 uppercase tracking-wide">
        Algorithm
      </label>
      <div className="grid grid-cols-3 gap-2">
        {algorithms.map((algo) => {
          const Icon = algo.icon;
          const isSelected = value === algo.id;
          const colorClasses = {
            emerald: {
              bg: "bg-emerald-500/20",
              border: "border-emerald-500",
              text: "text-emerald-400",
              ring: "ring-emerald-500/50",
            },
            blue: {
              bg: "bg-blue-500/20",
              border: "border-blue-500",
              text: "text-blue-400",
              ring: "ring-blue-500/50",
            },
            purple: {
              bg: "bg-purple-500/20",
              border: "border-purple-500",
              text: "text-purple-400",
              ring: "ring-purple-500/50",
            },
          }[algo.color];

          return (
            <button
              key={algo.id}
              onClick={() => onChange(algo.id)}
              disabled={disabled}
              className={cn(
                "relative flex flex-col items-center gap-1 p-3 rounded-lg border transition-all",
                "focus:outline-none focus:ring-2",
                disabled && "opacity-50 cursor-not-allowed",
                isSelected
                  ? cn(
                      colorClasses.bg,
                      colorClasses.border,
                      colorClasses.ring,
                      "ring-2"
                    )
                  : "border-surface-700 bg-surface-800/50 hover:bg-surface-700/50 hover:border-surface-600"
              )}
            >
              <Icon
                className={cn(
                  "w-5 h-5",
                  isSelected ? colorClasses.text : "text-surface-400"
                )}
              />
              <span
                className={cn(
                  "text-xs font-medium",
                  isSelected ? colorClasses.text : "text-surface-300"
                )}
              >
                {algo.shortLabel}
              </span>
            </button>
          );
        })}
      </div>
      {/* Description */}
      <div className="text-xs text-surface-500 text-center">
        {algorithms.find((a) => a.id === value)?.description}
      </div>
    </div>
  );
}
