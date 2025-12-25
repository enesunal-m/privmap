"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { cn, formatNumber } from "@/lib/utils";
import { api, MSEResponse, BoundsResponse } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import {
  BarChart3,
  Trophy,
  Target,
  TrendingDown,
  Calculator,
  Loader2,
} from "lucide-react";

interface MSEPanelProps {
  epsilon: number;
  bounds: BoundsResponse;
  className?: string;
}

export function MSEPanel({ epsilon, bounds, className }: MSEPanelProps) {
  const [numTrials, setNumTrials] = useState(5);
  const [mseResult, setMseResult] = useState<MSEResponse | null>(null);

  const mseMutation = useMutation({
    mutationFn: async () => {
      return api.getQuickMSE(epsilon, numTrials, bounds);
    },
    onSuccess: (data) => {
      setMseResult(data);
    },
  });

  const getWinnerColor = (winner: string) => {
    return winner === "privtree" ? "text-emerald-400" : "text-blue-400";
  };

  const getWinnerBg = (winner: string) => {
    return winner === "privtree"
      ? "bg-emerald-500/10 border-emerald-500/30"
      : "bg-blue-500/10 border-blue-500/30";
  };

  return (
    <div className={cn("space-y-3", className)}>
      {/* Controls */}
      <div className="flex items-center gap-2">
        <div className="flex-1">
          <label className="text-xs text-surface-400 mb-1 block">Trials</label>
          <select
            value={numTrials}
            onChange={(e) => setNumTrials(Number(e.target.value))}
            disabled={mseMutation.isPending}
            className="w-full px-2 py-1.5 rounded-lg bg-surface-800 border border-surface-700
                     text-sm text-surface-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          >
            <option value={3}>3 trials</option>
            <option value={5}>5 trials</option>
            <option value={10}>10 trials</option>
            <option value={20}>20 trials</option>
          </select>
        </div>
        <div className="pt-5">
          <Button
            onClick={() => mseMutation.mutate()}
            disabled={mseMutation.isPending}
            size="sm"
            className="gap-1"
          >
            {mseMutation.isPending ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Calculator className="w-3.5 h-3.5" />
            )}
            Calculate
          </Button>
        </div>
      </div>

      {/* Loading State */}
      {mseMutation.isPending && (
        <div className="flex items-center justify-center py-6 text-surface-400">
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="w-6 h-6 animate-spin" />
            <span className="text-xs">Running {numTrials} trials...</span>
          </div>
        </div>
      )}

      {/* Error State */}
      {mseMutation.isError && (
        <div className="p-3 rounded-lg bg-red-950/30 border border-red-800 text-red-400 text-sm">
          {mseMutation.error instanceof Error
            ? mseMutation.error.message
            : "Failed to calculate MSE"}
        </div>
      )}

      {/* Results */}
      {mseResult && !mseMutation.isPending && (
        <div className="space-y-3">
          {/* Winner Badge */}
          <div
            className={cn(
              "flex items-center gap-2 p-2.5 rounded-lg border",
              getWinnerBg(mseResult.winner)
            )}
          >
            <Trophy className={cn("w-4 h-4", getWinnerColor(mseResult.winner))} />
            <span className="text-sm">
              <span className={cn("font-medium", getWinnerColor(mseResult.winner))}>
                {mseResult.winner === "privtree" ? "PrivTree" : "Adaptive Grid"}
              </span>
              <span className="text-surface-400"> wins with lower MSE</span>
            </span>
          </div>

          {/* MSE Comparison */}
          <div className="grid grid-cols-2 gap-2">
            {/* PrivTree MSE */}
            <div
              className={cn(
                "p-2.5 rounded-lg border",
                mseResult.winner === "privtree"
                  ? "bg-emerald-950/30 border-emerald-800/50"
                  : "bg-surface-900/50 border-surface-700"
              )}
            >
              <div className="flex items-center gap-1.5 mb-2">
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <span className="text-xs font-medium text-emerald-400">
                  PrivTree
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-surface-500">MSE</span>
                  <span className="font-mono text-surface-200">
                    {formatNumber(mseResult.privtree.mse, 1)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-surface-500">RMSE</span>
                  <span className="font-mono text-surface-200">
                    {formatNumber(mseResult.privtree.rmse, 2)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-surface-500">Mean Err</span>
                  <span className="font-mono text-surface-200">
                    {formatNumber(mseResult.privtree.mean_error, 2)}
                  </span>
                </div>
              </div>
            </div>

            {/* Adaptive Grid MSE */}
            <div
              className={cn(
                "p-2.5 rounded-lg border",
                mseResult.winner === "adaptive_grid"
                  ? "bg-blue-950/30 border-blue-800/50"
                  : "bg-surface-900/50 border-surface-700"
              )}
            >
              <div className="flex items-center gap-1.5 mb-2">
                <div className="w-2 h-2 rounded-full bg-blue-500" />
                <span className="text-xs font-medium text-blue-400">
                  Adaptive Grid
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-surface-500">MSE</span>
                  <span className="font-mono text-surface-200">
                    {formatNumber(mseResult.adaptive_grid.mse, 1)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-surface-500">RMSE</span>
                  <span className="font-mono text-surface-200">
                    {formatNumber(mseResult.adaptive_grid.rmse, 2)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-surface-500">Mean Err</span>
                  <span className="font-mono text-surface-200">
                    {formatNumber(mseResult.adaptive_grid.mean_error, 2)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Visual Comparison Bar */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs text-surface-400">
              <span>Relative Accuracy (lower is better)</span>
            </div>
            <div className="flex h-3 rounded-full overflow-hidden bg-surface-800">
              {(() => {
                const total =
                  mseResult.privtree.mse + mseResult.adaptive_grid.mse;
                const ptPercent =
                  total > 0 ? (mseResult.privtree.mse / total) * 100 : 50;
                const agPercent = 100 - ptPercent;
                return (
                  <>
                    <div
                      className="bg-emerald-500 transition-all"
                      style={{ width: `${ptPercent}%` }}
                      title={`PrivTree: ${ptPercent.toFixed(1)}%`}
                    />
                    <div
                      className="bg-blue-500 transition-all"
                      style={{ width: `${agPercent}%` }}
                      title={`Adaptive Grid: ${agPercent.toFixed(1)}%`}
                    />
                  </>
                );
              })()}
            </div>
            <div className="flex justify-between text-[10px] text-surface-500">
              <span>PrivTree MSE share</span>
              <span>AG MSE share</span>
            </div>
          </div>

          {/* Meta Info */}
          <div className="flex items-center justify-between text-xs text-surface-500 pt-2 border-t border-surface-800">
            <div className="flex items-center gap-1">
              <Target className="w-3 h-3" />
              <span>ε = {mseResult.epsilon_used}</span>
            </div>
            <div className="flex items-center gap-1">
              <BarChart3 className="w-3 h-3" />
              <span>{mseResult.num_trials} trials</span>
            </div>
            <div className="flex items-center gap-1">
              <TrendingDown className="w-3 h-3" />
              <span>{mseResult.privtree.num_cells} cells</span>
            </div>
          </div>
        </div>
      )}

      {/* Initial State */}
      {!mseResult && !mseMutation.isPending && (
        <div className="flex flex-col items-center justify-center py-4 text-surface-400 text-center">
          <BarChart3 className="w-6 h-6 mb-2 opacity-50" />
          <p className="text-xs">
            Calculate MSE to compare algorithm accuracy
          </p>
          <p className="text-[10px] text-surface-500 mt-1">
            Does not consume privacy budget
          </p>
        </div>
      )}
    </div>
  );
}
