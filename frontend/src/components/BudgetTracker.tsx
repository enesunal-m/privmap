"use client";

import { cn } from "@/lib/utils";
import { getBudgetPercentage, getBudgetColor, formatNumber } from "@/lib/utils";
import { Shield, AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "./ui/Button";

interface BudgetTrackerProps {
  remainingBudget: number;
  initialBudget: number;
  queriesMade: number;
  isActive: boolean;
  onReset?: () => void;
  className?: string;
}

export function BudgetTracker({
  remainingBudget,
  initialBudget,
  queriesMade,
  isActive,
  onReset,
  className,
}: BudgetTrackerProps) {
  const percentage = getBudgetPercentage(remainingBudget, initialBudget);
  const isLow = percentage < 30;
  const isExhausted = remainingBudget <= 0;

  return (
    <div
      className={cn(
        "rounded-xl p-4 transition-all duration-300",
        isExhausted
          ? "bg-red-950/50 border border-red-800"
          : isLow
          ? "bg-orange-950/30 border border-orange-800/50"
          : "bg-surface-800 border border-surface-700",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Shield
            className={cn(
              "w-5 h-5",
              isExhausted
                ? "text-red-400"
                : isLow
                ? "text-orange-400"
                : "text-primary-400"
            )}
          />
          <span className="font-medium text-surface-100">Privacy Budget</span>
        </div>
        {onReset && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onReset}
            className="text-xs"
          >
            <RefreshCw className="w-3 h-3 mr-1" />
            New Session
          </Button>
        )}
      </div>

      {/* Progress bar */}
      <div className="relative h-3 bg-surface-700 rounded-full overflow-hidden mb-3">
        <div
          className={cn(
            "absolute inset-y-0 left-0 rounded-full transition-all duration-500",
            getBudgetColor(percentage)
          )}
          style={{ width: `${percentage}%` }}
        />
        {/* Glow effect */}
        <div
          className={cn(
            "absolute inset-y-0 left-0 rounded-full blur-sm opacity-50 transition-all duration-500",
            getBudgetColor(percentage)
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {/* Stats */}
      <div className="flex justify-between text-sm">
        <div>
          <span className="text-surface-400">Remaining: </span>
          <span
            className={cn(
              "font-mono font-medium",
              isExhausted
                ? "text-red-400"
                : isLow
                ? "text-orange-400"
                : "text-primary-400"
            )}
          >
            ε = {formatNumber(remainingBudget, 3)}
          </span>
        </div>
        <div className="text-surface-400">
          {queriesMade} {queriesMade === 1 ? "query" : "queries"} made
        </div>
      </div>

      {/* Warning */}
      {isExhausted && (
        <div className="mt-3 flex items-center gap-2 text-red-400 text-sm">
          <AlertTriangle className="w-4 h-4" />
          <span>Budget exhausted. Start a new session to continue.</span>
        </div>
      )}

      {!isActive && !isExhausted && (
        <div className="mt-3 flex items-center gap-2 text-orange-400 text-sm">
          <AlertTriangle className="w-4 h-4" />
          <span>Session inactive. Start a new session.</span>
        </div>
      )}
    </div>
  );
}
