"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { getPrivacyLevel, formatEpsilon } from "@/lib/utils";
import { Slider } from "./ui/Slider";
import { Lock, Unlock, Eye, EyeOff, Info } from "lucide-react";

interface EpsilonSelectorProps {
  value: number;
  onChange: (value: number) => void;
  maxBudget?: number;
  disabled?: boolean;
  className?: string;
}

const PRESET_VALUES = [
  { value: 0.1, label: "High Privacy" },
  { value: 0.5, label: "Balanced" },
  { value: 1.0, label: "Standard" },
  { value: 2.0, label: "High Utility" },
];

export function EpsilonSelector({
  value,
  onChange,
  maxBudget,
  disabled = false,
  className,
}: EpsilonSelectorProps) {
  const [showInfo, setShowInfo] = useState(false);
  const privacyLevel = getPrivacyLevel(value);
  const effectiveMax = maxBudget ? Math.min(10, maxBudget) : 10;

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header with info toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {value <= 1 ? (
            <Lock className="w-4 h-4 text-primary-400" />
          ) : (
            <Unlock className="w-4 h-4 text-orange-400" />
          )}
          <span className="font-medium text-surface-100">
            Privacy Parameter (ε)
          </span>
        </div>
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="text-surface-400 hover:text-surface-200 transition-colors"
        >
          <Info className="w-4 h-4" />
        </button>
      </div>

      {/* Info panel */}
      {showInfo && (
        <div className="p-3 bg-surface-900 rounded-lg text-sm text-surface-300 border border-surface-700">
          <p className="mb-2">
            <strong className="text-surface-100">Epsilon (ε)</strong> controls
            the privacy-utility tradeoff:
          </p>
          <ul className="space-y-1 list-disc list-inside">
            <li>
              <span className="text-green-400">Lower ε</span> = More privacy,
              more noise
            </li>
            <li>
              <span className="text-orange-400">Higher ε</span> = Less privacy,
              clearer data
            </li>
          </ul>
        </div>
      )}

      {/* Current value display */}
      <div className="flex items-center justify-between p-3 bg-surface-900 rounded-lg border border-surface-700">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              "w-10 h-10 rounded-full flex items-center justify-center",
              value <= 0.5
                ? "bg-green-500/20"
                : value <= 1
                ? "bg-yellow-500/20"
                : "bg-red-500/20"
            )}
          >
            {value <= 1 ? (
              <EyeOff
                className={cn(
                  "w-5 h-5",
                  value <= 0.5 ? "text-green-400" : "text-yellow-400"
                )}
              />
            ) : (
              <Eye className="w-5 h-5 text-red-400" />
            )}
          </div>
          <div>
            <div className={cn("font-medium", privacyLevel.color)}>
              {privacyLevel.label}
            </div>
            <div className="text-xs text-surface-400">
              {privacyLevel.description}
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-2xl font-bold text-surface-100">
            {formatEpsilon(value)}
          </div>
          <div className="text-xs text-surface-400">epsilon</div>
        </div>
      </div>

      {/* Slider */}
      <Slider
        value={value}
        onChange={onChange}
        min={0.01}
        max={effectiveMax}
        step={0.01}
        disabled={disabled}
      />

      {/* Preset buttons */}
      <div className="flex gap-2">
        {PRESET_VALUES.map((preset) => (
          <button
            key={preset.value}
            onClick={() => onChange(preset.value)}
            disabled={disabled || preset.value > effectiveMax}
            className={cn(
              "flex-1 py-2 px-3 rounded-lg text-xs font-medium transition-all",
              value === preset.value
                ? "bg-primary-600 text-white"
                : "bg-surface-700 text-surface-300 hover:bg-surface-600",
              (disabled || preset.value > effectiveMax) &&
                "opacity-50 cursor-not-allowed"
            )}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Budget warning */}
      {maxBudget && value > maxBudget && (
        <div className="p-2 bg-red-950/50 border border-red-800 rounded-lg text-sm text-red-400">
          This query would exceed your remaining budget of ε ={" "}
          {formatEpsilon(maxBudget)}
        </div>
      )}
    </div>
  );
}
