"use client";

/**
 * Utility helpers used across the frontend components.
 */

// Simple className joiner (avoids bringing an extra dependency)
export function cn(...inputs: Array<string | null | undefined | false>): string {
  return inputs.filter(Boolean).join(" ");
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function formatNumber(value: number, decimals = 2): string {
  if (Number.isNaN(value)) return "0";
  const formatter = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return formatter.format(value);
}

export function formatEpsilon(value: number): string {
  return formatNumber(value, value < 0.1 ? 3 : 2);
}

export function getBudgetPercentage(remaining: number, initial: number): number {
  if (initial <= 0) return 0;
  return clamp((remaining / initial) * 100, 0, 100);
}

export function getBudgetColor(percentage: number): string {
  if (percentage < 20) return "bg-red-500";
  if (percentage < 50) return "bg-orange-400";
  if (percentage < 80) return "bg-yellow-400";
  return "bg-green-500";
}

export function getPrivacyLevel(epsilon: number): {
  label: string;
  description: string;
  color: string;
} {
  if (epsilon <= 0.1) {
    return {
      label: "Very High Privacy",
      description: "Strong noise, safest for sensitive data",
      color: "text-green-400",
    };
  }
  if (epsilon <= 0.5) {
    return {
      label: "High Privacy",
      description: "Good balance with meaningful protection",
      color: "text-emerald-400",
    };
  }
  if (epsilon <= 1.0) {
    return {
      label: "Balanced",
      description: "Standard setting with moderate noise",
      color: "text-yellow-400",
    };
  }
  return {
    label: "High Utility",
    description: "Clearer data with weaker privacy",
    color: "text-orange-400",
  };
}

export function getHeatmapColor(normalized: number, alpha = 1): string {
  const t = clamp(normalized, 0, 1);
  // Gradient from deep purple → teal → lime
  const stops = [
    { r: 59, g: 7, b: 100 }, // purple
    { r: 6, g: 182, b: 212 }, // teal
    { r: 132, g: 204, b: 22 }, // lime
  ];

  const segment = t * (stops.length - 1);
  const i = Math.floor(segment);
  const frac = segment - i;
  const start = stops[i] ?? stops[0];
  const end = stops[i + 1] ?? stops[stops.length - 1];

  const r = Math.round(start.r + (end.r - start.r) * frac);
  const g = Math.round(start.g + (end.g - start.g) * frac);
  const b = Math.round(start.b + (end.b - start.b) * frac);

  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

