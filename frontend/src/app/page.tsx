"use client";

import { useState, useCallback, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Header } from "@/components/Header";
import { EpsilonSelector } from "@/components/EpsilonSelector";
import { BudgetTracker } from "@/components/BudgetTracker";
import { StatisticsPanel } from "@/components/StatisticsPanel";
import { ComparisonStats } from "@/components/ComparisonStats";
import { AlgorithmSelector, Algorithm } from "@/components/AlgorithmSelector";
import { MSEPanel } from "@/components/MSEPanel";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { usePrivacySession } from "@/hooks/usePrivacySession";
import {
  api,
  DecompositionResponse,
  AdaptiveGridResponse,
  BoundsResponse,
} from "@/lib/api";
import { Play, RotateCcw, Info, Zap } from "lucide-react";
import { PrivacyMap } from "@/components/PrivacyMap";
import { ComparisonMap } from "@/components/ComparisonMap";

// Default bounds for Porto, Portugal
const DEFAULT_BOUNDS: BoundsResponse = {
  min_lon: -8.7,
  max_lon: -8.5,
  min_lat: 41.1,
  max_lat: 41.25,
  center: { lon: -8.6, lat: 41.175 },
};

interface ComparisonResult {
  privtree: DecompositionResponse;
  adaptiveGrid: AdaptiveGridResponse;
}

export default function HomePage() {
  const [epsilon, setEpsilon] = useState(1.0);
  const [algorithm, setAlgorithm] = useState<Algorithm>("privtree");
  const [privtreeResult, setPrivtreeResult] =
    useState<DecompositionResponse | null>(null);
  const [adaptiveGridResult, setAdaptiveGridResult] =
    useState<AdaptiveGridResponse | null>(null);
  const [comparisonResult, setComparisonResult] =
    useState<ComparisonResult | null>(null);

  // Privacy session management
  const {
    session,
    remainingBudget,
    canQuery,
    createSession,
    refreshStatus,
    clearSession,
    isLoading: sessionLoading,
  } = usePrivacySession(true);

  // Fetch bounds from API
  const { data: bounds = DEFAULT_BOUNDS } = useQuery({
    queryKey: ["bounds"],
    queryFn: () => api.getBounds(),
    staleTime: Infinity,
  });

  // PrivTree mutation
  const privtreeMutation = useMutation({
    mutationFn: async (eps: number) => {
      return api.getDecomposition({
        epsilon: eps,
        bounds: {
          min_lon: bounds.min_lon,
          max_lon: bounds.max_lon,
          min_lat: bounds.min_lat,
          max_lat: bounds.max_lat,
        },
      });
    },
    onSuccess: (data) => {
      setPrivtreeResult(data);
      refreshStatus();
    },
  });

  // Adaptive Grid mutation
  const adaptiveGridMutation = useMutation({
    mutationFn: async (eps: number) => {
      return api.getAdaptiveGrid({
        epsilon: eps,
        bounds: {
          min_lon: bounds.min_lon,
          max_lon: bounds.max_lon,
          min_lat: bounds.min_lat,
          max_lat: bounds.max_lat,
        },
      });
    },
    onSuccess: (data) => {
      setAdaptiveGridResult(data);
      refreshStatus();
    },
  });

  // Comparison mutation
  const comparisonMutation = useMutation({
    mutationFn: async (eps: number) => {
      return api.getComparison({
        epsilon: eps,
        bounds: {
          min_lon: bounds.min_lon,
          max_lon: bounds.max_lon,
          min_lat: bounds.min_lat,
          max_lat: bounds.max_lat,
        },
      });
    },
    onSuccess: (data) => {
      setComparisonResult({
        privtree: data.privtree,
        adaptiveGrid: data.adaptive_grid,
      });
      refreshStatus();
    },
  });

  // Handle query execution based on selected algorithm
  const handleRunQuery = useCallback(() => {
    switch (algorithm) {
      case "privtree":
        privtreeMutation.mutate(epsilon);
        break;
      case "adaptive_grid":
        adaptiveGridMutation.mutate(epsilon);
        break;
      case "comparison":
        comparisonMutation.mutate(epsilon);
        break;
    }
  }, [
    algorithm,
    epsilon,
    privtreeMutation,
    adaptiveGridMutation,
    comparisonMutation,
  ]);

  // Handle session reset
  const handleReset = useCallback(async () => {
    clearSession();
    setPrivtreeResult(null);
    setAdaptiveGridResult(null);
    setComparisonResult(null);
    await createSession(5.0);
  }, [clearSession, createSession]);

  // Keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        handleRunQuery();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleRunQuery]);

  // Check if any mutation is pending
  const isQueryPending =
    privtreeMutation.isPending ||
    adaptiveGridMutation.isPending ||
    comparisonMutation.isPending;

  // Calculate required budget (comparison uses 2x epsilon)
  const requiredBudget = algorithm === "comparison" ? epsilon * 2 : epsilon;
  const hasSufficientBudget = remainingBudget >= requiredBudget;

  // Get current GeoJSON based on algorithm
  const getCurrentGeojson = () => {
    switch (algorithm) {
      case "privtree":
        return privtreeResult?.geojson || null;
      case "adaptive_grid":
        return adaptiveGridResult?.geojson || null;
      case "comparison":
        return null; // Handled separately
    }
  };

  // Get current statistics based on algorithm
  const getCurrentStatistics = () => {
    switch (algorithm) {
      case "privtree":
        return privtreeResult?.statistics || null;
      case "adaptive_grid":
        return adaptiveGridResult?.statistics || null;
      case "comparison":
        return null; // Handled separately
    }
  };

  // Get current error
  const getCurrentError = () => {
    switch (algorithm) {
      case "privtree":
        return privtreeMutation.error;
      case "adaptive_grid":
        return adaptiveGridMutation.error;
      case "comparison":
        return comparisonMutation.error;
    }
  };

  return (
    <div className="min-h-screen bg-surface-950">
      <Header />

      <main className="pt-14 h-screen flex">
        {/* Sidebar */}
        <aside className="w-96 flex-shrink-0 border-r border-surface-800 overflow-y-auto">
          <div className="p-4 space-y-4">
            {/* Welcome Card */}
            <Card variant="glass">
              <CardContent className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-primary-500/20 flex items-center justify-center flex-shrink-0">
                  <Info className="w-4 h-4 text-primary-400" />
                </div>
                <div className="text-sm">
                  <p className="text-surface-200 mb-1">
                    Compare differential privacy algorithms on Porto taxi data.
                  </p>
                  <p className="text-surface-400 text-xs">
                    <a
                      href="https://arxiv.org/abs/1601.03229"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-emerald-400 hover:underline"
                    >
                      PrivTree
                    </a>
                    {" vs "}
                    <a
                      href="https://www.cs.purdue.edu/homes/ninghui/papers/dp_grid_icde13.pdf"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:underline"
                    >
                      Adaptive Grid
                    </a>
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Budget Tracker */}
            {session && (
              <BudgetTracker
                remainingBudget={remainingBudget}
                initialBudget={session.initial_budget}
                queriesMade={session.query_count}
                isActive={session.is_active}
                onReset={handleReset}
              />
            )}

            {/* Algorithm Selector */}
            <Card>
              <CardContent className="pt-4">
                <AlgorithmSelector
                  value={algorithm}
                  onChange={setAlgorithm}
                  disabled={isQueryPending}
                />
              </CardContent>
            </Card>

            {/* Epsilon Selector */}
            <Card>
              <CardHeader>
                <CardTitle>
                  Privacy Budget
                  {algorithm === "comparison" && (
                    <span className="text-xs font-normal text-surface-400 ml-2">
                      (×2 for comparison)
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <EpsilonSelector
                  value={epsilon}
                  onChange={setEpsilon}
                  maxBudget={
                    algorithm === "comparison"
                      ? remainingBudget / 2
                      : remainingBudget
                  }
                  disabled={isQueryPending}
                />
                {algorithm === "comparison" && (
                  <div className="mt-2 text-xs text-surface-500">
                    Total cost: ε = {(epsilon * 2).toFixed(2)} (
                    {epsilon.toFixed(2)} per algorithm)
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="flex gap-2">
              <Button
                onClick={handleRunQuery}
                disabled={!canQuery || isQueryPending || !hasSufficientBudget}
                isLoading={isQueryPending}
                className="flex-1"
                size="lg"
              >
                {!isQueryPending && <Play className="w-4 h-4 mr-2" />}
                {algorithm === "comparison" ? "Compare" : "Run Query"}
              </Button>
              <Button
                variant="secondary"
                onClick={handleReset}
                disabled={isQueryPending}
                title="Reset session"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            {/* Quick Actions */}
            <div className="flex gap-2">
              {[0.1, 0.5, 1.0, 2.0].map((eps) => {
                const required =
                  algorithm === "comparison" ? eps * 2 : eps;
                const canUse = remainingBudget >= required;
                return (
                  <button
                    key={eps}
                    onClick={() => {
                      setEpsilon(eps);
                      setTimeout(() => {
                        switch (algorithm) {
                          case "privtree":
                            privtreeMutation.mutate(eps);
                            break;
                          case "adaptive_grid":
                            adaptiveGridMutation.mutate(eps);
                            break;
                          case "comparison":
                            comparisonMutation.mutate(eps);
                            break;
                        }
                      }, 100);
                    }}
                    disabled={isQueryPending || !canUse}
                    className="flex-1 py-2 px-2 rounded-lg bg-surface-800 hover:bg-surface-700
                             text-xs font-mono text-surface-300 transition-colors
                             disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-1"
                    title={
                      !canUse
                        ? `Need ε=${required.toFixed(2)}, have ${remainingBudget.toFixed(2)}`
                        : `Run with ε=${eps}`
                    }
                  >
                    <Zap className="w-3 h-3" />ε={eps}
                  </button>
                );
              })}
            </div>

            {/* Statistics */}
            <Card>
              <CardHeader>
                <CardTitle>
                  {algorithm === "comparison"
                    ? "Algorithm Comparison"
                    : "Decomposition Statistics"}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {algorithm === "comparison" ? (
                  <ComparisonStats
                    privtreeStats={comparisonResult?.privtree.statistics || null}
                    adaptiveGridStats={
                      comparisonResult?.adaptiveGrid.statistics || null
                    }
                    isLoading={comparisonMutation.isPending}
                  />
                ) : (
                  <StatisticsPanel
                    statistics={getCurrentStatistics()}
                    isLoading={
                      algorithm === "privtree"
                        ? privtreeMutation.isPending
                        : adaptiveGridMutation.isPending
                    }
                  />
                )}
              </CardContent>
            </Card>

            {/* MSE Calculator - shown in comparison mode */}
            {algorithm === "comparison" && (
              <Card>
                <CardHeader>
                  <CardTitle>Accuracy Analysis (MSE)</CardTitle>
                </CardHeader>
                <CardContent>
                  <MSEPanel epsilon={epsilon} bounds={bounds} />
                </CardContent>
              </Card>
            )}

            {/* Error Display */}
            {getCurrentError() && (
              <Card className="border-red-800 bg-red-950/30">
                <CardContent className="text-red-400 text-sm">
                  {getCurrentError() instanceof Error
                    ? getCurrentError()?.message
                    : "An error occurred"}
                </CardContent>
              </Card>
            )}
          </div>
        </aside>

        {/* Map Area */}
        <div className="flex-1 p-4">
          {algorithm === "comparison" ? (
            <ComparisonMap
              privtreeGeojson={comparisonResult?.privtree.geojson || null}
              adaptiveGridGeojson={comparisonResult?.adaptiveGrid.geojson || null}
              bounds={bounds}
              isLoading={comparisonMutation.isPending || sessionLoading}
            />
          ) : (
            <PrivacyMap
              geojson={getCurrentGeojson()}
              bounds={bounds}
              isLoading={isQueryPending || sessionLoading}
            />
          )}
        </div>
      </main>
    </div>
  );
}
