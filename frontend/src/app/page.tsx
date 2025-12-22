"use client";

import { useState, useCallback, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Header } from "@/components/Header";
import { EpsilonSelector } from "@/components/EpsilonSelector";
import { BudgetTracker } from "@/components/BudgetTracker";
import { StatisticsPanel } from "@/components/StatisticsPanel";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { usePrivacySession } from "@/hooks/usePrivacySession";
import { api, DecompositionResponse, BoundsResponse } from "@/lib/api";
import { Play, RotateCcw, Info, Zap } from "lucide-react";
import { PrivacyMap } from "@/components/PrivacyMap";

// Default bounds for Porto, Portugal
const DEFAULT_BOUNDS: BoundsResponse = {
  min_lon: -8.7,
  max_lon: -8.5,
  min_lat: 41.1,
  max_lat: 41.25,
  center: { lon: -8.6, lat: 41.175 },
};

export default function HomePage() {
  const [epsilon, setEpsilon] = useState(1.0);
  const [decomposition, setDecomposition] =
    useState<DecompositionResponse | null>(null);

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

  // Decomposition mutation - uses session-tracked endpoint for budget management
  const decompositionMutation = useMutation({
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
      setDecomposition(data);
      refreshStatus(); // Refresh session to get updated budget
    },
  });

  // Handle query execution
  const handleRunQuery = useCallback(() => {
    decompositionMutation.mutate(epsilon);
  }, [epsilon, decompositionMutation]);

  // Handle session reset
  const handleReset = useCallback(async () => {
    clearSession();
    setDecomposition(null);
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
                    Explore Porto taxi pickups with mathematical privacy
                    guarantees.
                  </p>
                  <p className="text-surface-400 text-xs">
                    Based on the{" "}
                    <a
                      href="https://arxiv.org/abs/1601.03229"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary-400 hover:underline"
                    >
                      PrivTree algorithm
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

            {/* Epsilon Selector */}
            <Card>
              <CardHeader>
                <CardTitle>Query Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <EpsilonSelector
                  value={epsilon}
                  onChange={setEpsilon}
                  maxBudget={remainingBudget}
                  disabled={decompositionMutation.isPending}
                />
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="flex gap-2">
              <Button
                onClick={handleRunQuery}
                disabled={!canQuery || decompositionMutation.isPending}
                isLoading={decompositionMutation.isPending}
                className="flex-1"
                size="lg"
              >
                {!decompositionMutation.isPending && (
                  <Play className="w-4 h-4 mr-2" />
                )}
                Run Query
              </Button>
              <Button
                variant="secondary"
                onClick={handleReset}
                disabled={decompositionMutation.isPending}
                title="Reset session"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            {/* Quick Actions */}
            <div className="flex gap-2">
              {[0.1, 0.5, 1.0, 2.0].map((eps) => (
                <button
                  key={eps}
                  onClick={() => {
                    setEpsilon(eps);
                    setTimeout(() => decompositionMutation.mutate(eps), 100);
                  }}
                  disabled={decompositionMutation.isPending || remainingBudget < eps}
                  className="flex-1 py-2 px-2 rounded-lg bg-surface-800 hover:bg-surface-700 
                           text-xs font-mono text-surface-300 transition-colors
                           disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-1"
                  title={remainingBudget < eps ? `Need ε=${eps}, have ${remainingBudget.toFixed(2)}` : `Run with ε=${eps}`}
                >
                  <Zap className="w-3 h-3" />
                  ε={eps}
                </button>
              ))}
            </div>

            {/* Statistics */}
            <Card>
              <CardHeader>
                <CardTitle>Decomposition Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <StatisticsPanel
                  statistics={decomposition?.statistics || null}
                  isLoading={decompositionMutation.isPending}
                />
              </CardContent>
            </Card>

            {/* Error Display */}
            {decompositionMutation.isError && (
              <Card className="border-red-800 bg-red-950/30">
                <CardContent className="text-red-400 text-sm">
                  {decompositionMutation.error instanceof Error
                    ? decompositionMutation.error.message
                    : "An error occurred"}
                </CardContent>
              </Card>
            )}
          </div>
        </aside>

        {/* Map Area */}
        <div className="flex-1 p-4">
          <PrivacyMap
            geojson={decomposition?.geojson || null}
            bounds={bounds}
            isLoading={decompositionMutation.isPending || sessionLoading}
          />
        </div>
      </main>
    </div>
  );
}
