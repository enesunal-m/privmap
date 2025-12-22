"use client";

import { useState, useEffect, useCallback } from "react";
import { api, SessionResponse } from "@/lib/api";

interface PrivacySessionState {
  session: SessionResponse | null;
  isLoading: boolean;
  error: string | null;
}

export function usePrivacySession(autoCreate: boolean = false) {
  const [state, setState] = useState<PrivacySessionState>({
    session: null,
    isLoading: false,
    error: null,
  });

  const createSession = useCallback(async (budget: number = 5.0) => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));
    try {
      const session = await api.createSession(budget);
      setState({ session, isLoading: false, error: null });
      return session;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to create session";
      setState((prev) => ({ ...prev, isLoading: false, error: message }));
      return null;
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    const status = await api.getSessionStatus();
    if (status && state.session) {
      setState((prev) => ({
        ...prev,
        session: prev.session
          ? {
              ...prev.session,
              remaining_budget: status.remaining_budget,
              is_active: status.is_active,
              query_count: status.queries_made,
            }
          : null,
      }));
    }
  }, [state.session]);

  const clearSession = useCallback(() => {
    api.clearSession();
    setState({ session: null, isLoading: false, error: null });
  }, []);

  // Check for existing session on mount
  useEffect(() => {
    const existingToken = api.getSessionToken();
    if (existingToken) {
      api.getSessionStatus().then((status) => {
        if (status?.is_active) {
          // Reconstruct session from status
          setState({
            session: {
              session_token: existingToken,
              initial_budget: 5.0, // Default, we don't know the original
              remaining_budget: status.remaining_budget,
              created_at: new Date().toISOString(),
              last_query_at: null,
              is_active: status.is_active,
              query_count: status.queries_made,
            },
            isLoading: false,
            error: null,
          });
        } else if (autoCreate) {
          createSession();
        }
      });
    } else if (autoCreate) {
      createSession();
    }
  }, [autoCreate, createSession]);

  return {
    ...state,
    createSession,
    refreshStatus,
    clearSession,
    remainingBudget: state.session?.remaining_budget ?? 0,
    canQuery: state.session?.is_active && (state.session?.remaining_budget ?? 0) > 0,
  };
}

