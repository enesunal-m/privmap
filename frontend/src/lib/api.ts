"use client";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const SESSION_TOKEN_KEY = "privmap_session_token";

export interface BoundsResponse {
  min_lon: number;
  max_lon: number;
  min_lat: number;
  max_lat: number;
  center: { lon: number; lat: number };
}

export interface DecompositionStatistics {
  total_leaves: number;
  max_depth: number;
  min_depth: number;
  avg_depth: number;
  total_noisy_count: number;
  epsilon_used: number;
  noise_scale: number;
  delta: number;
}

export interface DecompositionResponse {
  geojson: GeoJSON.FeatureCollection;
  statistics: DecompositionStatistics;
  epsilon_spent: number;
  remaining_budget: number;
}

export interface SessionResponse {
  session_token: string;
  initial_budget: number;
  remaining_budget: number;
  created_at: string;
  last_query_at: string | null;
  is_active: boolean;
  query_count: number;
}

export interface SessionStatus {
  remaining_budget: number;
  queries_made: number;
  is_active: boolean;
  can_query: boolean;
}

export interface AdaptiveGridStatistics {
  algorithm: string;
  total_cells: number;
  level1_cells: number;
  level2_cells: number;
  m1: number;
  m2: number;
  max_depth: number;
  min_depth: number;
  avg_depth: number;
  total_noisy_count: number;
  epsilon_used: number;
  epsilon1: number;
  epsilon2: number;
  noise_scale_l1: number;
  noise_scale_l2: number;
  density_threshold: number;
  total_leaves: number;
  noise_scale: number;
  delta: number;
}

export interface AdaptiveGridResponse {
  geojson: GeoJSON.FeatureCollection;
  statistics: AdaptiveGridStatistics;
  epsilon_spent: number;
  remaining_budget: number;
}

export interface ComparisonResponse {
  privtree: DecompositionResponse;
  adaptive_grid: AdaptiveGridResponse;
  epsilon_spent: number;
  remaining_budget: number;
}

function getBaseUrl(): string {
  // Ensure trailing slash is not duplicated
  return API_URL.replace(/\/+$/, "");
}

function getSessionToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(SESSION_TOKEN_KEY);
}

function setSessionToken(token: string) {
  if (typeof window === "undefined") return;
  localStorage.setItem(SESSION_TOKEN_KEY, token);
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  includeSession = true
): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  if (includeSession) {
    const token = getSessionToken();
    if (token) {
      headers["x-session-token"] = token;
    }
  }

  const res = await fetch(`${getBaseUrl()}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || `Request failed: ${res.status}`);
  }

  return res.json();
}

async function getBounds(): Promise<BoundsResponse> {
  return request<BoundsResponse>("/api/spatial/bounds", {}, false);
}

async function getDecomposition(params: {
  epsilon: number;
  bounds?: Omit<BoundsResponse, "center">;
}): Promise<DecompositionResponse> {
  return request<DecompositionResponse>("/api/spatial/decomposition", {
    method: "POST",
    body: JSON.stringify({
      epsilon: params.epsilon,
      bounds: params.bounds,
      use_cache: true,
    }),
  });
}

async function getAdaptiveGrid(params: {
  epsilon: number;
  bounds?: Omit<BoundsResponse, "center">;
}): Promise<AdaptiveGridResponse> {
  return request<AdaptiveGridResponse>("/api/spatial/adaptive-grid", {
    method: "POST",
    body: JSON.stringify({
      epsilon: params.epsilon,
      bounds: params.bounds,
      use_cache: true,
    }),
  });
}

async function getComparison(params: {
  epsilon: number;
  bounds?: Omit<BoundsResponse, "center">;
}): Promise<ComparisonResponse> {
  return request<ComparisonResponse>("/api/spatial/comparison", {
    method: "POST",
    body: JSON.stringify({
      epsilon: params.epsilon,
      bounds: params.bounds,
    }),
  });
}

async function createSession(initialBudget = 5.0): Promise<SessionResponse> {
  const session = await request<SessionResponse>(
    "/api/sessions/",
    {
      method: "POST",
      body: JSON.stringify({ initial_budget: initialBudget }),
    },
    false
  );
  setSessionToken(session.session_token);
  return session;
}

async function getSessionStatus(): Promise<SessionStatus | null> {
  const token = getSessionToken();
  if (!token) return null;
  return request<SessionStatus>("/api/sessions/status");
}

function clearSession() {
  if (typeof window !== "undefined") {
    localStorage.removeItem(SESSION_TOKEN_KEY);
  }
}

export const api = {
  getBounds,
  getDecomposition,
  getAdaptiveGrid,
  getComparison,
  createSession,
  getSessionStatus,
  clearSession,
  getSessionToken,
};

