"use client";

import dynamic from "next/dynamic";

interface MapBounds {
  min_lon: number;
  max_lon: number;
  min_lat: number;
  max_lat: number;
  center: { lon: number; lat: number };
}

interface ComparisonMapProps {
  privtreeGeojson: GeoJSON.FeatureCollection | null;
  adaptiveGridGeojson: GeoJSON.FeatureCollection | null;
  bounds: MapBounds;
  isLoading?: boolean;
}

// Dynamically import the Leaflet map component to avoid SSR issues
const LeafletMap = dynamic(() => import("./LeafletMap"), {
  ssr: false,
  loading: () => (
    <div className="relative w-full h-full rounded-xl overflow-hidden border border-surface-700 bg-surface-800 flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
    </div>
  ),
});

export function ComparisonMap({
  privtreeGeojson,
  adaptiveGridGeojson,
  bounds,
  isLoading = false,
}: ComparisonMapProps) {
  return (
    <div className="w-full h-full flex gap-2">
      {/* PrivTree Panel (Left) */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center gap-2 mb-2 px-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500" />
          <span className="text-sm font-medium text-surface-200">
            PrivTree
          </span>
          <span className="text-xs text-surface-400">
            (Recursive Adaptive)
          </span>
        </div>
        <div className="flex-1 relative">
          <LeafletMap
            geojson={privtreeGeojson}
            bounds={bounds}
            isLoading={isLoading}
          />
          {/* Algorithm label overlay */}
          <div className="absolute top-3 left-3 z-[1000] bg-surface-900/90 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-surface-700">
            <span className="text-xs font-medium text-emerald-400">
              PrivTree
            </span>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="w-px bg-surface-700" />

      {/* Adaptive Grid Panel (Right) */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center gap-2 mb-2 px-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-sm font-medium text-surface-200">
            Adaptive Grid
          </span>
          <span className="text-xs text-surface-400">
            (Two-Level)
          </span>
        </div>
        <div className="flex-1 relative">
          <LeafletMap
            geojson={adaptiveGridGeojson}
            bounds={bounds}
            isLoading={isLoading}
          />
          {/* Algorithm label overlay */}
          <div className="absolute top-3 left-3 z-[1000] bg-surface-900/90 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-surface-700">
            <span className="text-xs font-medium text-blue-400">
              Adaptive Grid
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
