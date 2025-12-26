"use client";

import { useEffect, useMemo, useCallback, useRef, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import type { Map as LeafletMapType } from "leaflet";
import { getHeatmapColor } from "@/lib/utils";
import "leaflet/dist/leaflet.css";

// Performance: Limit cells to render for smooth performance
const MAX_CELLS_TO_RENDER = 10000;

interface MapBounds {
  min_lon: number;
  max_lon: number;
  min_lat: number;
  max_lat: number;
  center: { lon: number; lat: number };
}

interface LeafletMapProps {
  geojson: GeoJSON.FeatureCollection | null;
  bounds: MapBounds;
  isLoading?: boolean;
}

// Component to handle map bounds updates
function MapController({ bounds }: { bounds: MapBounds }) {
  const map = useMap();

  useEffect(() => {
    if (map) {
      map.fitBounds([
        [bounds.min_lat, bounds.min_lon],
        [bounds.max_lat, bounds.max_lon],
      ]);
    }
  }, [map, bounds]);

  return null;
}

// Style function for GeoJSON features
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getFeatureStyle(feature: any): Record<string, unknown> {
  if (!feature?.properties) {
    return {
      fillColor: "#27272a",
      fillOpacity: 0.1,
      color: "#3f3f46",
      weight: 0.5,
    };
  }

  const count = feature.properties.count || 0;
  const maxCount = feature.properties.maxCount || 1;
  const normalized = Math.min(1, count / maxCount);

  // Very low density cells should be nearly invisible
  if (normalized < 0.05) {
    return {
      fillColor: "#27272a",
      fillOpacity: 0.1,
      color: "#3f3f46",
      weight: 0.3,
      opacity: 0.3,
    };
  }

  // Low density cells - subtle visibility
  if (normalized < 0.15) {
    return {
      fillColor: getHeatmapColor(normalized, 1),
      fillOpacity: 0.25 + normalized * 2,
      color: "#3f3f46",
      weight: 0.3,
      opacity: 0.4,
    };
  }

  // Medium to high density - progressively more visible
  return {
    fillColor: getHeatmapColor(normalized, 1),
    fillOpacity: 0.5 + normalized * 0.4,
    color: normalized > 0.5 ? "#52525b" : "#3f3f46",
    weight: normalized > 0.5 ? 0.8 : 0.5,
    opacity: 0.6 + normalized * 0.3,
  };
}

// Process GeoJSON to add maxCount for normalization and limit cells for performance
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function processGeoJSON(geojson: any): { data: any; truncated: boolean } {
  if (!geojson?.features) return { data: geojson, truncated: false };

  let features = geojson.features;
  let truncated = false;

  // Performance optimization: limit number of cells rendered
  if (features.length > MAX_CELLS_TO_RENDER) {
    // Sort by count (descending) and keep the most significant cells
    features = [...features]
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .sort(
        (a: any, b: any) =>
          (b.properties?.count || 0) - (a.properties?.count || 0)
      )
      .slice(0, MAX_CELLS_TO_RENDER);
    truncated = true;
  }

  const counts: number[] = features
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .map((f: any) => f.properties?.count || 0)
    .filter((c: number) => c > 0);

  if (counts.length === 0) return { data: geojson, truncated };

  counts.sort((a, b) => a - b);
  const maxCount =
    counts[Math.floor(counts.length * 0.95)] || Math.max(...counts);

  return {
    data: {
      ...geojson,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      features: features.map((feature: any) => ({
        ...feature,
        properties: {
          ...feature.properties,
          maxCount,
        },
      })),
    },
    truncated,
  };
}

export default function LeafletMap({
  geojson,
  bounds,
  isLoading = false,
}: LeafletMapProps) {
  const mapRef = useRef<LeafletMapType | null>(null);
  // Generate a unique key on each mount to ensure fresh map instance
  const [mapKey] = useState(() => `map-${Date.now()}-${Math.random()}`);

  // Cleanup map on unmount
  useEffect(() => {
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Process GeoJSON for visualization
  const { processedGeoJSON, isTruncated } = useMemo(() => {
    if (!geojson) return { processedGeoJSON: null, isTruncated: false };
    const result = processGeoJSON(geojson);
    return { processedGeoJSON: result.data, isTruncated: result.truncated };
  }, [geojson]);

  // Generate unique key for GeoJSON layer
  const geoJsonKey = useMemo(() => {
    if (!processedGeoJSON) return "empty";
    return `geojson-${processedGeoJSON.features?.length || 0}-${Date.now()}`;
  }, [processedGeoJSON]);

  // Tooltip and hover handler
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onEachFeature = useCallback((feature: any, layer: any) => {
    if (feature.properties) {
      const count =
        feature.properties.count?.toFixed?.(1) ??
        feature.properties.count ??
        "0";
      const depth = feature.properties.depth || 0;

      layer.bindTooltip(
        `<div style="font-family: monospace; font-size: 12px;">
          <div><strong>Count:</strong> ${count}</div>
          <div><strong>Depth:</strong> ${depth}</div>
        </div>`,
        { sticky: true }
      );

      layer.on({
        mouseover: (e: L.LeafletMouseEvent) => {
          e.target.setStyle({
            weight: 2,
            color: "#22c55e",
            fillOpacity: 0.9,
          });
        },
        mouseout: (e: L.LeafletMouseEvent) => {
          e.target.setStyle(getFeatureStyle(feature));
        },
      });
    }
  }, []);

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden border border-surface-700">
      <MapContainer
        key={mapKey}
        center={[bounds.center.lat, bounds.center.lon]}
        zoom={13}
        className="w-full h-full"
        zoomControl={true}
        ref={mapRef}
      >
        <MapController bounds={bounds} />
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {processedGeoJSON && (
          <GeoJSON
            key={geoJsonKey}
            data={processedGeoJSON}
            style={getFeatureStyle}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-surface-900/80 backdrop-blur-sm flex items-center justify-center z-[1000]">
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-surface-300 text-sm">
              Computing private decomposition...
            </span>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 right-4 z-[1000] bg-surface-900/95 backdrop-blur-sm rounded-lg p-3 border border-surface-700 shadow-lg">
        <div className="text-xs font-medium text-surface-300 mb-2">
          Taxi Pickup Density
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-surface-500">Sparse</span>
          <div className="w-28 h-2.5 rounded-full density-gradient" />
          <span className="text-[10px] text-surface-500">Dense</span>
        </div>
        <div className="text-[10px] text-surface-500 mt-1.5 flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-surface-700/50" />
          <span>Empty areas hidden</span>
        </div>
        {isTruncated && (
          <div className="text-[10px] text-amber-400 mt-1.5">
            Showing top {MAX_CELLS_TO_RENDER} cells for performance
          </div>
        )}
      </div>

      {/* No data message */}
      {!isLoading && !geojson && (
        <div className="absolute inset-0 flex items-center justify-center z-[1000] pointer-events-none">
          <div className="text-surface-400 text-center">
            <p>Click &quot;Run Query&quot; to visualize</p>
            <p className="text-sm">private spatial decomposition</p>
          </div>
        </div>
      )}
    </div>
  );
}
