"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { getHeatmapColor } from "@/lib/utils";
import "leaflet/dist/leaflet.css";

// Dynamically import react-leaflet components to avoid SSR issues
const MapContainerDynamic = dynamic(
  () => import("react-leaflet").then((mod) => mod.MapContainer),
  { ssr: false }
);

const TileLayerDynamic = dynamic(
  () => import("react-leaflet").then((mod) => mod.TileLayer),
  { ssr: false }
);

const GeoJSONDynamic = dynamic(
  () => import("react-leaflet").then((mod) => mod.GeoJSON),
  { ssr: false }
);

interface MapBounds {
  min_lon: number;
  max_lon: number;
  min_lat: number;
  max_lat: number;
  center: { lon: number; lat: number };
}

interface PrivacyMapProps {
  geojson: GeoJSON.FeatureCollection | null;
  bounds: MapBounds;
  isLoading?: boolean;
  onBoundsChange?: (bounds: MapBounds) => void;
}

// Style function for GeoJSON features
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getFeatureStyle(feature: any): Record<string, unknown> {
  if (!feature?.properties) {
    return {
      fillColor: "#3f3f46",
      fillOpacity: 0.3,
      color: "#52525b",
      weight: 1,
    };
  }

  const count = feature.properties.count || 0;
  const maxCount = feature.properties.maxCount || 1;
  const normalized = Math.min(1, count / maxCount);

  return {
    fillColor: getHeatmapColor(normalized, 1),
    fillOpacity: 0.6 + normalized * 0.3,
    color: "#18181b",
    weight: 0.5,
    opacity: 0.8,
  };
}

// Process GeoJSON to add maxCount for normalization
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function processGeoJSON(geojson: any): any {
  if (!geojson?.features) return geojson;

  const counts: number[] = geojson.features
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .map((f: any) => f.properties?.count || 0)
    .filter((c: number) => c > 0);

  if (counts.length === 0) return geojson;

  counts.sort((a, b) => a - b);
  const maxCount =
    counts[Math.floor(counts.length * 0.95)] || Math.max(...counts);

  return {
    ...geojson,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    features: geojson.features.map((feature: any) => ({
      ...feature,
      properties: {
        ...feature.properties,
        maxCount,
      },
    })),
  };
}

export function PrivacyMap({
  geojson,
  bounds,
  isLoading = false,
}: PrivacyMapProps) {
  const [isClient, setIsClient] = useState(false);
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Only render on client side
  useEffect(() => {
    setIsClient(true);

    return () => {
      // Cleanup map instance on unmount
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Process GeoJSON for visualization
  const processedGeoJSON = useMemo(() => {
    if (!geojson) return null;
    return processGeoJSON(geojson);
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

  // Don't render map on server
  if (!isClient) {
    return (
      <div className="relative w-full h-full rounded-xl overflow-hidden border border-surface-700 bg-surface-800 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full rounded-xl overflow-hidden border border-surface-700"
    >
      <MapContainerDynamic
        center={[bounds.center.lat, bounds.center.lon]}
        zoom={13}
        className="w-full h-full"
        zoomControl={true}
        ref={mapRef}
        whenReady={() => {
          // Fit bounds when map is ready
          if (mapRef.current) {
            mapRef.current.fitBounds([
              [bounds.min_lat, bounds.min_lon],
              [bounds.max_lat, bounds.max_lon],
            ]);
          }
        }}
      >
        <TileLayerDynamic
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {processedGeoJSON && (
          <GeoJSONDynamic
            key={geoJsonKey}
            data={processedGeoJSON}
            style={getFeatureStyle}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainerDynamic>

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
      <div className="absolute bottom-4 right-4 z-[1000] bg-surface-900/90 backdrop-blur-sm rounded-lg p-3 border border-surface-700">
        <div className="text-xs text-surface-400 mb-2">Density</div>
        <div className="flex items-center gap-1">
          <span className="text-xs text-surface-500">Low</span>
          <div className="w-32 h-3 rounded-full privacy-gradient" />
          <span className="text-xs text-surface-500">High</span>
        </div>
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
