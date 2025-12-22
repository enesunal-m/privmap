"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import type { Layer, PathOptions, GeoJSON as LeafletGeoJSON } from "leaflet";
import { getHeatmapColor } from "@/lib/utils";
import "leaflet/dist/leaflet.css";

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

// Component to fit bounds when they change
function FitBounds({ bounds }: { bounds: MapBounds }) {
  const map = useMap();

  useEffect(() => {
    map.fitBounds([
      [bounds.min_lat, bounds.min_lon],
      [bounds.max_lat, bounds.max_lon],
    ]);
  }, [map, bounds]);

  return null;
}

// Style function for GeoJSON features
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getFeatureStyle(feature: any): PathOptions {
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
  const counts: number[] = geojson.features
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .map((f: any) => f.properties?.count || 0)
    .filter((c: number) => c > 0);

  if (counts.length === 0) return geojson;

  // Use 95th percentile for better color distribution
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

// Inner map component that handles GeoJSON updates
function MapContent({
  geojson,
  bounds,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  geojson: any;
  bounds: MapBounds;
}) {
  const geoJsonRef = useRef<LeafletGeoJSON | null>(null);

  // Process GeoJSON for visualization
  const processedGeoJSON = useMemo(() => {
    if (!geojson) return null;
    return processGeoJSON(geojson);
  }, [geojson]);

  // Tooltip for each feature
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onEachFeature = (feature: any, layer: Layer) => {
    if (feature.properties) {
      const count =
        feature.properties.count?.toFixed?.(1) ??
        feature.properties.count ??
        "0";
      const depth = feature.properties.depth || 0;

      layer.bindTooltip(
        `<div class="font-mono text-xs">
          <div><strong>Count:</strong> ${count}</div>
          <div><strong>Depth:</strong> ${depth}</div>
        </div>`,
        {
          sticky: true,
          className:
            "!bg-surface-800 !text-surface-100 !border-surface-600 !rounded-lg !px-2 !py-1",
        }
      );

      // Hover effect
      layer.on({
        mouseover: (e) => {
          const target = e.target;
          target.setStyle({
            weight: 2,
            color: "#22c55e",
            fillOpacity: 0.9,
          });
        },
        mouseout: (e) => {
          geoJsonRef.current?.resetStyle(e.target);
        },
      });
    }
  };

  // Generate a stable key for GeoJSON based on feature count
  const geoJsonKey = useMemo(() => {
    if (!processedGeoJSON) return "empty";
    return `geojson-${processedGeoJSON.features.length}-${Date.now()}`;
  }, [processedGeoJSON]);

  return (
    <>
      <TileLayer
        attribution='&copy; <a href="https://carto.com/">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />

      <FitBounds bounds={bounds} />

      {processedGeoJSON && (
        <GeoJSON
          ref={geoJsonRef}
          key={geoJsonKey}
          data={processedGeoJSON}
          style={getFeatureStyle}
          onEachFeature={onEachFeature}
        />
      )}
    </>
  );
}

export function PrivacyMap({
  geojson,
  bounds,
  isLoading = false,
}: PrivacyMapProps) {
  // Use state to track if map is mounted (prevents SSR issues)
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Generate a stable key for the map container based on bounds
  const mapKey = useMemo(() => {
    return `map-${bounds.min_lon}-${bounds.max_lon}-${bounds.min_lat}-${bounds.max_lat}`;
  }, [bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat]);

  if (!isMounted) {
    return (
      <div className="relative w-full h-full rounded-xl overflow-hidden border border-surface-700 bg-surface-800 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden border border-surface-700">
      <MapContainer
        key={mapKey}
        center={[bounds.center.lat, bounds.center.lon]}
        zoom={13}
        className="w-full h-full"
        zoomControl={true}
      >
        <MapContent geojson={geojson} bounds={bounds} />
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
