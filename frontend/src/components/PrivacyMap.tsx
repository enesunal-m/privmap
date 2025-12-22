"use client";

import dynamic from "next/dynamic";

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

// Dynamically import the entire Leaflet map component to avoid SSR issues
// This is the recommended pattern for react-leaflet with Next.js
const LeafletMap = dynamic(() => import("./LeafletMap"), {
  ssr: false,
  loading: () => (
    <div className="relative w-full h-full rounded-xl overflow-hidden border border-surface-700 bg-surface-800 flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
    </div>
  ),
});

export function PrivacyMap({
  geojson,
  bounds,
  isLoading = false,
}: PrivacyMapProps) {
  return <LeafletMap geojson={geojson} bounds={bounds} isLoading={isLoading} />;
}
