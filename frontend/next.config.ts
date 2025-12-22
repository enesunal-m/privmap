import type { NextConfig } from "next";

const nextConfig: NextConfig = {
    // Disabled due to incompatibility with Leaflet map initialization
    // Leaflet stores state on DOM elements which conflicts with Strict Mode's double-mounting
    reactStrictMode: false,

    // Enable standalone output for Docker builds
    output: "standalone",

    // API proxy to backend during development
    async rewrites() {
        return [
            {
                source: "/api/:path*",
                destination: "http://localhost:8000/api/:path*",
            },
        ];
    },
};

export default nextConfig;

