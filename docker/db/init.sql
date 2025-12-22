-- PrivMap Database Initialization Script
-- This script runs automatically when the PostgreSQL container starts for the first time

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'PostGIS extensions enabled successfully';
END $$;

-- Create taxi_pickups table
CREATE TABLE IF NOT EXISTS taxi_pickups (
    id SERIAL PRIMARY KEY,
    trip_id VARCHAR(50),
    longitude DOUBLE PRECISION NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP,
    location GEOMETRY(Point, 4326)
);

-- Create spatial index for efficient queries
CREATE INDEX IF NOT EXISTS idx_taxi_pickups_location 
ON taxi_pickups USING GIST(location);

-- Create B-tree index on longitude/latitude for fast range queries
CREATE INDEX IF NOT EXISTS idx_taxi_pickups_lon_lat 
ON taxi_pickups(longitude, latitude);

CREATE INDEX IF NOT EXISTS idx_taxi_pickups_trip_id 
ON taxi_pickups(trip_id);

-- Create privacy_sessions table
CREATE TABLE IF NOT EXISTS privacy_sessions (
    id SERIAL PRIMARY KEY,
    session_token VARCHAR(64) UNIQUE NOT NULL,
    initial_budget DOUBLE PRECISION DEFAULT 5.0,
    remaining_budget DOUBLE PRECISION DEFAULT 5.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_query_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    query_history JSONB DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_privacy_sessions_token 
ON privacy_sessions(session_token);

-- Create cached_decompositions table
CREATE TABLE IF NOT EXISTS cached_decompositions (
    id SERIAL PRIMARY KEY,
    epsilon DOUBLE PRECISION NOT NULL,
    min_lon DOUBLE PRECISION NOT NULL,
    max_lon DOUBLE PRECISION NOT NULL,
    min_lat DOUBLE PRECISION NOT NULL,
    max_lat DOUBLE PRECISION NOT NULL,
    geojson JSONB NOT NULL,
    statistics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    point_count INTEGER
);

-- Create index for cache lookups
CREATE INDEX IF NOT EXISTS idx_cached_decompositions_params 
ON cached_decompositions(epsilon, min_lon, max_lon, min_lat, max_lat);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO privmap;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO privmap;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'PrivMap database initialization complete';
END $$;

