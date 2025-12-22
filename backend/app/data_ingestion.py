"""
Data ingestion utilities for loading Porto Taxi dataset.

The Porto Taxi dataset contains GPS trajectories as polylines.
Each polyline is a JSON array of [longitude, latitude] pairs.
We extract the first point of each trajectory as the pickup location.
"""

import json
import csv
import numpy as np
from datetime import datetime
from typing import Generator, Tuple, Optional
from pathlib import Path


def parse_polyline(polyline_str: str) -> Optional[list]:
    """
    Parse a polyline JSON string into a list of coordinates.
    
    Args:
        polyline_str: JSON string like '[[-8.618, 41.141], [-8.619, 41.142], ...]'
        
    Returns:
        List of [lon, lat] pairs or None if parsing fails
    """
    if not polyline_str or polyline_str == '[]':
        return None
    
    try:
        coords = json.loads(polyline_str)
        if coords and len(coords) > 0:
            return coords
    except json.JSONDecodeError:
        pass
    
    return None


def extract_pickup_point(polyline: list) -> Optional[Tuple[float, float]]:
    """
    Extract pickup location (first point) from trajectory.
    
    Args:
        polyline: List of [lon, lat] coordinate pairs
        
    Returns:
        Tuple of (longitude, latitude) for the pickup point
    """
    if polyline and len(polyline) > 0:
        first_point = polyline[0]
        if len(first_point) >= 2:
            return (float(first_point[0]), float(first_point[1]))
    return None


def is_valid_porto_coordinate(lon: float, lat: float) -> bool:
    """
    Check if coordinates are within Porto, Portugal bounds.
    
    Filters out obvious GPS errors and points outside the city.
    """
    # Porto metropolitan area approximate bounds
    return (
        -8.75 <= lon <= -8.45 and
        41.05 <= lat <= 41.30
    )


def load_taxi_data(
    csv_path: str,
    limit: Optional[int] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Generator[dict, None, None]:
    """
    Load taxi pickup points from the Porto CSV dataset.
    
    Args:
        csv_path: Path to the train.csv file
        limit: Maximum number of records to load (None for all)
        bounds: Optional (min_lon, max_lon, min_lat, max_lat) filter
        
    Yields:
        Dictionary with trip_id, longitude, latitude, timestamp
    """
    count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if limit and count >= limit:
                break
            
            # Skip rows with missing data
            if row.get('MISSING_DATA') == 'True':
                continue
            
            polyline = parse_polyline(row.get('POLYLINE', ''))
            if not polyline:
                continue
            
            pickup = extract_pickup_point(polyline)
            if not pickup:
                continue
            
            lon, lat = pickup
            
            # Validate coordinates
            if not is_valid_porto_coordinate(lon, lat):
                continue
            
            # Apply bounds filter if specified
            if bounds:
                min_lon, max_lon, min_lat, max_lat = bounds
                if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                    continue
            
            # Parse timestamp
            timestamp = None
            if row.get('TIMESTAMP'):
                try:
                    timestamp = datetime.fromtimestamp(int(row['TIMESTAMP']))
                except (ValueError, OSError):
                    pass
            
            count += 1
            yield {
                'trip_id': row.get('TRIP_ID', ''),
                'longitude': lon,
                'latitude': lat,
                'timestamp': timestamp,
            }


def load_points_as_array(
    csv_path: str,
    limit: Optional[int] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Load taxi pickup points as a NumPy array for PrivTree.
    
    Args:
        csv_path: Path to the train.csv file
        limit: Maximum number of records to load
        bounds: Optional spatial filter
        
    Returns:
        Nx2 NumPy array of (longitude, latitude) pairs
    """
    points = []
    
    for record in load_taxi_data(csv_path, limit=limit, bounds=bounds):
        points.append([record['longitude'], record['latitude']])
    
    if not points:
        return np.array([]).reshape(0, 2)
    
    return np.array(points)


def get_data_statistics(csv_path: str, sample_size: int = 10000) -> dict:
    """
    Compute statistics about the taxi dataset.
    
    Args:
        csv_path: Path to the train.csv file
        sample_size: Number of records to sample for statistics
        
    Returns:
        Dictionary with min/max coordinates, count, etc.
    """
    points = load_points_as_array(csv_path, limit=sample_size)
    
    if len(points) == 0:
        return {"error": "No valid points found"}
    
    return {
        "sample_size": len(points),
        "min_longitude": float(np.min(points[:, 0])),
        "max_longitude": float(np.max(points[:, 0])),
        "min_latitude": float(np.min(points[:, 1])),
        "max_latitude": float(np.max(points[:, 1])),
        "center_longitude": float(np.mean(points[:, 0])),
        "center_latitude": float(np.mean(points[:, 1])),
    }


if __name__ == "__main__":
    # Test data loading
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "../../../train.csv"
    
    print(f"Loading data from {csv_path}...")
    stats = get_data_statistics(csv_path)
    print(f"Statistics: {stats}")
    
    points = load_points_as_array(csv_path, limit=1000)
    print(f"Loaded {len(points)} points")
    if len(points) > 0:
        print(f"First point: {points[0]}")
        print(f"Last point: {points[-1]}")

