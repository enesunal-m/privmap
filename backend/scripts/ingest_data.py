#!/usr/bin/env python3
"""
Data Ingestion Script for PrivMap

Loads Porto taxi pickup data from train.csv into PostgreSQL/PostGIS database.
Extracts the first point of each trajectory as the pickup location.

Usage:
    python scripts/ingest_data.py --csv ../train.csv
    python scripts/ingest_data.py --csv ../train.csv --limit 100000
    python scripts/ingest_data.py --csv ../train.csv --batch-size 5000
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


def parse_polyline(polyline_str: str) -> Optional[list]:
    """Parse polyline JSON string into coordinates list."""
    if not polyline_str or polyline_str == '[]':
        return None
    try:
        coords = json.loads(polyline_str)
        if coords and len(coords) > 0:
            return coords
    except json.JSONDecodeError:
        pass
    return None


def is_valid_coordinate(lon: float, lat: float) -> bool:
    """Check if coordinates are within Porto area."""
    return -8.75 <= lon <= -8.45 and 41.05 <= lat <= 41.30


def load_csv_records(
    csv_path: str,
    limit: Optional[int] = None
) -> Generator[dict, None, None]:
    """
    Generator that yields pickup records from the CSV.
    
    Args:
        csv_path: Path to train.csv
        limit: Maximum number of records to yield
        
    Yields:
        Dict with trip_id, longitude, latitude, timestamp
    """
    import csv
    
    count = 0
    skipped = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if limit and count >= limit:
                break
            
            # Skip rows with missing data
            if row.get('MISSING_DATA') == 'True':
                skipped += 1
                continue
            
            polyline = parse_polyline(row.get('POLYLINE', ''))
            if not polyline or len(polyline) == 0:
                skipped += 1
                continue
            
            # Get first point as pickup location
            first_point = polyline[0]
            if len(first_point) < 2:
                skipped += 1
                continue
            
            lon, lat = float(first_point[0]), float(first_point[1])
            
            # Validate coordinates
            if not is_valid_coordinate(lon, lat):
                skipped += 1
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
    
    print(f"Processed {count} valid records, skipped {skipped} invalid")


async def create_tables(engine):
    """Ensure tables exist."""
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS taxi_pickups (
                id SERIAL PRIMARY KEY,
                trip_id VARCHAR(50),
                longitude DOUBLE PRECISION NOT NULL,
                latitude DOUBLE PRECISION NOT NULL,
                timestamp TIMESTAMP,
                location GEOMETRY(Point, 4326)
            )
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_taxi_pickups_location 
            ON taxi_pickups USING GIST(location)
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_taxi_pickups_trip_id 
            ON taxi_pickups(trip_id)
        """))
    
    print("Tables created/verified")


async def clear_existing_data(engine):
    """Clear existing data from taxi_pickups table."""
    async with engine.begin() as conn:
        result = await conn.execute(text("DELETE FROM taxi_pickups"))
        print(f"Cleared {result.rowcount} existing records")


async def insert_batch(session: AsyncSession, records: list):
    """Insert a batch of records into the database."""
    if not records:
        return 0
    
    # Build bulk insert query
    values = []
    params = {}
    
    for i, rec in enumerate(records):
        values.append(
            f"(:trip_id_{i}, :lon_{i}, :lat_{i}, :ts_{i}, "
            f"ST_SetSRID(ST_MakePoint(:lon_{i}, :lat_{i}), 4326))"
        )
        params[f'trip_id_{i}'] = rec['trip_id']
        params[f'lon_{i}'] = rec['longitude']
        params[f'lat_{i}'] = rec['latitude']
        params[f'ts_{i}'] = rec['timestamp']
    
    query = text(f"""
        INSERT INTO taxi_pickups (trip_id, longitude, latitude, timestamp, location)
        VALUES {', '.join(values)}
    """)
    
    await session.execute(query, params)
    await session.commit()
    
    return len(records)


async def ingest_data(
    database_url: str,
    csv_path: str,
    limit: Optional[int] = None,
    batch_size: int = 1000,
    clear_existing: bool = True
):
    """
    Main ingestion function.
    
    Args:
        database_url: PostgreSQL connection URL
        csv_path: Path to train.csv
        limit: Max records to ingest
        batch_size: Records per batch insert
        clear_existing: Whether to clear existing data first
    """
    print(f"Connecting to database...")
    engine = create_async_engine(database_url, echo=False)
    
    # Create tables
    await create_tables(engine)
    
    # Clear existing data if requested
    if clear_existing:
        await clear_existing_data(engine)
    
    # Create session
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    print(f"Loading data from {csv_path}...")
    print(f"Batch size: {batch_size}, Limit: {limit or 'None'}")
    
    total_inserted = 0
    batch = []
    
    async with async_session() as session:
        for record in load_csv_records(csv_path, limit=limit):
            batch.append(record)
            
            if len(batch) >= batch_size:
                inserted = await insert_batch(session, batch)
                total_inserted += inserted
                print(f"Inserted {total_inserted} records...", end='\r')
                batch = []
        
        # Insert remaining records
        if batch:
            inserted = await insert_batch(session, batch)
            total_inserted += inserted
    
    print(f"\nCompleted! Total records inserted: {total_inserted}")
    
    # Show sample
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT COUNT(*) as count,
                   MIN(longitude) as min_lon, MAX(longitude) as max_lon,
                   MIN(latitude) as min_lat, MAX(latitude) as max_lat
            FROM taxi_pickups
        """))
        row = result.fetchone()
        print(f"\nDatabase statistics:")
        print(f"  Total records: {row[0]}")
        print(f"  Longitude range: [{row[1]:.4f}, {row[2]:.4f}]")
        print(f"  Latitude range: [{row[3]:.4f}, {row[4]:.4f}]")
    
    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Porto taxi data into PostgreSQL/PostGIS"
    )
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="Path to train.csv file"
    )
    parser.add_argument(
        "--database-url", "-d",
        default="postgresql+asyncpg://privmap:privmap_secret@localhost:5432/privmap",
        help="Database connection URL"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of records to import"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1000,
        help="Batch size for inserts (default: 1000)"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing data before import"
    )
    
    args = parser.parse_args()
    
    # Validate CSV path
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Run ingestion
    asyncio.run(ingest_data(
        database_url=args.database_url,
        csv_path=str(csv_path),
        limit=args.limit,
        batch_size=args.batch_size,
        clear_existing=not args.no_clear,
    ))


if __name__ == "__main__":
    main()

