"""
Data collection module for GreenPath.
Handles fetching data from Google Earth Engine and OpenStreetMap.
"""

import ee
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import h3
from shapely.geometry import Polygon, Point, box
import pickle
import os
from datetime import datetime
import warnings

from config import (
    BBOX, H3_RESOLUTION, DATE_START, DATE_END,
    CLOUD_COVER_MAX, CACHE_DIR, GEE_PROJECT
)

warnings.filterwarnings('ignore')


def initialize_gee():
    """
    Initialize Google Earth Engine.
    Requires prior authentication via `earthengine authenticate`.
    """
    try:
        if GEE_PROJECT:
            ee.Initialize(project=GEE_PROJECT)
        else:
            ee.Initialize()
        print("[OK] Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"[FAIL] GEE initialization failed: {e}")
        print("Please run 'earthengine authenticate' first")
        return False


def get_study_area_geometry():
    """Get the study area as an EE geometry."""
    return ee.Geometry.Rectangle([
        BBOX['min_lon'], BBOX['min_lat'],
        BBOX['max_lon'], BBOX['max_lat']
    ])


def get_hexagon_grid():
    """
    Generate H3 hexagon grid covering the study area.
    Returns GeoDataFrame with hexagon geometries.
    """
    cache_path = os.path.join(CACHE_DIR, 'hexagon_grid.pkl')

    if os.path.exists(cache_path):
        print("Loading hexagon grid from cache...")
        return pickle.load(open(cache_path, 'rb'))

    print("Generating H3 hexagon grid...")

    # Get all hexagons covering the bounding box
    # Create a dense point grid and get unique hexagons
    lats = np.linspace(BBOX['min_lat'], BBOX['max_lat'], 100)
    lons = np.linspace(BBOX['min_lon'], BBOX['max_lon'], 100)

    hex_ids = set()
    for lat in lats:
        for lon in lons:
            hex_id = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
            hex_ids.add(hex_id)

    # Convert hexagons to polygons
    hex_data = []
    for hex_id in hex_ids:
        boundary = h3.cell_to_boundary(hex_id)
        # h3 returns (lat, lon), convert to (lon, lat) for shapely
        coords = [(lon, lat) for lat, lon in boundary]
        polygon = Polygon(coords)
        center = h3.cell_to_latlng(hex_id)

        hex_data.append({
            'hex_id': hex_id,
            'geometry': polygon,
            'center_lat': center[0],
            'center_lon': center[1]
        })

    gdf = gpd.GeoDataFrame(hex_data, crs='EPSG:4326')

    # Cache the result
    os.makedirs(CACHE_DIR, exist_ok=True)
    pickle.dump(gdf, open(cache_path, 'wb'))

    print(f"[OK] Generated {len(gdf)} hexagons")
    return gdf


def fetch_sentinel2_ndvi():
    """
    Fetch Sentinel-2 imagery and calculate NDVI.
    Returns mean NDVI values for the study area.
    """
    cache_path = os.path.join(CACHE_DIR, 'ndvi_data.pkl')

    if os.path.exists(cache_path):
        print("Loading NDVI data from cache...")
        return pickle.load(open(cache_path, 'rb'))

    print("Fetching Sentinel-2 NDVI data from GEE...")

    try:
        study_area = get_study_area_geometry()

        # Get Sentinel-2 Surface Reflectance collection
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(study_area) \
            .filterDate(DATE_START, DATE_END) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER_MAX))

        # Calculate NDVI for each image
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)

        s2_ndvi = s2.map(add_ndvi)

        # Get median NDVI
        ndvi_median = s2_ndvi.select('NDVI').median()

        # Sample the NDVI at a grid of points
        # Create sampling points
        scale = 10  # 10m resolution for Sentinel-2

        ndvi_image = ndvi_median.clip(study_area)

        # Get the NDVI values as a dictionary
        # Sample at regular intervals
        sample_points = ndvi_image.sample(
            region=study_area,
            scale=scale,
            numPixels=5000,
            geometries=True
        )

        # Convert to features
        features = sample_points.getInfo()['features']

        ndvi_points = []
        for f in features:
            coords = f['geometry']['coordinates']
            ndvi_val = f['properties'].get('NDVI', 0)
            ndvi_points.append({
                'lon': coords[0],
                'lat': coords[1],
                'ndvi': ndvi_val
            })

        ndvi_df = pd.DataFrame(ndvi_points)

        # Cache the result
        os.makedirs(CACHE_DIR, exist_ok=True)
        pickle.dump(ndvi_df, open(cache_path, 'wb'))

        print(f"[OK] Fetched {len(ndvi_df)} NDVI samples")
        return ndvi_df

    except Exception as e:
        print(f"[FAIL] Error fetching NDVI: {e}")
        return generate_synthetic_ndvi()


def fetch_landsat_lst():
    """
    Fetch Landsat 8/9 thermal data and calculate Land Surface Temperature.
    Returns LST values for the study area.
    """
    cache_path = os.path.join(CACHE_DIR, 'lst_data.pkl')

    if os.path.exists(cache_path):
        print("Loading LST data from cache...")
        return pickle.load(open(cache_path, 'rb'))

    print("Fetching Landsat LST data from GEE...")

    try:
        study_area = get_study_area_geometry()

        # Get Landsat 8 Collection 2 Level 2 (includes surface temperature)
        landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(study_area) \
            .filterDate(DATE_START, DATE_END) \
            .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))

        # Convert thermal band to Celsius
        def calc_lst(image):
            # ST_B10 is surface temperature band, scale factor is 0.00341802, offset is 149
            thermal = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
            return thermal.rename('LST')

        lst_collection = landsat.map(calc_lst)
        lst_median = lst_collection.median()

        # Sample the LST
        lst_image = lst_median.clip(study_area)

        sample_points = lst_image.sample(
            region=study_area,
            scale=30,  # 30m resolution for Landsat
            numPixels=3000,
            geometries=True
        )

        features = sample_points.getInfo()['features']

        lst_points = []
        for f in features:
            coords = f['geometry']['coordinates']
            lst_val = f['properties'].get('LST', 35)
            lst_points.append({
                'lon': coords[0],
                'lat': coords[1],
                'lst': lst_val
            })

        lst_df = pd.DataFrame(lst_points)

        # Cache the result
        os.makedirs(CACHE_DIR, exist_ok=True)
        pickle.dump(lst_df, open(cache_path, 'wb'))

        print(f"[OK] Fetched {len(lst_df)} LST samples")
        return lst_df

    except Exception as e:
        print(f"[FAIL] Error fetching LST: {e}")
        return generate_synthetic_lst()


def fetch_srtm_elevation():
    """
    Fetch SRTM Digital Elevation Model for slope calculation.
    Returns elevation data for the study area.
    """
    cache_path = os.path.join(CACHE_DIR, 'elevation_data.pkl')

    if os.path.exists(cache_path):
        print("Loading elevation data from cache...")
        return pickle.load(open(cache_path, 'rb'))

    print("Fetching SRTM elevation data from GEE...")

    try:
        study_area = get_study_area_geometry()

        # Get SRTM elevation
        srtm = ee.Image('USGS/SRTMGL1_003')
        elevation = srtm.select('elevation')

        # Calculate slope in degrees
        slope = ee.Terrain.slope(elevation)

        # Sample the slope
        slope_image = slope.clip(study_area)

        sample_points = slope_image.sample(
            region=study_area,
            scale=30,
            numPixels=3000,
            geometries=True
        )

        features = sample_points.getInfo()['features']

        slope_points = []
        for f in features:
            coords = f['geometry']['coordinates']
            slope_val = f['properties'].get('slope', 0)
            slope_points.append({
                'lon': coords[0],
                'lat': coords[1],
                'slope': slope_val
            })

        slope_df = pd.DataFrame(slope_points)

        # Cache the result
        os.makedirs(CACHE_DIR, exist_ok=True)
        pickle.dump(slope_df, open(cache_path, 'wb'))

        print(f"[OK] Fetched {len(slope_df)} slope samples")
        return slope_df

    except Exception as e:
        print(f"[FAIL] Error fetching elevation: {e}")
        return generate_synthetic_slope()


def fetch_osm_roads():
    """
    Fetch OpenStreetMap road network using OSMnx.
    Returns a NetworkX graph of the road network.
    """
    cache_path = os.path.join(CACHE_DIR, 'road_network.pkl')

    if os.path.exists(cache_path):
        print("Loading road network from cache...")
        return pickle.load(open(cache_path, 'rb'))

    print("Fetching OSM road network...")

    try:
        # Define the bounding box (new OSMnx API uses bbox tuple: (left, bottom, right, top))
        bbox = (BBOX['min_lon'], BBOX['min_lat'], BBOX['max_lon'], BBOX['max_lat'])

        # Get the road network for walking
        G = ox.graph_from_bbox(
            bbox=bbox,
            network_type='walk',
            simplify=True
        )

        # Project to UTM for accurate distance calculations
        G = ox.project_graph(G)

        # Cache the result
        os.makedirs(CACHE_DIR, exist_ok=True)
        pickle.dump(G, open(cache_path, 'wb'))

        print(f"[OK] Fetched road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    except Exception as e:
        print(f"[FAIL] Error fetching road network: {e}")
        raise


def fetch_osm_buildings():
    """
    Fetch OpenStreetMap building footprints.
    Returns GeoDataFrame with building geometries and heights.
    """
    cache_path = os.path.join(CACHE_DIR, 'buildings.pkl')

    if os.path.exists(cache_path):
        print("Loading buildings from cache...")
        return pickle.load(open(cache_path, 'rb'))

    print("Fetching OSM building footprints...")

    try:
        # Define the bounding box (new OSMnx API uses bbox tuple: (left, bottom, right, top))
        bbox = (BBOX['min_lon'], BBOX['min_lat'], BBOX['max_lon'], BBOX['max_lat'])

        # Get building footprints
        tags = {'building': True}
        buildings = ox.features_from_bbox(
            bbox=bbox,
            tags=tags
        )

        # Filter to polygons only
        buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]

        # Extract height information if available
        if 'height' in buildings.columns:
            buildings['height_m'] = pd.to_numeric(
                buildings['height'].str.replace(' m', ''),
                errors='coerce'
            )
        else:
            buildings['height_m'] = np.nan

        # Estimate height from building:levels if height not available
        if 'building:levels' in buildings.columns:
            levels = pd.to_numeric(buildings['building:levels'], errors='coerce')
            buildings['height_m'] = buildings['height_m'].fillna(levels * 3)  # 3m per floor

        # Default height for unknown buildings
        buildings['height_m'] = buildings['height_m'].fillna(6)  # 2 floors default

        # Keep only necessary columns
        buildings = buildings[['geometry', 'height_m']].reset_index(drop=True)

        # Cache the result
        os.makedirs(CACHE_DIR, exist_ok=True)
        pickle.dump(buildings, open(cache_path, 'wb'))

        print(f"[OK] Fetched {len(buildings)} buildings")
        return buildings

    except Exception as e:
        print(f"[FAIL] Error fetching buildings: {e}")
        # Return empty GeoDataFrame
        return gpd.GeoDataFrame(columns=['geometry', 'height_m'], crs='EPSG:4326')


# Synthetic data generators for fallback when GEE is unavailable
def generate_synthetic_ndvi():
    """Generate synthetic NDVI data for testing."""
    print("Generating synthetic NDVI data...")

    np.random.seed(42)
    n_points = 5000

    lats = np.random.uniform(BBOX['min_lat'], BBOX['max_lat'], n_points)
    lons = np.random.uniform(BBOX['min_lon'], BBOX['max_lon'], n_points)

    # Create spatial pattern - higher NDVI in certain areas (parks, green spaces)
    ndvi = np.random.uniform(0.1, 0.6, n_points)

    # Add some "park" areas with higher NDVI
    park_centers = [
        (33.57, 73.01), (33.56, 73.02), (33.55, 73.00)
    ]
    for center in park_centers:
        dist = np.sqrt((lats - center[0])**2 + (lons - center[1])**2)
        ndvi += 0.3 * np.exp(-dist / 0.005)

    ndvi = np.clip(ndvi, 0, 1)

    return pd.DataFrame({'lon': lons, 'lat': lats, 'ndvi': ndvi})


def generate_synthetic_lst():
    """Generate synthetic LST data for testing."""
    print("Generating synthetic LST data...")

    np.random.seed(43)
    n_points = 3000

    lats = np.random.uniform(BBOX['min_lat'], BBOX['max_lat'], n_points)
    lons = np.random.uniform(BBOX['min_lon'], BBOX['max_lon'], n_points)

    # Base temperature around 35°C for summer in Rawalpindi
    lst = np.random.uniform(32, 42, n_points)

    # Lower temperature in green areas
    park_centers = [
        (33.57, 73.01), (33.56, 73.02), (33.55, 73.00)
    ]
    for center in park_centers:
        dist = np.sqrt((lats - center[0])**2 + (lons - center[1])**2)
        lst -= 5 * np.exp(-dist / 0.005)

    lst = np.clip(lst, 25, 45)

    return pd.DataFrame({'lon': lons, 'lat': lats, 'lst': lst})


def generate_synthetic_slope():
    """Generate synthetic slope data for testing."""
    print("Generating synthetic slope data...")

    np.random.seed(44)
    n_points = 3000

    lats = np.random.uniform(BBOX['min_lat'], BBOX['max_lat'], n_points)
    lons = np.random.uniform(BBOX['min_lon'], BBOX['max_lon'], n_points)

    # Mostly flat with some slopes
    slope = np.random.exponential(2, n_points)
    slope = np.clip(slope, 0, 30)

    return pd.DataFrame({'lon': lons, 'lat': lats, 'slope': slope})


def collect_all_data(use_cache=True):
    """
    Collect all required data from GEE and OSM.
    Returns dictionary with all datasets.
    """
    print("\n" + "="*50)
    print("GreenPath Data Collection")
    print("="*50 + "\n")

    # Initialize GEE
    gee_available = initialize_gee()

    # Generate hexagon grid
    hex_grid = get_hexagon_grid()

    # Fetch satellite data
    if gee_available:
        ndvi_data = fetch_sentinel2_ndvi()
        lst_data = fetch_landsat_lst()
        slope_data = fetch_srtm_elevation()
    else:
        print("\nUsing synthetic data (GEE unavailable)...")
        ndvi_data = generate_synthetic_ndvi()
        lst_data = generate_synthetic_lst()
        slope_data = generate_synthetic_slope()

    # Fetch OSM data
    road_network = fetch_osm_roads()
    buildings = fetch_osm_buildings()

    print("\n" + "="*50)
    print("Data Collection Complete!")
    print("="*50 + "\n")

    return {
        'hex_grid': hex_grid,
        'ndvi': ndvi_data,
        'lst': lst_data,
        'slope': slope_data,
        'roads': road_network,
        'buildings': buildings
    }


def collect_all_data_for_city(city_config):
    """
    Collect all required data for a specific city.

    Args:
        city_config: Dictionary with city name, lat, lon, and bbox

    Returns:
        Dictionary with all datasets for the city
    """
    city_name = city_config['name']
    bbox = city_config['bbox']

    print("\n" + "="*50)
    print(f"GreenPath Data Collection - {city_name}")
    print("="*50 + "\n")

    # Initialize GEE
    gee_available = initialize_gee()

    # Generate hexagon grid for this city
    hex_grid = get_hexagon_grid_for_bbox(bbox)

    # Fetch satellite data
    if gee_available:
        ndvi_data = fetch_sentinel2_ndvi_for_bbox(bbox)
        lst_data = fetch_landsat_lst_for_bbox(bbox)
        slope_data = fetch_srtm_elevation_for_bbox(bbox)
    else:
        print("\nUsing synthetic data (GEE unavailable)...")
        ndvi_data = generate_synthetic_ndvi_for_bbox(bbox)
        lst_data = generate_synthetic_lst_for_bbox(bbox)
        slope_data = generate_synthetic_slope_for_bbox(bbox)

    # Fetch OSM data
    road_network = fetch_osm_roads_for_bbox(bbox)
    buildings = fetch_osm_buildings_for_bbox(bbox)

    print("\n" + "="*50)
    print(f"Data Collection Complete for {city_name}!")
    print("="*50 + "\n")

    return {
        'hex_grid': hex_grid,
        'ndvi': ndvi_data,
        'lst': lst_data,
        'slope': slope_data,
        'roads': road_network,
        'buildings': buildings
    }


def get_hexagon_grid_for_bbox(bbox):
    """Generate H3 hexagon grid for a specific bounding box."""
    print("Generating H3 hexagon grid...")

    lats = np.linspace(bbox['min_lat'], bbox['max_lat'], 100)
    lons = np.linspace(bbox['min_lon'], bbox['max_lon'], 100)

    hex_ids = set()
    for lat in lats:
        for lon in lons:
            hex_id = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
            hex_ids.add(hex_id)

    hex_data = []
    for hex_id in hex_ids:
        boundary = h3.cell_to_boundary(hex_id)
        coords = [(lon, lat) for lat, lon in boundary]
        polygon = Polygon(coords)
        center = h3.cell_to_latlng(hex_id)

        hex_data.append({
            'hex_id': hex_id,
            'geometry': polygon,
            'center_lat': center[0],
            'center_lon': center[1]
        })

    gdf = gpd.GeoDataFrame(hex_data, crs='EPSG:4326')
    print(f"[OK] Generated {len(gdf)} hexagons")
    return gdf


def fetch_osm_roads_for_bbox(bbox):
    """Fetch OSM road network for a specific bounding box."""
    print("Fetching OSM road network...")

    import time

    bbox_tuple = (bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat'])
    
    # Retry up to 3 times with increasing timeout
    for attempt in range(3):
        try:
            # Configure OSMnx to use longer timeout
            ox.settings.timeout = 180  # 3 minutes
            ox.settings.memory = None

            G = ox.graph_from_bbox(bbox=bbox_tuple, network_type='walk', simplify=True)
            G = ox.project_graph(G)
            print(f"[OK] Fetched road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            print(f"[WARN] Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print(f"  Retrying in {(attempt + 1) * 5} seconds...")
                time.sleep((attempt + 1) * 5)
            else:
                print(f"[FAIL] Could not fetch road network after 3 attempts")
                raise


def fetch_osm_buildings_for_bbox(bbox):
    """Fetch OSM building footprints for a specific bounding box."""
    print("Fetching OSM building footprints...")

    import time

    bbox_tuple = (bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat'])
    tags = {'building': True}

    # Retry up to 3 times with increasing timeout
    for attempt in range(3):
        try:
            # Configure OSMnx to use longer timeout
            ox.settings.timeout = 180  # 3 minutes
            ox.settings.memory = None

            buildings = ox.features_from_bbox(bbox=bbox_tuple, tags=tags)
            buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]

            if 'height' in buildings.columns:
                buildings['height_m'] = pd.to_numeric(
                    buildings['height'].str.replace(' m', ''), errors='coerce')
            else:
                buildings['height_m'] = np.nan

            if 'building:levels' in buildings.columns:
                levels = pd.to_numeric(buildings['building:levels'], errors='coerce')
                buildings['height_m'] = buildings['height_m'].fillna(levels * 3)

            buildings['height_m'] = buildings['height_m'].fillna(6)
            buildings = buildings[['geometry', 'height_m']].reset_index(drop=True)

            print(f"[OK] Fetched {len(buildings)} buildings")
            return buildings

        except Exception as e:
            print(f"[WARN] Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print(f"  Retrying in {(attempt + 1) * 5} seconds...")
                time.sleep((attempt + 1) * 5)
            else:
                print(f"[FAIL] Could not fetch buildings after 3 attempts")
                # Return empty GeoDataFrame - shadow score will be 0
                return gpd.GeoDataFrame(columns=['geometry', 'height_m'], crs='EPSG:4326')


def generate_synthetic_ndvi_for_bbox(bbox):
    """Generate synthetic NDVI data for a specific bounding box."""
    print("Generating synthetic NDVI data...")
    # Use bbox coordinates to create unique seed for each city
    seed = int(abs(bbox['min_lat'] * 1000 + bbox['min_lon'] * 100)) % 10000
    np.random.seed(seed)
    n_points = 5000

    lats = np.random.uniform(bbox['min_lat'], bbox['max_lat'], n_points)
    lons = np.random.uniform(bbox['min_lon'], bbox['max_lon'], n_points)
    ndvi = np.random.uniform(0.1, 0.6, n_points)

    # Add park areas
    center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2
    center_lon = (bbox['min_lon'] + bbox['max_lon']) / 2
    park_centers = [
        (center_lat + 0.008, center_lon - 0.008),
        (center_lat - 0.008, center_lon + 0.008),
        (center_lat + 0.003, center_lon + 0.003),
        (center_lat - 0.005, center_lon - 0.005),
        (center_lat, center_lon)
    ]
    for center in park_centers:
        dist = np.sqrt((lats - center[0])**2 + (lons - center[1])**2)
        ndvi += 0.3 * np.exp(-dist / 0.006)

    ndvi = np.clip(ndvi, 0, 1)
    return pd.DataFrame({'lon': lons, 'lat': lats, 'ndvi': ndvi})


def generate_synthetic_lst_for_bbox(bbox):
    """Generate synthetic LST data for a specific bounding box."""
    print("Generating synthetic LST data...")
    # Use bbox coordinates to create unique seed for each city
    seed = int(abs(bbox['min_lat'] * 1000 + bbox['min_lon'] * 100 + 1)) % 10000
    np.random.seed(seed)
    n_points = 3000

    lats = np.random.uniform(bbox['min_lat'], bbox['max_lat'], n_points)
    lons = np.random.uniform(bbox['min_lon'], bbox['max_lon'], n_points)
    lst = np.random.uniform(32, 42, n_points)

    center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2
    center_lon = (bbox['min_lon'] + bbox['max_lon']) / 2
    park_centers = [
        (center_lat + 0.008, center_lon - 0.008),
        (center_lat - 0.008, center_lon + 0.008),
        (center_lat + 0.003, center_lon + 0.003),
        (center_lat - 0.005, center_lon - 0.005),
        (center_lat, center_lon)
    ]
    for center in park_centers:
        dist = np.sqrt((lats - center[0])**2 + (lons - center[1])**2)
        lst -= 5 * np.exp(-dist / 0.006)

    lst = np.clip(lst, 25, 45)
    return pd.DataFrame({'lon': lons, 'lat': lats, 'lst': lst})


def generate_synthetic_slope_for_bbox(bbox):
    """Generate synthetic slope data for a specific bounding box."""
    print("Generating synthetic slope data...")
    # Use bbox coordinates to create unique seed for each city
    seed = int(abs(bbox['min_lat'] * 1000 + bbox['min_lon'] * 100 + 2)) % 10000
    np.random.seed(seed)
    n_points = 3000

    lats = np.random.uniform(bbox['min_lat'], bbox['max_lat'], n_points)
    lons = np.random.uniform(bbox['min_lon'], bbox['max_lon'], n_points)
    slope = np.random.exponential(2, n_points)
    slope = np.clip(slope, 0, 30)

    return pd.DataFrame({'lon': lons, 'lat': lats, 'slope': slope})


def fetch_sentinel2_ndvi_for_bbox(bbox):
    """Fetch Sentinel-2 NDVI for a specific bounding box."""
    print("Fetching Sentinel-2 NDVI data from GEE...")

    try:
        study_area = ee.Geometry.Rectangle([
            bbox['min_lon'], bbox['min_lat'],
            bbox['max_lon'], bbox['max_lat']
        ])

        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(study_area) \
            .filterDate(DATE_START, DATE_END) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER_MAX))

        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)

        s2_ndvi = s2.map(add_ndvi)
        ndvi_median = s2_ndvi.select('NDVI').median()
        ndvi_image = ndvi_median.clip(study_area)

        sample_points = ndvi_image.sample(
            region=study_area, scale=10, numPixels=5000, geometries=True
        )

        features = sample_points.getInfo()['features']
        ndvi_points = [{'lon': f['geometry']['coordinates'][0],
                        'lat': f['geometry']['coordinates'][1],
                        'ndvi': f['properties'].get('NDVI', 0)} for f in features]

        print(f"[OK] Fetched {len(ndvi_points)} NDVI samples")
        return pd.DataFrame(ndvi_points)

    except Exception as e:
        print(f"[FAIL] Error fetching NDVI: {e}")
        return generate_synthetic_ndvi_for_bbox(bbox)


def fetch_landsat_lst_for_bbox(bbox):
    """Fetch Landsat LST for a specific bounding box."""
    print("Fetching Landsat LST data from GEE...")

    try:
        study_area = ee.Geometry.Rectangle([
            bbox['min_lon'], bbox['min_lat'],
            bbox['max_lon'], bbox['max_lat']
        ])

        landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(study_area) \
            .filterDate(DATE_START, DATE_END) \
            .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))

        def calc_lst(image):
            thermal = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
            return thermal.rename('LST')

        lst_collection = landsat.map(calc_lst)
        lst_median = lst_collection.median()
        lst_image = lst_median.clip(study_area)

        sample_points = lst_image.sample(
            region=study_area, scale=30, numPixels=3000, geometries=True
        )

        features = sample_points.getInfo()['features']
        lst_points = [{'lon': f['geometry']['coordinates'][0],
                       'lat': f['geometry']['coordinates'][1],
                       'lst': f['properties'].get('LST', 35)} for f in features]

        print(f"[OK] Fetched {len(lst_points)} LST samples")
        return pd.DataFrame(lst_points)

    except Exception as e:
        print(f"[FAIL] Error fetching LST: {e}")
        return generate_synthetic_lst_for_bbox(bbox)


def fetch_srtm_elevation_for_bbox(bbox):
    """Fetch SRTM elevation/slope for a specific bounding box."""
    print("Fetching SRTM elevation data from GEE...")

    try:
        study_area = ee.Geometry.Rectangle([
            bbox['min_lon'], bbox['min_lat'],
            bbox['max_lon'], bbox['max_lat']
        ])

        srtm = ee.Image('USGS/SRTMGL1_003')
        slope = ee.Terrain.slope(srtm.select('elevation'))
        slope_image = slope.clip(study_area)

        sample_points = slope_image.sample(
            region=study_area, scale=30, numPixels=3000, geometries=True
        )

        features = sample_points.getInfo()['features']
        slope_points = [{'lon': f['geometry']['coordinates'][0],
                         'lat': f['geometry']['coordinates'][1],
                         'slope': f['properties'].get('slope', 0)} for f in features]

        print(f"[OK] Fetched {len(slope_points)} slope samples")
        return pd.DataFrame(slope_points)

    except Exception as e:
        print(f"[FAIL] Error fetching elevation: {e}")
        return generate_synthetic_slope_for_bbox(bbox)


if __name__ == '__main__':
    # Test data collection
    data = collect_all_data()
    print("\nData summary:")
    print(f"  Hexagons: {len(data['hex_grid'])}")
    print(f"  NDVI samples: {len(data['ndvi'])}")
    print(f"  LST samples: {len(data['lst'])}")
    print(f"  Slope samples: {len(data['slope'])}")
    print(f"  Road nodes: {data['roads'].number_of_nodes()}")
    print(f"  Buildings: {len(data['buildings'])}")
