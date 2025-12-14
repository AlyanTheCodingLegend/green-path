"""
Configuration settings for GreenPath application.
"""

# Multi-city configurations
CITIES = {
    'Rawalpindi': {
        'lat': 33.5651,
        'lon': 73.0169,
        'bbox_buffer': 0.025
    },
    'Islamabad': {
        'lat': 33.6844,
        'lon': 73.0479,
        'bbox_buffer': 0.025
    },
    'Lahore': {
        'lat': 31.5204,
        'lon': 74.3587,
        'bbox_buffer': 0.025
    },
    'Karachi': {
        'lat': 24.8138,
        'lon': 67.0297,
        'bbox_buffer': 0.025
    }
}

# Default city (for backward compatibility)
CITY_NAME = "Rawalpindi"
CITY_CENTER_LAT = CITIES[CITY_NAME]['lat']
CITY_CENTER_LON = CITIES[CITY_NAME]['lon']

# Bounding box for the study area (5km x 5km around center)
BBOX_BUFFER = 0.025  # ~2.5km in each direction
BBOX = {
    'min_lat': CITY_CENTER_LAT - BBOX_BUFFER,
    'max_lat': CITY_CENTER_LAT + BBOX_BUFFER,
    'min_lon': CITY_CENTER_LON - BBOX_BUFFER,
    'max_lon': CITY_CENTER_LON + BBOX_BUFFER
}

def get_city_config(city_name):
    """Get configuration for a specific city."""
    city = CITIES[city_name]
    buffer = city['bbox_buffer']
    return {
        'name': city_name,
        'lat': city['lat'],
        'lon': city['lon'],
        'bbox': {
            'min_lat': city['lat'] - buffer,
            'max_lat': city['lat'] + buffer,
            'min_lon': city['lon'] - buffer,
            'max_lon': city['lon'] + buffer
        }
    }

# H3 hexagon resolution (9 = ~170m diameter)
H3_RESOLUTION = 9

# Date range for satellite imagery (Summer 2024)
DATE_START = '2024-06-01'
DATE_END = '2024-08-31'

# Thermal comfort score weights
WEIGHTS = {
    'ndvi': 0.30,      # Vegetation coverage
    'lst': 0.40,       # Land surface temperature (inverted)
    'slope': 0.20,     # Terrain flatness
    'shadow': 0.10     # Building shadow potential
}

# Satellite data parameters
CLOUD_COVER_MAX = 20  # Maximum cloud cover percentage

# Routing parameters
COMFORT_WEIGHT = 4.0  # Weight for comfort vs distance in cool route
DISTANCE_WEIGHT = 0.5

# Cache settings
import os as _os
CACHE_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'cache')
CACHE_EXPIRY_DAYS = 7

# Map display settings
MAP_ZOOM_START = 14
ROUTE_COOL_COLOR = '#2E7D32'  # Green
ROUTE_FAST_COLOR = '#D32F2F'  # Red
HEATMAP_COLORMAP = 'RdYlGn'   # Red-Yellow-Green

# GEE settings
GEE_PROJECT = 'ml-proj-481211'

# Export settings
EXPORT_FORMAT = 'gpx'
