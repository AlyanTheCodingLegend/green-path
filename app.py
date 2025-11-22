"""
Streamlit GUI application for GreenPath.
Interactive map interface for shade-optimized route finding.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import pickle
from datetime import datetime
import pyproj
from shapely.ops import transform
import branca.colormap as cm

from config import (
    CITY_NAME, CITY_CENTER_LAT, CITY_CENTER_LON, BBOX,
    MAP_ZOOM_START, ROUTE_COOL_COLOR, ROUTE_FAST_COLOR,
    CACHE_DIR, H3_RESOLUTION, WEIGHTS, CITIES, get_city_config
)

# Set page config
st.set_page_config(
    page_title="GreenPath - Shade Navigator",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_city_data(city_name):
    """Load and cache data for a specific city with lazy loading."""
    from data_collection import collect_all_data_for_city
    from preprocessing import preprocess_data
    from scoring import calculate_comfort_scores
    from routing import assign_comfort_to_edges

    city_config = get_city_config(city_name)
    city_cache_dir = os.path.join(CACHE_DIR, city_name.lower())

    # Check for cached complete data
    complete_cache = os.path.join(city_cache_dir, 'complete_data.pkl')

    if os.path.exists(complete_cache):
        data = pickle.load(open(complete_cache, 'rb'))
        return data['hex_grid'], data['G'], data['raw_data']

    # Collect raw data for this city
    raw_data = collect_all_data_for_city(city_config)

    # Preprocess
    hex_grid = preprocess_data(raw_data)

    # Calculate scores using ML model if available, otherwise weighted
    hex_grid = calculate_comfort_scores(hex_grid, method='ml')

    # Assign scores to road network
    G = assign_comfort_to_edges(raw_data['roads'], hex_grid)

    # Cache complete data
    os.makedirs(city_cache_dir, exist_ok=True)
    pickle.dump({
        'hex_grid': hex_grid,
        'G': G,
        'raw_data': raw_data
    }, open(complete_cache, 'wb'))

    return hex_grid, G, raw_data


# Keep old function for backward compatibility
@st.cache_resource
def load_data():
    """Load and cache all processed data (backward compatibility)."""
    return load_city_data(CITY_NAME)


def create_base_map(center_lat, center_lon, zoom=14):
    """Create a base Folium map."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='CartoDB positron'
    )
    return m


def add_comfort_heatmap(m, hex_grid):
    """Add thermal comfort heatmap overlay to map."""
    # Create colormap
    colormap = cm.LinearColormap(
        colors=['red', 'orange', 'yellow', 'lightgreen', 'green'],
        vmin=0, vmax=1,
        caption='Thermal Comfort Score'
    )

    # Add hexagons as GeoJson
    for idx, row in hex_grid.iterrows():
        comfort = row['comfort_score']

        # Handle NaN or invalid values
        if pd.isna(comfort) or not np.isfinite(comfort):
            comfort = 0.5
        comfort = float(np.clip(comfort, 0, 1))

        color = colormap(comfort)

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': 'gray',
                'weight': 0.5,
                'fillOpacity': 0.5
            },
            tooltip=f"Comfort: {comfort:.2f}<br>NDVI: {row['ndvi']:.2f}<br>LST: {row['lst']:.1f}°C"
        ).add_to(m)

    colormap.add_to(m)
    return m


def add_route_to_map(m, route_stats, color, name, graph_crs):
    """Add a route to the map."""
    if route_stats is None or route_stats['geometry'] is None:
        return m

    # Convert to WGS84
    transformer = pyproj.Transformer.from_crs(
        graph_crs, 'EPSG:4326', always_xy=True
    )

    def transform_coords(x, y):
        return transformer.transform(x, y)

    geom_wgs84 = transform(transform_coords, route_stats['geometry'])

    # Get coordinates
    if geom_wgs84.geom_type == 'MultiLineString':
        for line in geom_wgs84.geoms:
            coords = [(lat, lon) for lon, lat in line.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                opacity=0.8,
                popup=f"{name}<br>Distance: {route_stats['distance_km']:.2f} km<br>Comfort: {route_stats['avg_comfort']:.0%}"
            ).add_to(m)
    else:
        coords = [(lat, lon) for lon, lat in geom_wgs84.coords]
        folium.PolyLine(
            coords,
            color=color,
            weight=5,
            opacity=0.8,
            popup=f"{name}<br>Distance: {route_stats['distance_km']:.2f} km<br>Comfort: {route_stats['avg_comfort']:.0%}"
        ).add_to(m)

    return m


def add_markers(m, start_coords, end_coords):
    """Add start and end markers to map."""
    if start_coords:
        folium.Marker(
            start_coords,
            popup="Start",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

    if end_coords:
        folium.Marker(
            end_coords,
            popup="End",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

    return m


def geocode_address(address):
    """
    Simple geocoding using Nominatim.
    Returns (lat, lon) or None if not found.
    """
    try:
        import requests
        url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
        headers = {'User-Agent': 'GreenPath/1.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            results = response.json()
            if results:
                return float(results[0]['lat']), float(results[0]['lon'])
    except Exception:
        pass
    return None


def main():
    """Main Streamlit application."""

    # Header
    st.title("GreenPath: The Shade & Walkability Navigator")
    st.markdown("*Find thermally comfortable walking routes in Pakistani cities*")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Display options
        show_heatmap = st.checkbox("Show comfort heatmap", value=True)
        show_both_routes = st.checkbox("Show both routes", value=True)

        st.markdown("---")

        # Data info
        st.subheader("Info")
        st.info(f"H3 resolution: {H3_RESOLUTION}\nWeights: NDVI {WEIGHTS['ndvi']*100:.0f}%, LST {WEIGHTS['lst']*100:.0f}%")

        # Clear cache
        if st.button("Clear Cache"):
            if os.path.exists(CACHE_DIR):
                import shutil
                shutil.rmtree(CACHE_DIR)
            st.cache_resource.clear()
            st.rerun()

    # Create tabs for each city
    tabs = st.tabs(list(CITIES.keys()))

    for i, city_name in enumerate(CITIES.keys()):
        with tabs[i]:
            # Lazy loading - only load data when tab is active
            if st.session_state.get(f'{city_name}_loaded') or i == 0:
                st.session_state[f'{city_name}_loaded'] = True

                try:
                    with st.spinner(f"Loading {city_name} data..."):
                        hex_grid, G, raw_data = load_city_data(city_name)
                        graph_crs = G.graph.get('crs', 'EPSG:32643')
                except Exception as e:
                    st.error(f"Error loading {city_name}: {e}")
                    continue

                city_config = get_city_config(city_name)
                city_lat = city_config['lat']
                city_lon = city_config['lon']

                # City-specific session state keys
                start_key = f'{city_name}_start'
                end_key = f'{city_name}_end'
                mode_key = f'{city_name}_mode'
                result_key = f'{city_name}_result'

                if start_key not in st.session_state:
                    st.session_state[start_key] = None
                if end_key not in st.session_state:
                    st.session_state[end_key] = None
                if mode_key not in st.session_state:
                    st.session_state[mode_key] = 'start'
                if result_key not in st.session_state:
                    st.session_state[result_key] = None

                # Calculate routes if requested
                if st.session_state.get(f'{city_name}_find'):
                    from routing import compare_routes
                    with st.spinner("Calculating routes..."):
                        st.session_state[result_key] = compare_routes(
                            G,
                            st.session_state[start_key][0],
                            st.session_state[start_key][1],
                            st.session_state[end_key][0],
                            st.session_state[end_key][1],
                            hex_grid
                        )
                    st.session_state[f'{city_name}_find'] = False

                # Show route results at top
                if st.session_state[result_key] and st.session_state[result_key].get('fast_route') and st.session_state[result_key].get('cool_route'):
                    result = st.session_state[result_key]
                    st.subheader("Route Results")

                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col1:
                        st.markdown("**Cool Route**")
                        cool = result['cool_route']
                        st.metric("Distance", f"{cool['distance_km']:.2f} km")
                        st.metric("Time", f"{cool['walking_time_min']:.0f} min")
                        st.metric("Comfort", f"{cool['avg_comfort']:.0%}")

                    with col2:
                        route_map = create_base_map(
                            (st.session_state[start_key][0] + st.session_state[end_key][0]) / 2,
                            (st.session_state[start_key][1] + st.session_state[end_key][1]) / 2,
                            MAP_ZOOM_START + 1
                        )
                        if show_heatmap:
                            route_map = add_comfort_heatmap(route_map, hex_grid)
                        if show_both_routes:
                            route_map = add_route_to_map(route_map, result['fast_route'], ROUTE_FAST_COLOR, "Fast", graph_crs)
                        route_map = add_route_to_map(route_map, result['cool_route'], ROUTE_COOL_COLOR, "Cool", graph_crs)
                        route_map = add_markers(route_map, st.session_state[start_key], st.session_state[end_key])
                        folium_static(route_map, width=450, height=350)

                    with col3:
                        st.markdown("**Fast Route**")
                        fast = result['fast_route']
                        st.metric("Distance", f"{fast['distance_km']:.2f} km")
                        st.metric("Time", f"{fast['walking_time_min']:.0f} min")
                        st.metric("Comfort", f"{fast['avg_comfort']:.0%}")

                    st.markdown("---")

                # Selection interface
                st.subheader("Select Points")

                btn1, btn2, status = st.columns([1, 1, 3])
                with btn1:
                    if st.button("Start", key=f"{city_name}_s", type="primary" if st.session_state[mode_key] == 'start' else "secondary"):
                        st.session_state[mode_key] = 'start'
                with btn2:
                    if st.button("End", key=f"{city_name}_e", type="primary" if st.session_state[mode_key] == 'end' else "secondary"):
                        st.session_state[mode_key] = 'end'
                with status:
                    s = f"Start: {st.session_state[start_key][0]:.3f},{st.session_state[start_key][1]:.3f}" if st.session_state[start_key] else "Start: -"
                    e = f"End: {st.session_state[end_key][0]:.3f},{st.session_state[end_key][1]:.3f}" if st.session_state[end_key] else "End: -"
                    st.caption(f"{s} | {e}")

                # Map
                m = create_base_map(city_lat, city_lon, MAP_ZOOM_START)
                if show_heatmap:
                    m = add_comfort_heatmap(m, hex_grid)
                m = add_markers(m, st.session_state[start_key], st.session_state[end_key])

                map_data = st_folium(m, height=400, key=f"{city_name}_map", returned_objects=["last_clicked"])

                if map_data and map_data.get('last_clicked'):
                    clicked = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
                    if st.session_state[mode_key] == 'start' and st.session_state[start_key] != clicked:
                        st.session_state[start_key] = clicked
                        st.rerun()
                    elif st.session_state[mode_key] == 'end' and st.session_state[end_key] != clicked:
                        st.session_state[end_key] = clicked
                        st.rerun()

                # Find button
                _, btn_col, _ = st.columns([1, 2, 1])
                with btn_col:
                    if st.button("Find Routes", key=f"{city_name}_f", type="primary", use_container_width=True):
                        if st.session_state[start_key] and st.session_state[end_key]:
                            st.session_state[f'{city_name}_find'] = True
                            st.rerun()
                        else:
                            st.warning("Set both points first")
            else:
                # Show placeholder until tab is clicked
                st.info(f"Click to load {city_name} data")
                if st.button(f"Load {city_name}", key=f"load_{city_name}"):
                    st.session_state[f'{city_name}_loaded'] = True
                    st.rerun()


if __name__ == '__main__':
    main()
