"""
Routing module for GreenPath.
Implements pathfinding algorithms with thermal comfort optimization.
"""

import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj
import heapq

from config import COMFORT_WEIGHT, DISTANCE_WEIGHT
from scoring import get_discomfort_cost


def assign_comfort_to_edges(G, hex_grid):
    """
    Assign thermal comfort scores to each edge in the road network.
    Score is based on hexagons that the edge passes through.

    Args:
        G: NetworkX graph (projected)
        hex_grid: GeoDataFrame with comfort scores (WGS84)

    Returns:
        Graph with comfort scores added to edges
    """
    print("Assigning comfort scores to road segments...")

    # Get the CRS of the graph
    graph_crs = G.graph.get('crs', 'EPSG:32643')  # Default to UTM 43N

    # Project hex_grid to match graph CRS
    hex_projected = hex_grid.to_crs(graph_crs)

    # Build spatial index for hexagons
    hex_sindex = hex_projected.sindex

    # Iterate through edges
    edges_processed = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        # Get edge geometry
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            # Create geometry from nodes
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            edge_geom = LineString([
                (u_data['x'], u_data['y']),
                (v_data['x'], v_data['y'])
            ])

        # Find hexagons that intersect with this edge
        possible_matches_idx = list(hex_sindex.intersection(edge_geom.bounds))

        if not possible_matches_idx:
            # No matching hexagons, use default medium comfort
            data['comfort_score'] = 0.5
            data['discomfort_cost'] = get_discomfort_cost(0.5)
        else:
            possible_matches = hex_projected.iloc[possible_matches_idx]
            intersecting = possible_matches[possible_matches.intersects(edge_geom)]

            if len(intersecting) == 0:
                data['comfort_score'] = 0.5
                data['discomfort_cost'] = get_discomfort_cost(0.5)
            else:
                # Weight by intersection length
                total_length = 0
                weighted_comfort = 0

                for idx, hex_row in intersecting.iterrows():
                    intersection = edge_geom.intersection(hex_row.geometry)
                    if intersection.is_empty:
                        continue

                    length = intersection.length
                    total_length += length
                    weighted_comfort += length * hex_row['comfort_score']

                if total_length > 0:
                    avg_comfort = weighted_comfort / total_length
                else:
                    avg_comfort = intersecting['comfort_score'].mean()

                data['comfort_score'] = avg_comfort
                data['discomfort_cost'] = get_discomfort_cost(avg_comfort)

        edges_processed += 1

    print(f"[OK] Processed {edges_processed} edges")
    return G


def find_shortest_route(G, start_node, end_node):
    """
    Find the shortest route by distance.

    Args:
        G: NetworkX graph with edge weights
        start_node: Origin node ID
        end_node: Destination node ID

    Returns:
        List of node IDs forming the path, or None if no path exists
    """
    try:
        path = nx.shortest_path(G, start_node, end_node, weight='length')
        return path
    except nx.NetworkXNoPath:
        return None


def find_coolest_route(G, start_node, end_node, comfort_weight=None, distance_weight=None):
    """
    Find the coolest/most comfortable route.
    Uses combined cost of distance and thermal discomfort.

    Args:
        G: NetworkX graph with comfort scores
        start_node: Origin node ID
        end_node: Destination node ID
        comfort_weight: Weight for comfort in cost function
        distance_weight: Weight for distance in cost function

    Returns:
        List of node IDs forming the path, or None if no path exists
    """
    if comfort_weight is None:
        comfort_weight = COMFORT_WEIGHT
    if distance_weight is None:
        distance_weight = DISTANCE_WEIGHT

    # Calculate combined cost for each edge
    def combined_cost(u, v, data):
        length = data.get('length', 1)
        discomfort = data.get('discomfort_cost', 0.5)

        # Normalize length (assume typical edge is 50m)
        norm_length = length / 50

        # Combined cost
        cost = (distance_weight * norm_length +
                comfort_weight * discomfort * norm_length)

        return cost

    try:
        path = nx.shortest_path(G, start_node, end_node, weight=combined_cost)
        return path
    except nx.NetworkXNoPath:
        return None


def get_route_stats(G, path, hex_grid=None):
    """
    Calculate statistics for a route.

    Args:
        G: NetworkX graph
        path: List of node IDs
        hex_grid: GeoDataFrame with comfort data (optional)

    Returns:
        Dictionary with route statistics
    """
    if path is None or len(path) < 2:
        return None

    total_distance = 0
    total_comfort = 0
    comfort_segments = 0
    edge_geometries = []

    # Get CRS for coordinate conversion
    graph_crs = G.graph.get('crs', 'EPSG:32643')

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Get edge data (there might be multiple edges between nodes)
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue

        # Get the first edge if multiple exist
        if isinstance(edge_data, dict) and 0 in edge_data:
            data = edge_data[0]
        else:
            data = list(edge_data.values())[0] if isinstance(edge_data, dict) else edge_data

        # Accumulate distance
        length = data.get('length', 0)
        total_distance += length

        # Accumulate comfort
        comfort = data.get('comfort_score', 0.5)
        total_comfort += comfort * length
        comfort_segments += length

        # Collect geometry
        if 'geometry' in data:
            edge_geometries.append(data['geometry'])
        else:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            edge_geometries.append(LineString([
                (u_data['x'], u_data['y']),
                (v_data['x'], v_data['y'])
            ]))

    # Calculate average comfort
    avg_comfort = total_comfort / comfort_segments if comfort_segments > 0 else 0.5

    # Estimate walking time (assuming 5 km/h walking speed)
    walking_time = (total_distance / 1000) / 5 * 60  # minutes

    # Estimate tree coverage from comfort score (rough approximation)
    tree_coverage = avg_comfort * 50  # percentage

    # Create route geometry
    if edge_geometries:
        from shapely.ops import linemerge
        try:
            route_geometry = linemerge(edge_geometries)
        except Exception:
            route_geometry = edge_geometries[0]
    else:
        route_geometry = None

    return {
        'distance_m': total_distance,
        'distance_km': total_distance / 1000,
        'avg_comfort': avg_comfort,
        'walking_time_min': walking_time,
        'tree_coverage_pct': tree_coverage,
        'geometry': route_geometry,
        'path': path,
        'crs': graph_crs
    }


def get_nearest_node(G, lat, lon):
    """
    Find the nearest node in the graph to a given lat/lon.

    Args:
        G: NetworkX graph (projected)
        lat: Latitude
        lon: Longitude

    Returns:
        Node ID
    """
    # Convert lat/lon to projected coordinates if graph is projected
    graph_crs = G.graph.get('crs', None)

    if graph_crs and str(graph_crs) != 'EPSG:4326':
        # Graph is projected, need to transform coordinates
        transformer = pyproj.Transformer.from_crs(
            'EPSG:4326', graph_crs, always_xy=True
        )
        x, y = transformer.transform(lon, lat)
        return ox.nearest_nodes(G, x, y)
    else:
        return ox.nearest_nodes(G, lon, lat)


def route_to_geojson(route_stats):
    """
    Convert route statistics to GeoJSON format for mapping.

    Args:
        route_stats: Dictionary from get_route_stats

    Returns:
        GeoJSON dictionary
    """
    if route_stats is None or route_stats['geometry'] is None:
        return None

    # Convert to WGS84
    graph_crs = route_stats.get('crs', 'EPSG:32643')

    transformer = pyproj.Transformer.from_crs(
        graph_crs, 'EPSG:4326', always_xy=True
    )

    def transform_coords(x, y):
        return transformer.transform(x, y)

    geom_wgs84 = transform(transform_coords, route_stats['geometry'])

    # Create GeoJSON
    if geom_wgs84.geom_type == 'MultiLineString':
        coordinates = [list(line.coords) for line in geom_wgs84.geoms]
    else:
        coordinates = list(geom_wgs84.coords)

    return {
        'type': 'Feature',
        'geometry': {
            'type': geom_wgs84.geom_type,
            'coordinates': coordinates
        },
        'properties': {
            'distance_km': round(route_stats['distance_km'], 2),
            'avg_comfort': round(route_stats['avg_comfort'], 3),
            'walking_time_min': round(route_stats['walking_time_min'], 1),
            'tree_coverage_pct': round(route_stats['tree_coverage_pct'], 1)
        }
    }


def export_route_gpx(route_stats, filename):
    """
    Export route to GPX file format.

    Args:
        route_stats: Dictionary from get_route_stats
        filename: Output filename

    Returns:
        Path to saved file
    """
    if route_stats is None or route_stats['geometry'] is None:
        return None

    # Convert to WGS84
    graph_crs = route_stats.get('crs', 'EPSG:32643')

    transformer = pyproj.Transformer.from_crs(
        graph_crs, 'EPSG:4326', always_xy=True
    )

    def transform_coords(x, y):
        return transformer.transform(x, y)

    geom_wgs84 = transform(transform_coords, route_stats['geometry'])

    # Get coordinates
    if geom_wgs84.geom_type == 'MultiLineString':
        coords = []
        for line in geom_wgs84.geoms:
            coords.extend(list(line.coords))
    else:
        coords = list(geom_wgs84.coords)

    # Create GPX
    gpx_content = '''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="GreenPath">
  <metadata>
    <name>GreenPath Route</name>
    <desc>Distance: {:.2f}km, Comfort: {:.0%}</desc>
  </metadata>
  <trk>
    <name>GreenPath Route</name>
    <trkseg>
'''.format(route_stats['distance_km'], route_stats['avg_comfort'])

    for lon, lat in coords:
        gpx_content += f'      <trkpt lat="{lat}" lon="{lon}"></trkpt>\n'

    gpx_content += '''    </trkseg>
  </trk>
</gpx>'''

    with open(filename, 'w') as f:
        f.write(gpx_content)

    return filename


def compare_routes(G, start_lat, start_lon, end_lat, end_lon, hex_grid=None):
    """
    Compare cool route vs fast route between two points.

    Args:
        G: NetworkX graph with comfort scores
        start_lat, start_lon: Start coordinates
        end_lat, end_lon: End coordinates
        hex_grid: GeoDataFrame with comfort data

    Returns:
        Dictionary with both routes and comparison metrics
    """
    # Find nearest nodes
    start_node = get_nearest_node(G, start_lat, start_lon)
    end_node = get_nearest_node(G, end_lat, end_lon)

    # Find both routes
    fast_path = find_shortest_route(G, start_node, end_node)
    cool_path = find_coolest_route(G, start_node, end_node)

    # Get statistics
    fast_stats = get_route_stats(G, fast_path, hex_grid)
    cool_stats = get_route_stats(G, cool_path, hex_grid)

    # Calculate comparison
    comparison = {}
    if fast_stats and cool_stats:
        comparison['distance_diff_m'] = cool_stats['distance_m'] - fast_stats['distance_m']
        comparison['distance_diff_pct'] = (
            (cool_stats['distance_m'] - fast_stats['distance_m']) /
            fast_stats['distance_m'] * 100
        )
        comparison['comfort_improvement'] = cool_stats['avg_comfort'] - fast_stats['avg_comfort']
        comparison['time_diff_min'] = cool_stats['walking_time_min'] - fast_stats['walking_time_min']

    return {
        'fast_route': fast_stats,
        'cool_route': cool_stats,
        'comparison': comparison,
        'start_node': start_node,
        'end_node': end_node
    }


if __name__ == '__main__':
    # Test routing
    from data_collection import collect_all_data
    from preprocessing import preprocess_data
    from scoring import calculate_comfort_scores

    # Collect and process data
    data = collect_all_data()
    hex_grid = preprocess_data(data)
    hex_grid = calculate_comfort_scores(hex_grid)

    # Get road network and assign comfort scores
    G = data['roads']
    G = assign_comfort_to_edges(G, hex_grid)

    # Test route between two points
    start = (33.5651, 73.0169)  # City center
    end = (33.5751, 73.0269)    # ~1km away

    result = compare_routes(G, start[0], start[1], end[0], end[1], hex_grid)

    print("\nRoute comparison:")
    if result['fast_route']:
        print(f"\nFast Route:")
        print(f"  Distance: {result['fast_route']['distance_km']:.2f} km")
        print(f"  Time: {result['fast_route']['walking_time_min']:.1f} min")
        print(f"  Comfort: {result['fast_route']['avg_comfort']:.2%}")

    if result['cool_route']:
        print(f"\nCool Route:")
        print(f"  Distance: {result['cool_route']['distance_km']:.2f} km")
        print(f"  Time: {result['cool_route']['walking_time_min']:.1f} min")
        print(f"  Comfort: {result['cool_route']['avg_comfort']:.2%}")

    if result['comparison']:
        print(f"\nComparison:")
        print(f"  Extra distance: {result['comparison']['distance_diff_pct']:.1f}%")
        print(f"  Comfort gain: {result['comparison']['comfort_improvement']:.2%}")
