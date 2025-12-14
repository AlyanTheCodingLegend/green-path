"""
Flask API Backend for GreenPath.
Provides REST endpoints for the Next.js frontend with real-time progress updates via SSE.
"""

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import json
import pickle
import os
import queue
import threading
from datetime import datetime

from config import CITIES, get_city_config, CACHE_DIR
from data_collection import collect_all_data_for_city
from preprocessing import preprocess_data
from scoring import calculate_comfort_scores
from routing import assign_comfort_to_edges, compare_routes, get_nearest_node

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Store progress queues for SSE
progress_queues = {}


class ProgressTracker:
    """Track and emit progress updates for long-running operations."""

    def __init__(self, operation_id):
        self.operation_id = operation_id
        self.queue = queue.Queue()
        progress_queues[operation_id] = self.queue

    def emit(self, message, progress=None, data=None):
        """Emit a progress update."""
        event = {
            'message': message,
            'progress': progress,
            'timestamp': datetime.now().isoformat()
        }
        if data:
            event['data'] = data

        self.queue.put(event)

    def complete(self, data=None):
        """Mark operation as complete."""
        event = {
            'message': 'Complete',
            'progress': 100,
            'complete': True,
            'timestamp': datetime.now().isoformat()
        }
        if data:
            event['data'] = data

        self.queue.put(event)

    def error(self, error_message):
        """Report an error."""
        event = {
            'message': error_message,
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        self.queue.put(event)


def load_city_data_with_progress(city_name, tracker):
    """Load city data with progress tracking."""
    try:
        city_config = get_city_config(city_name)
        city_cache_dir = os.path.join(CACHE_DIR, city_name.lower())
        complete_cache = os.path.join(city_cache_dir, 'complete_data.pkl')

        # Check for cached data
        if os.path.exists(complete_cache):
            tracker.emit(f"Loading cached data for {city_name}...", 20)
            data = pickle.load(open(complete_cache, 'rb'))
            tracker.emit(f"Loaded {city_name} data from cache", 100)
            return data['hex_grid'], data['G'], data['raw_data']

        # Collect raw data
        tracker.emit(f"Collecting satellite and map data for {city_name}...", 10)
        raw_data = collect_all_data_for_city(city_config)

        tracker.emit("Processing vegetation and temperature data...", 40)
        hex_grid = preprocess_data(raw_data)

        tracker.emit("Calculating thermal comfort scores...", 60)
        hex_grid = calculate_comfort_scores(hex_grid, method='ml')

        tracker.emit("Assigning scores to road network...", 80)
        G = assign_comfort_to_edges(raw_data['roads'], hex_grid)

        # Cache the data
        tracker.emit("Caching data for faster future access...", 90)
        os.makedirs(city_cache_dir, exist_ok=True)
        pickle.dump({
            'hex_grid': hex_grid,
            'G': G,
            'raw_data': raw_data
        }, open(complete_cache, 'wb'))

        tracker.emit(f"Successfully loaded {city_name} data", 100)
        return hex_grid, G, raw_data

    except Exception:
        raise


@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities."""
    cities_list = []
    for city_name, config in CITIES.items():
        cities_list.append({
            'name': city_name,
            'lat': config['lat'],
            'lon': config['lon']
        })

    return jsonify({'cities': cities_list})


@app.route('/api/city/<city_name>/data', methods=['GET'])
def get_city_data(city_name):
    """Get preprocessed city data (hexagons with comfort scores)."""
    if city_name not in CITIES:
        return jsonify({'error': 'City not found'}), 404

    try:
        city_cache_dir = os.path.join(CACHE_DIR, city_name.lower())
        complete_cache = os.path.join(city_cache_dir, 'complete_data.pkl')

        if not os.path.exists(complete_cache):
            return jsonify({
                'error': 'City data not loaded yet',
                'message': 'Please use /api/city/<city_name>/load endpoint first'
            }), 404

        data = pickle.load(open(complete_cache, 'rb'))
        hex_grid = data['hex_grid']

        # Convert hexagons to GeoJSON
        features = []
        for idx, row in hex_grid.iterrows():
            geometry = row.geometry.__geo_interface__
            properties = {
                'hex_id': idx,
                'comfort_score': float(row['comfort_score']),
                'ndvi': float(row['ndvi']),
                'lst': float(row['lst']),
                'slope': float(row['slope']),
                'shadow': float(row['shadow']),
                'category': str(row.get('comfort_category', 'Fair'))
            }
            features.append({
                'type': 'Feature',
                'geometry': geometry,
                'properties': properties
            })

        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        return jsonify({
            'city': city_name,
            'hexagons': geojson,
            'stats': {
                'total': len(hex_grid),
                'mean_comfort': float(hex_grid['comfort_score'].mean()),
                'min_comfort': float(hex_grid['comfort_score'].min()),
                'max_comfort': float(hex_grid['comfort_score'].max())
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/city/<city_name>/load', methods=['POST'])
def load_city_data_endpoint(city_name):
    """Initiate city data loading (returns operation ID for SSE)."""
    if city_name not in CITIES:
        return jsonify({'error': 'City not found'}), 404

    # Create operation ID
    operation_id = f"load_{city_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Start loading in background thread
    def load_in_background():
        tracker = ProgressTracker(operation_id)
        try:
            print(f"Starting data loading for {city_name}...") 
            hex_grid, G, raw_data = load_city_data_with_progress(city_name, tracker)
            tracker.complete({'city': city_name})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in background thread: {e}")
            tracker.error(str(e))

    thread = threading.Thread(target=load_in_background)
    thread.daemon = True
    thread.start()

    return jsonify({'operation_id': operation_id})


@app.route('/api/progress/<operation_id>')
def progress_stream(operation_id):
    """Server-Sent Events endpoint for progress updates."""

    def generate():
        if operation_id not in progress_queues:
            yield f"data: {json.dumps({'error': 'Operation not found'})}\n\n"
            return

        q = progress_queues[operation_id]

        while True:
            try:
                # Wait for next update (timeout after 30 seconds)
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"

                # If complete or error, cleanup and exit
                if event.get('complete') or event.get('error'):
                    del progress_queues[operation_id]
                    break

            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'keepalive': True})}\n\n"

    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET'
        }
    )
    return response


@app.route('/api/routes/compare', methods=['POST'])
def compare_routes_endpoint():
    """Compare cool route vs fast route."""
    data = request.json

    city_name = data.get('city')
    start_lat = data.get('start_lat')
    start_lon = data.get('start_lon')
    end_lat = data.get('end_lat')
    end_lon = data.get('end_lon')

    if not all([city_name, start_lat, start_lon, end_lat, end_lon]):
        return jsonify({'error': 'Missing required parameters'}), 400

    if city_name not in CITIES:
        return jsonify({'error': 'City not found'}), 404

    try:
        # Load city data
        city_cache_dir = os.path.join(CACHE_DIR, city_name.lower())
        complete_cache = os.path.join(city_cache_dir, 'complete_data.pkl')

        if not os.path.exists(complete_cache):
            return jsonify({
                'error': 'City data not loaded',
                'message': 'Please load city data first'
            }), 404

        cached_data = pickle.load(open(complete_cache, 'rb'))
        hex_grid = cached_data['hex_grid']
        G = cached_data['G']

        # Find routes
        result = compare_routes(G, start_lat, start_lon, end_lat, end_lon, hex_grid)

        if not result or not result.get('fast_route') or not result.get('cool_route'):
            return jsonify({'error': 'Could not find routes between these points'}), 404

        # Convert routes to GeoJSON
        def route_to_geojson(route_stats):
            if not route_stats or not route_stats.get('geometry'):
                return None

            from shapely.ops import transform
            import pyproj

            graph_crs = route_stats.get('crs', 'EPSG:32643')
            transformer = pyproj.Transformer.from_crs(graph_crs, 'EPSG:4326', always_xy=True)

            geom_wgs84 = transform(
                lambda x, y: transformer.transform(x, y),
                route_stats['geometry']
            )

            if geom_wgs84.geom_type == 'MultiLineString':
                coordinates = [list(line.coords) for line in geom_wgs84.geoms]
                geom_type = 'MultiLineString'
            else:
                coordinates = list(geom_wgs84.coords)
                geom_type = 'LineString'

            return {
                'type': 'Feature',
                'geometry': {
                    'type': geom_type,
                    'coordinates': coordinates
                },
                'properties': {
                    'distance_km': route_stats['distance_km'],
                    'avg_comfort': route_stats['avg_comfort'],
                    'walking_time_min': route_stats['walking_time_min'],
                    'tree_coverage_pct': route_stats['tree_coverage_pct']
                }
            }

        fast_geojson = route_to_geojson(result['fast_route'])
        cool_geojson = route_to_geojson(result['cool_route'])

        return jsonify({
            'fast_route': fast_geojson,
            'cool_route': cool_geojson,
            'comparison': result['comparison']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Run Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
