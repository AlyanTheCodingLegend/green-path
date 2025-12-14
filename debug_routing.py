
import pickle
import os
import networkx as nx
from routing import compare_routes, get_route_stats, get_nearest_node
from config import CACHE_DIR, COMFORT_WEIGHT, DISTANCE_WEIGHT
from scoring import get_discomfort_cost

def debug_routing():
    city_name = "Rawalpindi"
    city_cache_dir = os.path.join(CACHE_DIR, city_name.lower())
    complete_cache = os.path.join(city_cache_dir, 'complete_data.pkl')
    
    if not os.path.exists(complete_cache):
        print("Required data not found.")
        return

    print("Loading city data...")
    data = pickle.load(open(complete_cache, 'rb'))
    G = data['G']
    hex_grid = data['hex_grid']
    
    start = (33.5651, 73.0169) # Center
    end = (33.5751, 73.0269) # 1km away
    
    # 1. Inspect Edge Scores around start point
    start_node = get_nearest_node(G, start[0], start[1])
    print(f"\nScanning edges around start node {start_node}...")
    for u, v, k, d in list(G.edges(start_node, keys=True, data=True))[:5]:
        print(f"Edge {u}->{v}: Length={d.get('length'):.1f}m, Comfort={d.get('comfort_score'):.2f}, DiscomfortCost={d.get('discomfort_cost'):.4f}")

    # 2. Run Route Comparison
    print(f"\nRunning Comparison with W_C={COMFORT_WEIGHT}, W_D={DISTANCE_WEIGHT}...")
    result = compare_routes(G, start[0], start[1], end[0], end[1], hex_grid)
    
    if not result['fast_route'] or not result['cool_route']:
        print("Failed to find routes")
        return

    # 3. Analyze Costs
    def analyze_path(name, stats):
        print(f"\n{name} Analysis:")
        print(f"  Length: {stats['distance_m']:.1f}m")
        print(f"  Avg Comfort: {stats['avg_comfort']:.4f}")
        
        # Re-calculate total generic cost
        total_cost = 0
        w_dist = 0
        w_comfort = 0
        
        path = stats['path']
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge_data = G.get_edge_data(u, v)[0]
            l = edge_data.get('length', 0)
            c = edge_data.get('comfort_score', 0.5)
            dc = edge_data.get('discomfort_cost', get_discomfort_cost(0.5))
            
            # Cost func: W_D * (L/50) + W_C * DC * (L/50)
            norm_l = l / 50.0
            segment_cost = (DISTANCE_WEIGHT * norm_l) + (COMFORT_WEIGHT * dc * norm_l)
            total_cost += segment_cost
            
        print(f"  Calculated Algo Cost: {total_cost:.4f}")

    analyze_path("Fast Route", result['fast_route'])
    analyze_path("Cool Route", result['cool_route'])

    # 4. Check anomalies
    fast_comfort = result['fast_route']['avg_comfort']
    cool_comfort = result['cool_route']['avg_comfort']
    
    if fast_comfort > cool_comfort:
        print("\n[ANOMALY DETECTED]: Fast Route has higher comfort than Cool Route!")
    else:
        print("\nCheck passed: Cool route has higher/equal comfort.")

if __name__ == "__main__":
    debug_routing()
