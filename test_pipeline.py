"""
Test script for GreenPath data pipeline and ML training.
Demonstrates the complete workflow including ML model training and evaluation.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, '.')

from data_collection import collect_all_data
from preprocessing import preprocess_data
from scoring import calculate_comfort_scores
from routing import assign_comfort_to_edges, compare_routes
from ml_pipeline import run_ml_pipeline


def test_basic_pipeline():
    """Test basic data pipeline without ML training."""
    print("\n" + "="*60)
    print("Testing Basic GreenPath Pipeline")
    print("="*60)

    # Collect data
    data = collect_all_data()

    # Preprocess
    hex_grid = preprocess_data(data)

    # Calculate scores using weighted method
    hex_grid = calculate_comfort_scores(hex_grid, method='weighted')

    # Assign to roads
    G = assign_comfort_to_edges(data['roads'], hex_grid)

    # Test routing
    start = (33.5651, 73.0169)
    end = (33.5751, 73.0269)

    result = compare_routes(G, start[0], start[1], end[0], end[1], hex_grid)

    print("\nRoute comparison:")
    if result['fast_route']:
        print(f"Fast: {result['fast_route']['distance_km']:.2f} km, {result['fast_route']['avg_comfort']:.0%} comfort")
    if result['cool_route']:
        print(f"Cool: {result['cool_route']['distance_km']:.2f} km, {result['cool_route']['avg_comfort']:.0%} comfort")

    return data, hex_grid


def test_ml_pipeline(hex_grid=None):
    """Test the ML training pipeline with all models."""
    print("\n" + "="*60)
    print("Testing ML Training Pipeline")
    print("="*60)

    if hex_grid is None:
        # Load data if not provided
        data = collect_all_data()
        hex_grid = preprocess_data(data)

    # Run ML pipeline
    results = run_ml_pipeline(hex_grid, save_plots=True)

    # Print summary
    print("\n" + "="*60)
    print("ML PIPELINE TEST RESULTS")
    print("="*60)
    print(f"\nBest Model: {results['best_model_name']}")
    print("\nModel Comparison (sorted by Test R²):")
    print(results['comparison'][['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE']].to_string(index=False))

    return results


def test_ml_scoring(hex_grid=None):
    """Test using ML models for comfort scoring."""
    print("\n" + "="*60)
    print("Testing ML-based Comfort Scoring")
    print("="*60)

    if hex_grid is None:
        data = collect_all_data()
        hex_grid = preprocess_data(data)

    # Calculate scores using ML method
    hex_grid_ml = calculate_comfort_scores(hex_grid.copy(), method='ml')

    print("\nML-based comfort scores calculated.")
    print(f"Score range: {hex_grid_ml['comfort_score'].min():.3f} - {hex_grid_ml['comfort_score'].max():.3f}")
    print(f"Mean score: {hex_grid_ml['comfort_score'].mean():.3f}")

    return hex_grid_ml


def main():
    """Main test function."""
    print("\n" + "#"*60)
    print("#" + " "*20 + "GreenPath Test Suite" + " "*18 + "#")
    print("#"*60)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = 'all'

    if test_type == 'basic':
        # Test only basic pipeline
        data, hex_grid = test_basic_pipeline()

    elif test_type == 'ml':
        # Test only ML pipeline
        data = collect_all_data()
        hex_grid = preprocess_data(data)
        results = test_ml_pipeline(hex_grid)

    elif test_type == 'all':
        # Test everything
        data, hex_grid = test_basic_pipeline()
        results = test_ml_pipeline(hex_grid)
        hex_grid_ml = test_ml_scoring(hex_grid)

        # Compare weighted vs ML scoring
        print("\n" + "="*60)
        print("Comparison: Weighted vs ML Scoring")
        print("="*60)

        hex_grid_weighted = calculate_comfort_scores(hex_grid.copy(), method='weighted')

        print(f"\nWeighted method:")
        print(f"  Mean: {hex_grid_weighted['comfort_score'].mean():.3f}")
        print(f"  Std: {hex_grid_weighted['comfort_score'].std():.3f}")

        print(f"\nML method:")
        print(f"  Mean: {hex_grid_ml['comfort_score'].mean():.3f}")
        print(f"  Std: {hex_grid_ml['comfort_score'].std():.3f}")

        # Correlation between methods
        correlation = hex_grid_weighted['comfort_score'].corr(hex_grid_ml['comfort_score'])
        print(f"\nCorrelation between methods: {correlation:.3f}")

    else:
        print(f"Unknown test type: {test_type}")
        print("Usage: python test_pipeline.py [basic|ml|all]")
        return

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "#"*60)
    print("#" + " "*18 + "Test Suite Complete!" + " "*19 + "#")
    print("#"*60 + "\n")


if __name__ == '__main__':
    main()
