"""
Thermal comfort scoring module for GreenPath.
Calculates walkability scores using weighted combination or trained ML models.
Supports multiple ML models with proper evaluation metrics.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import WEIGHTS, CACHE_DIR

# Path to trained models
MODELS_DIR = os.path.join(CACHE_DIR, 'models')


def calculate_weighted_score(hex_grid):
    """
    Calculate thermal comfort score using weighted combination of factors.

    Args:
        hex_grid: GeoDataFrame with normalized scores for each factor

    Returns:
        GeoDataFrame with added 'comfort_score' column
    """
    hex_grid = hex_grid.copy()

    # Weighted sum of all factors
    hex_grid['comfort_score'] = (
        WEIGHTS['ndvi'] * hex_grid['ndvi_score'] +
        WEIGHTS['lst'] * hex_grid['lst_score'] +
        WEIGHTS['slope'] * hex_grid['slope_score'] +
        WEIGHTS['shadow'] * hex_grid['shadow_score']
    )

    # Ensure scores are in 0-1 range
    hex_grid['comfort_score'] = hex_grid['comfort_score'].clip(0, 1)

    return hex_grid


def calculate_synthetic_utci(hex_grid, air_temp=35, humidity=40, wind_speed=2):
    """
    Calculate synthetic UTCI-like thermal comfort index.
    This is a simplified approximation for demonstration purposes.

    UTCI considers: air temperature, radiant temperature, humidity, wind speed.

    Args:
        hex_grid: GeoDataFrame with LST and other metrics
        air_temp: Ambient air temperature (°C)
        humidity: Relative humidity (%)
        wind_speed: Wind speed (m/s)

    Returns:
        Series with UTCI-like values (lower is more comfortable)
    """
    # Estimate mean radiant temperature from LST
    # Higher LST = higher radiant temperature
    mrt = hex_grid['lst'] + 5  # Simple offset

    # Vegetation reduces radiant heat
    mrt = mrt - 3 * hex_grid['ndvi']

    # Shadow reduces radiant heat
    mrt = mrt - 2 * hex_grid['shadow']

    # Simplified UTCI-like formula
    # Based on thermal stress categories
    utci = (
        0.5 * air_temp +
        0.3 * mrt +
        0.1 * humidity / 10 -
        0.3 * wind_speed
    )

    # Adjust for slope (harder to walk uphill in heat)
    utci = utci + 0.2 * hex_grid['slope']

    return utci


def train_comfort_model(hex_grid, city_name='default'):
    """
    Train a Random Forest model to predict comfort scores.
    Uses synthetic UTCI as pseudo-labels for training.

    Args:
        hex_grid: GeoDataFrame with all features
        city_name: Name of the city for caching

    Returns:
        Trained model and scaler
    """
    # Use city-specific cache directory
    city_dir = os.path.join(CACHE_DIR, city_name.lower())
    os.makedirs(city_dir, exist_ok=True)
    cache_path = os.path.join(city_dir, 'comfort_model.pkl')

    if os.path.exists(cache_path):
        print(f"Loading comfort model for {city_name} from cache...")
        model_data = pickle.load(open(cache_path, 'rb'))
        return model_data['model'], model_data['scaler']

    print(f"Training comfort scoring model for {city_name}...")

    # Features for the model
    features = ['ndvi', 'lst', 'slope', 'shadow']
    X = hex_grid[features].values

    # Generate synthetic labels using UTCI approximation
    utci = calculate_synthetic_utci(hex_grid)

    # Convert UTCI to comfort score (invert so higher = more comfortable)
    # Normalize to 0-1 range based on LOCAL min/max
    utci_min, utci_max = utci.min(), utci.max()
    
    # Handle case with no variation
    if abs(utci_max - utci_min) < 1e-6:
        y = np.full(len(utci), 0.5) # Default to fair if no variation
    else:
        y = 1 - (utci - utci_min) / (utci_max - utci_min)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)

    # Cache the model
    pickle.dump({'model': model, 'scaler': scaler}, open(cache_path, 'wb'))

    # Print feature importance
    print("Feature importance:")
    for feat, imp in zip(features, model.feature_importances_):
        print(f"  {feat}: {imp:.3f}")

    return model, scaler


def calculate_ml_score(hex_grid, model=None, scaler=None, city_name='default'):
    """
    Calculate comfort score using ML model.

    Args:
        hex_grid: GeoDataFrame with features
        model: Trained model (or None to load best model)
        scaler: Feature scaler (or None to load from saved model)
        city_name: Name of the city for loading city-specific model

    Returns:
        GeoDataFrame with added 'comfort_score' column
    """
    hex_grid = hex_grid.copy()

    if model is None:
        # Try to load the trained best model (global)
        model_data = load_trained_model(city_name)
        if model_data is not None:
            model = model_data['model']
            scaler = model_data['scaler']
            print(f"Using trained model: {model_data.get('model_name', 'Unknown')}")
        else:
            # Fall back to training a new model for this city
            model, scaler = train_comfort_model(hex_grid, city_name)

    features = ['ndvi', 'lst', 'slope', 'shadow']
    X = hex_grid[features].values
    X_scaled = scaler.transform(X)

    hex_grid['comfort_score'] = model.predict(X_scaled)
    hex_grid['comfort_score'] = hex_grid['comfort_score'].clip(0, 1)

    return hex_grid


def load_trained_model(city_name='default'):
    """
    Load the best trained model from the models directory.
    Checks city-specific models first, then global models.

    Returns:
        Dictionary with model, scaler, and metadata, or None if not found
    """
    # 1. Try city-specific best model
    city_model_path = os.path.join(CACHE_DIR, city_name.lower(), 'best_model.pkl')
    if os.path.exists(city_model_path):
        try:
            model_data = pickle.load(open(city_model_path, 'rb'))
            print(f"[OK] Loaded trained model for {city_name}")
            return model_data
        except Exception:
            pass
            
    # 2. Try global best model
    global_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')

    if os.path.exists(global_model_path):
        try:
            model_data = pickle.load(open(global_model_path, 'rb'))
            print(f"[OK] Loaded global trained model from {global_model_path}")
            return model_data
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
            return None
    else:
        # Don't spam warnings if just falling back to on-the-fly training
        return None


def train_and_save_model(hex_grid):
    """
    Train ML models using the full pipeline and save the best one.
    
    Args:
        hex_grid: GeoDataFrame with preprocessed features

    Returns:
        Dictionary with trained model and scaler
    """
    from ml_pipeline import run_ml_pipeline

    print("\nRunning ML training pipeline...")
    results = run_ml_pipeline(hex_grid, save_plots=True)

    return {
        'model': results['best_model'],
        'scaler': results['scaler'],
        'model_name': results['best_model_name']
    }


def calculate_comfort_scores(hex_grid, method='weighted', city_name='default'):
    """
    Main function to calculate comfort scores.

    Args:
        hex_grid: GeoDataFrame with preprocessed features
        method: 'weighted' for simple weighted sum, 'ml' for Random Forest
        city_name: Name of the city (for ML model caching)

    Returns:
        GeoDataFrame with comfort scores
    """
    print(f"\nCalculating comfort scores using {method} method for {city_name}...")

    if method == 'weighted':
        hex_grid = calculate_weighted_score(hex_grid)
    elif method == 'ml':
        hex_grid = calculate_ml_score(hex_grid, city_name=city_name)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Add comfort category
    hex_grid['comfort_category'] = pd.cut(
        hex_grid['comfort_score'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )

    print(f"[OK] Comfort scores calculated")
    print(f"  Score range: {hex_grid['comfort_score'].min():.3f} - {hex_grid['comfort_score'].max():.3f}")
    print(f"  Mean score: {hex_grid['comfort_score'].mean():.3f}")

    # Print category distribution
    print("\n  Comfort distribution:")
    for cat in ['Excellent', 'Good', 'Fair', 'Poor']:
        count = (hex_grid['comfort_category'] == cat).sum()
        pct = count / len(hex_grid) * 100
        print(f"    {cat}: {count} ({pct:.1f}%)")

    return hex_grid


def get_discomfort_cost(comfort_score):
    """
    Convert comfort score to discomfort cost for routing.
    Lower comfort = higher cost.

    Args:
        comfort_score: Comfort score (0-1)

    Returns:
        Discomfort cost (higher = worse)
    """
    # Inverse relationship with exponential penalty for low comfort
    if comfort_score < 0.1:
        comfort_score = 0.1  # Avoid division by zero

    return (1 - comfort_score) ** 2 + 0.1


if __name__ == '__main__':
    # Test scoring
    from data_collection import collect_all_data
    from preprocessing import preprocess_data

    data = collect_all_data()
    hex_grid = preprocess_data(data)

    # Test weighted method
    scored_grid = calculate_comfort_scores(hex_grid, method='weighted')

    print("\nScored hexagon sample:")
    print(scored_grid[['ndvi', 'lst', 'slope', 'shadow', 'comfort_score', 'comfort_category']].head(10))
