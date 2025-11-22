"""
Machine Learning Pipeline for GreenPath.
Trains and evaluates multiple ML models for thermal comfort prediction on road segments.
Implements proper train/validation/test splits, cross-validation, and model comparison.
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, VotingRegressor, StackingRegressor
)

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

from config import CACHE_DIR, WEIGHTS

# Create models directory
MODELS_DIR = os.path.join(CACHE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def calculate_utci_labels(features_df, air_temp=35, humidity=40, wind_speed=2, noise_level=0.05):
    """
    Generate synthetic UTCI-based comfort labels for training.
    UTCI (Universal Thermal Climate Index) is a standard for thermal comfort assessment.

    Args:
        features_df: DataFrame with NDVI, LST, slope, shadow columns
        air_temp: Ambient air temperature (°C)
        humidity: Relative humidity (%)
        wind_speed: Wind speed (m/s)
        noise_level: Amount of random noise to add (simulates measurement uncertainty)

    Returns:
        Series with comfort scores (0-1, higher = more comfortable)
    """
    # Estimate mean radiant temperature from LST
    mrt = features_df['lst'] + 5

    # Vegetation reduces radiant heat (with non-linear effect)
    mrt = mrt - 3 * features_df['ndvi'] - 0.5 * (features_df['ndvi'] ** 2)

    # Shadow reduces radiant heat
    mrt = mrt - 2 * features_df['shadow']

    # Simplified UTCI-like formula with interaction terms
    utci = (
        0.5 * air_temp +
        0.3 * mrt +
        0.1 * humidity / 10 -
        0.3 * wind_speed +
        0.2 * features_df['slope'] +
        # Interaction: vegetation effectiveness increases with temperature
        0.1 * features_df['ndvi'] * (features_df['lst'] - 40) / 10
    )

    # Convert to comfort score (invert and normalize)
    utci_min, utci_max = utci.min(), utci.max()
    comfort = 1 - (utci - utci_min) / (utci_max - utci_min + 1e-6)

    # Add realistic noise to simulate measurement uncertainty and human perception variability
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, noise_level, len(comfort))
    comfort = comfort + noise

    # Clip to valid range
    comfort = np.clip(comfort, 0, 1)

    return comfort


def prepare_dataset(hex_grid):
    """
    Prepare dataset for ML training from hexagon grid data.

    Args:
        hex_grid: GeoDataFrame with preprocessed features

    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
    """
    # Define features
    feature_names = ['ndvi', 'lst', 'slope', 'shadow']

    # Extract features
    X = hex_grid[feature_names].values

    # Generate synthetic comfort labels using UTCI
    y = calculate_utci_labels(hex_grid).values

    # Handle any NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    return X, y, feature_names


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.

    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with train, val, test splits
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation set from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    print(f"Dataset split:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


class BaselineModel:
    """
    Baseline model using weighted average formula.
    This serves as the comparison baseline for ML models.
    """
    def __init__(self, weights=None):
        if weights is None:
            # Default weights from config
            self.weights = np.array([
                WEIGHTS['ndvi'],
                WEIGHTS['lst'],
                WEIGHTS['slope'],
                WEIGHTS['shadow']
            ])
        else:
            self.weights = np.array(weights)
        self.name = "Weighted Average Baseline"

    def fit(self, X, y):
        """Baseline doesn't need fitting, but we include for API consistency."""
        return self

    def predict(self, X):
        """
        Predict using weighted average.
        Note: Assumes X columns are [ndvi, lst, slope, shadow]
        and that they need normalization.
        """
        # Normalize each feature to 0-1
        X_norm = X.copy()
        for i in range(X.shape[1]):
            col_min, col_max = X[:, i].min(), X[:, i].max()
            if col_max > col_min:
                X_norm[:, i] = (X[:, i] - col_min) / (col_max - col_min)
            else:
                X_norm[:, i] = 0.5

        # Invert LST and slope (lower is better)
        X_norm[:, 1] = 1 - X_norm[:, 1]  # LST
        X_norm[:, 2] = 1 - X_norm[:, 2]  # slope

        # Weighted sum
        predictions = np.dot(X_norm, self.weights)
        return np.clip(predictions, 0, 1)


def evaluate_model(model, X, y, dataset_name=""):
    """
    Evaluate a model and return metrics.

    Args:
        model: Trained model with predict method
        X: Feature matrix
        y: True labels
        dataset_name: Name of dataset for printing

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X)

    # Regression metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Additional metrics
    mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

    if dataset_name:
        print(f"\n{dataset_name} Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    return metrics


def train_classical_models(data_splits, scaler):
    """
    Train classical ML models.

    Args:
        data_splits: Dictionary with train/val/test splits
        scaler: Fitted StandardScaler

    Returns:
        Dictionary of trained models
    """
    X_train = scaler.transform(data_splits['X_train'])
    y_train = data_splits['y_train']

    models = {}

    # 1. Linear Regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr

    # 2. Ridge Regression (L2 regularization)
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models['Ridge Regression'] = ridge

    # 3. Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt

    # 4. Support Vector Regression
    print("Training SVR...")
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    models['SVR'] = svr

    # 5. K-Nearest Neighbors
    print("Training KNN...")
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    models['KNN'] = knn

    return models


def train_ensemble_models(data_splits, scaler):
    """
    Train ensemble and advanced ML models.

    Args:
        data_splits: Dictionary with train/val/test splits
        scaler: Fitted StandardScaler

    Returns:
        Dictionary of trained models
    """
    X_train = scaler.transform(data_splits['X_train'])
    y_train = data_splits['y_train']

    models = {}

    # 1. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # 2. Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb

    # 3. AdaBoost
    print("Training AdaBoost...")
    ada = AdaBoostRegressor(
        n_estimators=50,
        learning_rate=0.1,
        random_state=42
    )
    ada.fit(X_train, y_train)
    models['AdaBoost'] = ada

    # 4. Stacking Ensemble
    print("Training Stacking Ensemble...")
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)),
        ('ridge', Ridge(alpha=1.0))
    ]
    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=0.5),
        cv=5
    )
    stacking.fit(X_train, y_train)
    models['Stacking Ensemble'] = stacking

    return models


def perform_cross_validation(model, X, y, cv=5, model_name=""):
    """
    Perform k-fold cross-validation.

    Args:
        model: Model to evaluate
        X: Feature matrix
        y: Target labels
        cv: Number of folds
        model_name: Name for printing

    Returns:
        Dictionary with CV results
    """
    # Use negative MSE (sklearn convention) and convert back
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)

    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    results = {
        'RMSE_mean': rmse_scores.mean(),
        'RMSE_std': rmse_scores.std(),
        'R2_mean': r2_scores.mean(),
        'R2_std': r2_scores.std()
    }

    if model_name:
        print(f"\n{model_name} Cross-Validation ({cv}-fold):")
        print(f"  RMSE: {results['RMSE_mean']:.4f} (+/- {results['RMSE_std']:.4f})")
        print(f"  R2: {results['R2_mean']:.4f} (+/- {results['R2_std']:.4f})")

    return results


def hyperparameter_tuning(data_splits, scaler):
    """
    Perform hyperparameter tuning for best models using GridSearchCV.

    Args:
        data_splits: Dictionary with train/val/test splits
        scaler: Fitted StandardScaler

    Returns:
        Best model after tuning
    """
    X_train = scaler.transform(data_splits['X_train'])
    y_train = data_splits['y_train']

    print("\nPerforming Hyperparameter Tuning...")

    # Tune Random Forest
    rf_params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [3, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"\nBest Random Forest parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

    return grid_search.best_estimator_


def analyze_feature_importance(model, feature_names, model_name=""):
    """
    Analyze and visualize feature importance.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name for plotting

    Returns:
        Dictionary with feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None

    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{model_name} Feature Importance:")
    for feature, importance in sorted_importance:
        print(f"  {feature}: {importance:.4f}")

    return importance_dict


def error_analysis(model, X, y, feature_names, scaler):
    """
    Perform error analysis to understand model weaknesses.

    Args:
        model: Trained model
        X: Feature matrix (unscaled)
        y: True labels
        feature_names: List of feature names
        scaler: Fitted scaler

    Returns:
        Dictionary with error analysis results
    """
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    errors = y - y_pred
    abs_errors = np.abs(errors)

    # Find samples with highest errors
    high_error_idx = np.argsort(abs_errors)[-10:]

    print("\nError Analysis:")
    print(f"  Mean Error: {errors.mean():.4f}")
    print(f"  Error Std: {errors.std():.4f}")
    print(f"  Max Absolute Error: {abs_errors.max():.4f}")

    # Analyze high-error samples
    print("\n  High-error samples characteristics:")
    high_error_samples = pd.DataFrame(X[high_error_idx], columns=feature_names)
    high_error_samples['true_comfort'] = y[high_error_idx]
    high_error_samples['predicted'] = y_pred[high_error_idx]
    high_error_samples['error'] = errors[high_error_idx]

    print(high_error_samples.describe())

    # Identify patterns in errors
    results = {
        'mean_error': errors.mean(),
        'error_std': errors.std(),
        'high_error_samples': high_error_samples
    }

    return results


def compare_models(all_models, data_splits, scaler, feature_names):
    """
    Compare all models and create comparison table.

    Args:
        all_models: Dictionary of all trained models
        data_splits: Dictionary with train/val/test splits
        scaler: Fitted StandardScaler
        feature_names: List of feature names

    Returns:
        DataFrame with comparison results
    """
    results = []

    X_train = scaler.transform(data_splits['X_train'])
    X_val = scaler.transform(data_splits['X_val'])
    X_test = scaler.transform(data_splits['X_test'])

    for name, model in all_models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print('='*50)

        # Training metrics
        train_metrics = evaluate_model(model, X_train, data_splits['y_train'], "Training")

        # Validation metrics
        val_metrics = evaluate_model(model, X_val, data_splits['y_val'], "Validation")

        # Test metrics
        test_metrics = evaluate_model(model, X_test, data_splits['y_test'], "Test")

        # Cross-validation (only on training data)
        if name != "Weighted Average Baseline":
            cv_results = perform_cross_validation(
                model, X_train, data_splits['y_train'], cv=5, model_name=name
            )
        else:
            cv_results = {'RMSE_mean': train_metrics['RMSE'], 'R2_mean': train_metrics['R2']}

        # Feature importance
        if name not in ["Weighted Average Baseline", "SVR", "KNN"]:
            analyze_feature_importance(model, feature_names, name)

        results.append({
            'Model': name,
            'Train_RMSE': train_metrics['RMSE'],
            'Train_R2': train_metrics['R2'],
            'Val_RMSE': val_metrics['RMSE'],
            'Val_R2': val_metrics['R2'],
            'Test_RMSE': test_metrics['RMSE'],
            'Test_R2': test_metrics['R2'],
            'Test_MAE': test_metrics['MAE'],
            'CV_RMSE': cv_results.get('RMSE_mean', 0)
        })

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Test_R2', ascending=False)

    return comparison_df


def plot_results(comparison_df, save_path=None):
    """
    Create visualization of model comparison results.

    Args:
        comparison_df: DataFrame with comparison results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: R2 scores
    models = comparison_df['Model']
    x = np.arange(len(models))
    width = 0.25

    axes[0].bar(x - width, comparison_df['Train_R2'], width, label='Train', color='blue', alpha=0.7)
    axes[0].bar(x, comparison_df['Val_R2'], width, label='Validation', color='orange', alpha=0.7)
    axes[0].bar(x + width, comparison_df['Test_R2'], width, label='Test', color='green', alpha=0.7)

    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Model Comparison - R² Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: RMSE
    axes[1].bar(x - width, comparison_df['Train_RMSE'], width, label='Train', color='blue', alpha=0.7)
    axes[1].bar(x, comparison_df['Val_RMSE'], width, label='Validation', color='orange', alpha=0.7)
    axes[1].bar(x + width, comparison_df['Test_RMSE'], width, label='Test', color='green', alpha=0.7)

    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Model Comparison - RMSE (Lower is Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def save_best_model(model, scaler, feature_names, metrics, model_name):
    """
    Save the best model along with scaler and metadata.

    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        metrics: Dictionary with model metrics
        model_name: Name of the model
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat()
    }

    save_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    pickle.dump(model_data, open(save_path, 'wb'))

    print(f"\nBest model saved to: {save_path}")

    # Also save metrics as JSON for easy viewing
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'timestamp': model_data['timestamp']
        }, f, indent=2)


def load_best_model():
    """
    Load the saved best model.

    Returns:
        Dictionary with model, scaler, and metadata
    """
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')

    if os.path.exists(model_path):
        return pickle.load(open(model_path, 'rb'))
    else:
        return None


def run_ml_pipeline(hex_grid, save_plots=True):
    """
    Run the complete ML pipeline: data preparation, training, evaluation, comparison.

    Args:
        hex_grid: GeoDataFrame with preprocessed features
        save_plots: Whether to save comparison plots

    Returns:
        Dictionary with best model and comparison results
    """
    print("\n" + "="*60)
    print("GreenPath ML Pipeline - Training and Evaluation")
    print("="*60)

    # 1. Prepare dataset
    print("\n1. Preparing Dataset...")
    X, y, feature_names = prepare_dataset(hex_grid)
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {feature_names}")

    # 2. Split data
    print("\n2. Splitting Data...")
    data_splits = split_data(X, y)

    # 3. Scale features
    print("\n3. Scaling Features...")
    scaler = StandardScaler()
    scaler.fit(data_splits['X_train'])

    # 4. Train baseline model
    print("\n4. Training Baseline Model...")
    baseline = BaselineModel()
    baseline.fit(data_splits['X_train'], data_splits['y_train'])

    # 5. Train classical ML models
    print("\n5. Training Classical ML Models...")
    classical_models = train_classical_models(data_splits, scaler)

    # 6. Train ensemble models
    print("\n6. Training Ensemble Models...")
    ensemble_models = train_ensemble_models(data_splits, scaler)

    # 7. Hyperparameter tuning
    print("\n7. Hyperparameter Tuning...")
    tuned_rf = hyperparameter_tuning(data_splits, scaler)

    # Combine all models
    all_models = {'Weighted Average Baseline': baseline}
    all_models.update(classical_models)
    all_models.update(ensemble_models)
    all_models['Tuned Random Forest'] = tuned_rf

    # 8. Compare all models
    print("\n8. Comparing All Models...")
    comparison_df = compare_models(all_models, data_splits, scaler, feature_names)

    # 9. Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(comparison_df.to_string(index=False))

    # 10. Error analysis on best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = all_models[best_model_name]

    print(f"\n\nBest Model: {best_model_name}")
    print("="*60)

    if best_model_name != "Weighted Average Baseline":
        error_analysis(best_model, data_splits['X_test'], data_splits['y_test'],
                      feature_names, scaler)

    # 11. Save best model
    best_metrics = {
        'Test_RMSE': comparison_df.iloc[0]['Test_RMSE'],
        'Test_R2': comparison_df.iloc[0]['Test_R2'],
        'Test_MAE': comparison_df.iloc[0]['Test_MAE']
    }
    save_best_model(best_model, scaler, feature_names, best_metrics, best_model_name)

    # 12. Save comparison results
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison results saved to: {comparison_path}")

    # 13. Create visualization
    if save_plots:
        plot_path = os.path.join(MODELS_DIR, 'model_comparison.png')
        plot_results(comparison_df, plot_path)

    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'feature_names': feature_names,
        'comparison': comparison_df,
        'all_models': all_models
    }


def optimize_routing_weights(G, hex_grid, n_samples=50, n_iterations=100):
    """
    Use ML optimization to find optimal comfort/distance weights for routing.

    This finds weights that maximize a utility function balancing:
    - Comfort improvement over fast route
    - Minimal distance penalty

    Args:
        G: NetworkX graph with comfort scores
        hex_grid: GeoDataFrame with comfort data
        n_samples: Number of random route samples to evaluate
        n_iterations: Number of optimization iterations

    Returns:
        Dictionary with optimal weights and evaluation results
    """
    from routing import get_nearest_node, find_shortest_route, find_coolest_route, get_route_stats
    from scipy.optimize import minimize
    import random

    print("\n" + "="*60)
    print("Optimizing Routing Weights using ML")
    print("="*60)

    # Get graph bounds for sampling random points
    nodes = list(G.nodes(data=True))
    node_coords = [(data['y'], data['x']) for _, data in nodes]  # lat, lon in projected

    # Sample random start/end pairs
    print(f"\nSampling {n_samples} random route pairs...")
    random.seed(42)
    route_pairs = []

    node_ids = list(G.nodes())
    for _ in range(n_samples * 2):  # Sample more, filter valid ones
        start_node = random.choice(node_ids)
        end_node = random.choice(node_ids)
        if start_node != end_node:
            # Check if route exists
            try:
                path = find_shortest_route(G, start_node, end_node)
                if path and len(path) > 5:  # Minimum path length
                    route_pairs.append((start_node, end_node))
                    if len(route_pairs) >= n_samples:
                        break
            except:
                continue

    print(f"  Found {len(route_pairs)} valid route pairs")

    if len(route_pairs) < 10:
        print("[WARN] Not enough valid routes for optimization")
        return {'optimal_comfort_weight': 0.7, 'optimal_distance_weight': 0.3}

    def evaluate_weights(weights):
        """
        Evaluate a set of weights across all route pairs.
        Returns negative utility (for minimization).
        """
        comfort_weight = weights[0]
        distance_weight = 1 - comfort_weight

        total_utility = 0
        valid_routes = 0

        for start_node, end_node in route_pairs:
            try:
                # Get fast route
                fast_path = find_shortest_route(G, start_node, end_node)
                fast_stats = get_route_stats(G, fast_path, hex_grid)

                # Get cool route with current weights
                cool_path = find_coolest_route(G, start_node, end_node,
                                               comfort_weight, distance_weight)
                cool_stats = get_route_stats(G, cool_path, hex_grid)

                if fast_stats and cool_stats and fast_stats['distance_m'] > 0:
                    # Calculate utility components
                    comfort_gain = cool_stats['avg_comfort'] - fast_stats['avg_comfort']
                    distance_penalty = (cool_stats['distance_m'] - fast_stats['distance_m']) / fast_stats['distance_m']

                    # Utility function: balance comfort gain vs distance penalty
                    # Scale comfort gain more aggressively since differences are small
                    # Penalize excessive distance more moderately
                    utility = comfort_gain * 50 - distance_penalty * 3

                    # Bonus for meaningful comfort improvement with acceptable distance
                    if comfort_gain > 0.005 and distance_penalty < 0.5:
                        utility += 2

                    # Penalty for routes that are much longer with little benefit
                    if distance_penalty > 0.3 and comfort_gain < 0.01:
                        utility -= 1

                    total_utility += utility
                    valid_routes += 1
            except:
                continue

        if valid_routes == 0:
            return 1000  # High penalty for invalid weights

        avg_utility = total_utility / valid_routes
        return -avg_utility  # Negative because we minimize

    # Grid search for initial estimate
    print("\nPerforming grid search for initial weights...")
    best_weight = 0.5
    best_utility = float('inf')

    for w in np.linspace(0.1, 0.9, 17):
        utility = evaluate_weights([w])
        if utility < best_utility:
            best_utility = utility
            best_weight = w

    print(f"  Grid search best: comfort_weight={best_weight:.2f}, utility={-best_utility:.4f}")

    # Fine-tune with scipy optimization
    print("\nFine-tuning with optimization...")
    result = minimize(
        evaluate_weights,
        x0=[best_weight],
        method='L-BFGS-B',
        bounds=[(0.1, 0.9)],
        options={'maxiter': n_iterations}
    )

    optimal_comfort_weight = result.x[0]
    optimal_distance_weight = 1 - optimal_comfort_weight

    # Apply minimum comfort weight to ensure route always considers comfort
    # Even if optimization suggests pure distance, we keep some comfort priority
    min_comfort_weight = 0.5
    if optimal_comfort_weight < min_comfort_weight:
        print(f"\n  Note: Optimizer suggested {optimal_comfort_weight:.2f} comfort weight")
        print(f"  Applying minimum threshold of {min_comfort_weight} for balanced routing")
        optimal_comfort_weight = min_comfort_weight
        optimal_distance_weight = 1 - min_comfort_weight

    # Evaluate final weights
    print("\nEvaluating optimal weights...")
    final_results = {
        'comfort_improvements': [],
        'distance_penalties': [],
        'utilities': []
    }

    for start_node, end_node in route_pairs[:20]:  # Evaluate on subset
        try:
            fast_path = find_shortest_route(G, start_node, end_node)
            fast_stats = get_route_stats(G, fast_path, hex_grid)

            cool_path = find_coolest_route(G, start_node, end_node,
                                           optimal_comfort_weight, optimal_distance_weight)
            cool_stats = get_route_stats(G, cool_path, hex_grid)

            if fast_stats and cool_stats and fast_stats['distance_m'] > 0:
                comfort_gain = cool_stats['avg_comfort'] - fast_stats['avg_comfort']
                distance_penalty = (cool_stats['distance_m'] - fast_stats['distance_m']) / fast_stats['distance_m']

                final_results['comfort_improvements'].append(comfort_gain)
                final_results['distance_penalties'].append(distance_penalty)
        except:
            continue

    # Calculate summary statistics
    avg_comfort_improvement = np.mean(final_results['comfort_improvements']) if final_results['comfort_improvements'] else 0
    avg_distance_penalty = np.mean(final_results['distance_penalties']) if final_results['distance_penalties'] else 0

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nOptimal Weights:")
    print(f"  Comfort Weight: {optimal_comfort_weight:.3f}")
    print(f"  Distance Weight: {optimal_distance_weight:.3f}")
    print(f"\nExpected Performance:")
    print(f"  Avg Comfort Improvement: {avg_comfort_improvement:.2%}")
    print(f"  Avg Distance Penalty: {avg_distance_penalty:.2%}")

    # Save optimal weights
    weights_data = {
        'optimal_comfort_weight': float(optimal_comfort_weight),
        'optimal_distance_weight': float(optimal_distance_weight),
        'avg_comfort_improvement': float(avg_comfort_improvement),
        'avg_distance_penalty': float(avg_distance_penalty),
        'n_samples': len(route_pairs),
        'timestamp': datetime.now().isoformat()
    }

    weights_path = os.path.join(MODELS_DIR, 'optimal_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(weights_data, f, indent=2)

    print(f"\nOptimal weights saved to: {weights_path}")

    return weights_data


def load_optimal_weights():
    """
    Load optimized routing weights from file.

    Returns:
        Tuple of (comfort_weight, distance_weight) or default values
    """
    weights_path = os.path.join(MODELS_DIR, 'optimal_weights.json')

    if os.path.exists(weights_path):
        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)
            return data['optimal_comfort_weight'], data['optimal_distance_weight']
        except:
            pass

    # Default weights
    return 0.7, 0.3


if __name__ == '__main__':
    # Test the ML pipeline
    from data_collection import collect_all_data
    from preprocessing import preprocess_data
    from scoring import calculate_comfort_scores
    from routing import assign_comfort_to_edges

    print("Loading and preprocessing data...")
    data = collect_all_data()
    hex_grid = preprocess_data(data)

    # Run ML pipeline for comfort scoring
    results = run_ml_pipeline(hex_grid, save_plots=True)

    print("\n" + "="*60)
    print("ML Pipeline Complete!")
    print("="*60)

    # Run weight optimization
    print("\nPreparing for weight optimization...")
    hex_grid = calculate_comfort_scores(hex_grid, method='ml')
    G = assign_comfort_to_edges(data['roads'], hex_grid)

    weight_results = optimize_routing_weights(G, hex_grid)

    print("\n" + "="*60)
    print("All Optimization Complete!")
    print("="*60)
