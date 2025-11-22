# 🌳 GreenPath: The Shade & Walkability Navigator

A Python application that finds thermally comfortable walking routes in hot climates by prioritizing shade, greenery, and thermal comfort over shortest distance.

## Features

- **Thermal Comfort Scoring**: Analyzes vegetation (NDVI), land surface temperature (LST), terrain slope, and building shadows
- **Dual Route Comparison**: Compare "Cool Route" vs "Fast Route" side-by-side
- **Interactive Map**: Click to select start/end points with comfort heatmap overlay
- **GPX Export**: Download routes for use in GPS devices or other apps
- **Data Caching**: Efficient caching to minimize API calls

## Project Structure

```
greenpath/
├── config.py           # Configuration settings
├── data_collection.py  # GEE and OSM data fetching
├── preprocessing.py    # NDVI, LST, slope calculations
├── scoring.py          # Thermal comfort scoring model
├── ml_pipeline.py      # ML training and evaluation pipeline
├── routing.py          # Pathfinding algorithms
├── app.py              # Streamlit GUI interface
├── test_pipeline.py    # Test script for pipeline and ML
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── cache/              # Cached data (created automatically)
    └── models/         # Trained ML models
```

## Installation

### 1. Clone or Download

```bash
cd greenpath
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Google Earth Engine Authentication

GreenPath uses Google Earth Engine for satellite data. You need to authenticate:

1. **Sign up for GEE**: Go to [https://earthengine.google.com/](https://earthengine.google.com/) and sign up
2. **Authenticate**:
   ```bash
   earthengine authenticate
   ```
3. **Follow the prompts** to complete browser-based authentication

If you're using a service account:
1. Create a service account in Google Cloud Console
2. Download the JSON key file
3. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
   ```
4. Update `GEE_PROJECT` in `config.py` with your project ID

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

This will open the application in your default web browser at `http://localhost:8501`.

### Using the Interface

1. **Select Location Method**: Choose between clicking on map, entering coordinates, or searching addresses
2. **Set Start & End Points**: Mark your origin and destination
3. **Click "Find Routes"**: Calculate both cool and fast routes
4. **Compare Results**: View distance, time, comfort scores, and tree coverage
5. **Export**: Download routes as GPX files

### Command-Line Testing

Test individual modules:

```bash
# Test data collection
python data_collection.py

# Test preprocessing
python preprocessing.py

# Test scoring
python scoring.py

# Test routing
python routing.py
```

## Configuration

Edit `config.py` to customize:

- **City coordinates**: Change `CITY_CENTER_LAT`, `CITY_CENTER_LON` for different cities
- **Bounding box**: Adjust `BBOX_BUFFER` for study area size
- **Scoring weights**: Modify `WEIGHTS` to prioritize different comfort factors
- **H3 resolution**: Change `H3_RESOLUTION` (8-10 recommended)
- **Date range**: Update `DATE_START`, `DATE_END` for different seasons

## Data Sources

| Data | Source | Resolution | Purpose |
|------|--------|------------|---------|
| Vegetation (NDVI) | Sentinel-2 | 10m | Tree/grass coverage |
| Surface Temperature (LST) | Landsat 8/9 | 30m | Heat mapping |
| Elevation/Slope | SRTM | 30m | Terrain difficulty |
| Road Network | OpenStreetMap | - | Walkable paths |
| Buildings | OpenStreetMap | - | Shadow estimation |

## How Comfort Score Works (Simple Explanation)

We look at **4 things** to decide how comfortable a spot is to walk:

### 1. Trees & Grass (NDVI) - 30% of score
- Satellite photos show how green an area is
- More green = more shade = cooler = **better score**
- Parks and tree-lined streets score high

### 2. Ground Temperature (LST) - 40% of score (biggest factor!)
- Satellites measure how hot the ground is
- Concrete and asphalt get very hot in summer
- Cooler ground = **better score**

### 3. How Flat (Slope) - 20% of score
- Flat ground = easier to walk = **better score**
- Steep hills make you sweat more in heat

### 4. Building Shadows - 10% of score
- Tall buildings cast shadows on streets
- More shadow = cooler walking = **better score**

### Score Categories
- **0.0 - 0.3** = Poor (hot, no trees, steep)
- **0.3 - 0.5** = Fair
- **0.5 - 0.7** = Good
- **0.7 - 1.0** = Excellent (shady, cool, flat, green)

## Technical Scoring Formula

### Weighted Method (Default)
```
Score = 0.30 × NDVI_norm + 0.40 × LST_inv + 0.20 × Slope_inv + 0.10 × Shadow_norm
```

Where:
- `NDVI_norm`: Normalized vegetation index (0-1, higher = more vegetation)
- `LST_inv`: Inverted land surface temperature (0-1, higher = cooler)
- `Slope_inv`: Inverted slope (0-1, higher = flatter)
- `Shadow_norm`: Building shadow potential (0-1, higher = more shade)

### Machine Learning Methods

GreenPath includes a complete ML pipeline for predicting thermal comfort scores. The pipeline trains and compares multiple models:

#### Models Implemented
1. **Baseline Model**: Weighted average formula (benchmark)
2. **Classical ML Models**:
   - Linear Regression
   - Ridge Regression (L2 regularization)
   - Decision Tree Regressor
   - Support Vector Regression (SVR)
   - K-Nearest Neighbors (KNN)
3. **Ensemble Methods**:
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - AdaBoost Regressor
   - Stacking Ensemble

#### ML Pipeline Features
- **Train/Validation/Test Split**: 70%/10%/20% split for proper evaluation
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Hyperparameter Tuning**: GridSearchCV for optimal model configuration
- **Feature Importance Analysis**: Understand which factors matter most
- **Error Analysis**: Identify model weaknesses and failure cases
- **Model Comparison**: Compare all models on RMSE, MAE, R² metrics

#### Training Labels
Training labels are generated using the UTCI (Universal Thermal Climate Index) formula:
```python
UTCI = 0.5 × air_temp + 0.3 × MRT + 0.1 × humidity/10 - 0.3 × wind_speed + 0.2 × slope
```
Where MRT (Mean Radiant Temperature) is estimated from LST, NDVI, and shadow data.

#### Running the ML Pipeline
```bash
# Train all models and compare
python test_pipeline.py ml

# Run complete test (basic + ML)
python test_pipeline.py all
```

Results are saved to `cache/models/`:
- `best_model.pkl`: Best performing model
- `model_comparison.csv`: Comparison table
- `model_comparison.png`: Visualization
- `model_metrics.json`: Best model metrics

## Sample City Data

Real satellite data collected from Google Earth Engine for Pakistani cities:

| City | Hexagons | NDVI Range | LST Range | Buildings | Mean Comfort | Distribution |
|------|----------|------------|-----------|-----------|--------------|--------------|
| Rawalpindi | 254 | 0.045-0.382 | 43-54°C | 12 | 0.524 | 62% Good |
| Islamabad | 252 | 0.021-0.745 | 38-48°C | 1,098 | 0.469 | 34% Good |
| Lahore | 266 | 0.014-0.707 | 41-51°C | 1,263 | 0.415 | 65% Fair |
| Karachi | 309 | -0.034-0.499 | 16-48°C | 13,395 | 0.387 | 39% Fair |

**Key Observations:**
- Islamabad has the highest vegetation (NDVI up to 0.745) and coolest temperatures
- Karachi has the most buildings (13,395) but lowest comfort due to less greenery
- Rawalpindi has the highest percentage of "Good" comfort ratings (62%)

## Routing Algorithm

Two routes are calculated using Dijkstra's algorithm:

1. **Fast Route**: Minimizes distance (standard shortest path)
2. **Cool Route**: Minimizes combined cost:
   ```
   Cost = 0.3 × Distance + 0.7 × Discomfort × Distance
   ```

## Fallback Mode

If Google Earth Engine is unavailable (quota limits, authentication issues), GreenPath automatically generates synthetic data for demonstration purposes. This allows you to explore the interface and routing algorithms without GEE access.

## Troubleshooting

### Common Issues

1. **"GEE initialization failed"**
   - Run `earthengine authenticate` again
   - Check internet connection
   - Verify GEE account is active

2. **"Could not find routes"**
   - Points may be outside the road network
   - Try points closer to streets
   - Check if bounding box includes both points

3. **Slow first load**
   - Initial data collection takes 2-5 minutes
   - Subsequent runs use cached data
   - Reduce `BBOX_BUFFER` for faster processing

4. **Memory errors**
   - Reduce H3 resolution (use 8 instead of 9)
   - Process smaller area
   - Increase available RAM

### Clear Cache

To force fresh data download:
- Click "Clear Cache & Reload" in sidebar, or
- Delete the `cache/` folder manually

## Extending the Project

### Add New City

1. Update coordinates in `config.py`:
   ```python
   CITY_NAME = "Your City"
   CITY_CENTER_LAT = your_latitude
   CITY_CENTER_LON = your_longitude
   ```
2. Clear cache and restart

### Custom Scoring Weights

Adjust in `config.py`:
```python
WEIGHTS = {
    'ndvi': 0.40,    # Increase vegetation importance
    'lst': 0.30,     # Reduce temperature importance
    'slope': 0.20,
    'shadow': 0.10
}
```

### Add New Data Layers

1. Add fetching function in `data_collection.py`
2. Add processing in `preprocessing.py`
3. Include in scoring formula in `scoring.py`

## License

MIT License - Feel free to use, modify, and distribute.

## Acknowledgments

- Google Earth Engine for satellite data access
- OpenStreetMap contributors for road network data
- Uber H3 for hexagonal spatial indexing
- OSMnx for street network analysis
