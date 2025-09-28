# NFL Game Prediction Model

A machine learning model for predicting NFL game spreads and outcomes based on historical game data and various factors.

## Features

- **Spread Prediction**: Predicts point spreads for NFL games
- **Outcome Prediction**: Predicts game winners with probability scores
- **Multiple Factors**: Considers offensive/defensive ratings, win percentages, weather, rest days, and more
- **Easy-to-Use Interface**: Simple Python interface for making predictions
- **Customizable**: Allows custom input of game factors
- **Batch Processing**: Predict multiple games at once

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from prediction_interface import NFLPredictionInterface

# Initialize the interface
interface = NFLPredictionInterface()

# Train the models
interface.train_models()

# Make a simple prediction
prediction = interface.predict_single_game("KC", "NE")
interface.display_prediction(prediction)
```

### Custom Factors Prediction

```python
# Create a custom prediction with specific factors
prediction = interface.create_custom_prediction(
    home_team="KC",
    away_team="NE",
    home_off_rating=85,      # Home team offensive rating
    away_off_rating=70,      # Away team offensive rating
    home_def_rating=80,      # Home team defensive rating
    away_def_rating=75,      # Away team defensive rating
    home_win_pct=0.7,        # Home team win percentage
    away_win_pct=0.4,        # Away team win percentage
    home_avg_points=28,      # Home team average points
    away_avg_points=22,      # Away team average points
    home_avg_allowed=20,     # Home team average points allowed
    away_avg_allowed=26,     # Away team average points allowed
    home_rest=10,            # Home team rest days
    away_rest=6,             # Away team rest days
    weather_temp=15,         # Weather temperature
    weather_wind=25,         # Weather wind speed
    is_playoff=False         # Is playoff game
)
```

### Multiple Games Prediction

```python
# Predict multiple games
games = [
    {"home_team": "KC", "away_team": "NE"},
    {"home_team": "BUF", "away_team": "MIA"},
    {"home_team": "BAL", "away_team": "CIN"}
]

predictions = interface.predict_multiple_games(games)
for pred in predictions:
    interface.display_prediction(pred)
```

## Model Features

The model considers the following factors:

### Team Performance
- Offensive rating
- Defensive rating
- Win percentage
- Average points scored
- Average points allowed

### Game Conditions
- Rest days advantage
- Weather conditions (temperature, wind)
- Playoff status
- Home field advantage

### Derived Features
- Offensive/defensive differentials
- Win percentage differentials
- Points per game differentials
- Rest advantage
- Weather impact
- Team strength composite

## File Structure

```
gamblingMlModel/
├── nfl_predictor.py          # Core ML model implementation
├── data_collector.py         # Data collection utilities
├── prediction_interface.py   # User-friendly interface
├── example_usage.py          # Usage examples
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Model Architecture

- **Spread Prediction**: XGBoost Regressor for point spread prediction
- **Outcome Prediction**: XGBoost Classifier for win/loss prediction
- **Feature Engineering**: Automated creation of derived features
- **Cross-Validation**: Built-in model evaluation

## Example Output

```
==================================================
NFL GAME PREDICTION
==================================================
Home Team: KC
Away Team: NE
Predicted Spread: +7.2
Predicted Winner: Home
Home Win Probability: 72.5%
Away Win Probability: 27.5%
==================================================
```

## Running Examples

Run the example script to see the model in action:

```bash
python example_usage.py
```

## Data Sources

The model currently uses sample data for demonstration. In production, you would integrate with:

- NFL.com API
- ESPN API
- Pro Football Reference
- SportsData.io API
- Other NFL data providers

## Customization

### Adding New Features

To add new features, modify the `engineer_features` method in `nfl_predictor.py`:

```python
def engineer_features(self, df):
    # Add your new features here
    df['new_feature'] = df['feature1'] * df['feature2']
    return df
```

### Modifying Models

To use different ML models, update the `train_models` method:

```python
# For spread prediction
self.spread_model = RandomForestRegressor(n_estimators=100)

# For outcome prediction  
self.outcome_model = RandomForestClassifier(n_estimators=100)
```

## Performance

The model provides:
- Spread prediction with RMSE metrics
- Outcome prediction with accuracy scores
- Feature importance analysis
- Cross-validation results

## Limitations

- Currently uses sample data (replace with real NFL data)
- Weather data is simplified
- Team ratings are generated randomly
- No real-time data integration

## Future Enhancements

- Real-time data integration
- More sophisticated weather modeling
- Player-level statistics
- Injury reports integration
- Advanced ensemble methods
- Web interface
- API endpoints

## Contributing

Feel free to contribute by:
- Adding new features
- Improving data collection
- Enhancing model performance
- Adding visualizations
- Creating tests

## License

This project is for educational and research purposes.
