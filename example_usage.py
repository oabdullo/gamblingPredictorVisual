"""
NFL Prediction Model - Example Usage
===================================

This script demonstrates how to use the NFL prediction model
for various scenarios and use cases.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from prediction_interface import NFLPredictionInterface
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    Main function demonstrating various usage examples.
    """
    print("NFL Prediction Model - Example Usage")
    print("=" * 50)
    
    # Initialize the prediction interface
    interface = NFLPredictionInterface()
    
    # Train the models
    print("\n1. Training Models...")
    interface.train_models()
    
    # Example 1: Basic prediction
    print("\n2. Basic Prediction Example")
    print("-" * 30)
    basic_prediction = interface.predict_single_game("KC", "NE")
    interface.display_prediction(basic_prediction)
    
    # Example 2: Custom factors prediction
    print("\n3. Custom Factors Prediction")
    print("-" * 30)
    custom_prediction = interface.create_custom_prediction(
        home_team="KC",
        away_team="NE",
        home_off_rating=90,  # Strong home offense
        away_off_rating=65,  # Weaker away offense
        home_def_rating=85,  # Strong home defense
        away_def_rating=70,  # Weaker away defense
        home_win_pct=0.8,    # 80% win rate
        away_win_pct=0.3,    # 30% win rate
        home_avg_points=32,  # High scoring home team
        away_avg_points=20,  # Lower scoring away team
        home_avg_allowed=18, # Strong home defense
        away_avg_allowed=28, # Weaker away defense
        home_rest=10,        # More rest for home team
        away_rest=6,         # Less rest for away team
        weather_temp=15,     # Cold weather
        weather_wind=25,     # High wind
        is_playoff=False
    )
    interface.display_prediction(custom_prediction)
    
    # Example 3: Multiple games prediction
    print("\n4. Multiple Games Prediction")
    print("-" * 30)
    week_games = [
        {"home_team": "KC", "away_team": "NE", "factors": {"is_playoff": 1}},
        {"home_team": "BUF", "away_team": "MIA", "factors": {"weather_temp": 5, "weather_wind": 30}},
        {"home_team": "BAL", "away_team": "CIN", "factors": {"home_win_pct": 0.9, "away_win_pct": 0.1}},
        {"home_team": "LAR", "away_team": "SF", "factors": {"is_playoff": 1, "weather_temp": 70}}
    ]
    
    multiple_predictions = interface.predict_multiple_games(week_games)
    for i, pred in enumerate(multiple_predictions, 1):
        print(f"\nGame {i}:")
        interface.display_prediction(pred)
    
    # Example 4: Scenario analysis
    print("\n5. Scenario Analysis")
    print("-" * 30)
    analyze_scenarios(interface)
    
    # Example 5: Model evaluation
    print("\n6. Model Performance Analysis")
    print("-" * 30)
    evaluate_model_performance(interface)

def analyze_scenarios(interface):
    """
    Analyze different scenarios for the same game.
    """
    base_teams = ("KC", "NE")
    
    scenarios = [
        {
            "name": "Neutral Conditions",
            "factors": {}
        },
        {
            "name": "Home Team Advantage",
            "factors": {
                "home_off_rating": 85,
                "away_off_rating": 70,
                "home_def_rating": 80,
                "away_def_rating": 75,
                "home_win_pct": 0.7,
                "away_win_pct": 0.4
            }
        },
        {
            "name": "Cold Weather Impact",
            "factors": {
                "weather_temp": 10,
                "weather_wind": 25
            }
        },
        {
            "name": "Playoff Game",
            "factors": {
                "is_playoff": True,
                "home_off_rating": 90,
                "away_off_rating": 90,
                "home_def_rating": 85,
                "away_def_rating": 85
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        prediction = interface.create_custom_prediction(
            base_teams[0], base_teams[1], **scenario['factors']
        )
        print(f"  Spread: {prediction['predicted_spread']:+.1f}")
        print(f"  Winner: {prediction['predicted_winner']}")
        print(f"  Home Win Prob: {prediction['home_win_probability']:.1%}")

def evaluate_model_performance(interface):
    """
    Evaluate model performance with sample data.
    """
    # Generate test scenarios
    test_scenarios = []
    for i in range(10):
        scenario = {
            "home_team": f"TEAM{i%5}",
            "away_team": f"TEAM{(i+1)%5}",
            "factors": {
                "home_off_rating": np.random.normal(75, 15),
                "away_off_rating": np.random.normal(75, 15),
                "home_def_rating": np.random.normal(75, 15),
                "away_def_rating": np.random.normal(75, 15),
                "home_win_pct": np.random.uniform(0.2, 0.8),
                "away_win_pct": np.random.uniform(0.2, 0.8),
                "weather_temp": np.random.normal(50, 20),
                "weather_wind": np.random.exponential(5)
            }
        }
        test_scenarios.append(scenario)
    
    # Make predictions
    predictions = interface.predict_multiple_games(test_scenarios)
    
    # Analyze results
    spreads = [p['predicted_spread'] for p in predictions]
    home_probs = [p['home_win_probability'] for p in predictions]
    
    print(f"Average predicted spread: {np.mean(spreads):.2f}")
    print(f"Spread standard deviation: {np.std(spreads):.2f}")
    print(f"Average home win probability: {np.mean(home_probs):.1%}")
    print(f"Home win probability range: {min(home_probs):.1%} - {max(home_probs):.1%}")
    
    # Count predictions by winner
    home_wins = sum(1 for p in predictions if p['predicted_winner'] == 'Home')
    away_wins = sum(1 for p in predictions if p['predicted_winner'] == 'Away')
    
    print(f"Home team wins: {home_wins}/{len(predictions)} ({home_wins/len(predictions):.1%})")
    print(f"Away team wins: {away_wins}/{len(predictions)} ({away_wins/len(predictions):.1%})")

def create_visualization():
    """
    Create visualizations for model analysis.
    """
    # This would create charts showing model performance
    # For now, just print a message
    print("\n7. Visualization")
    print("-" * 30)
    print("Visualization features would be implemented here:")
    print("- Spread prediction distribution")
    print("- Win probability analysis")
    print("- Feature importance charts")
    print("- Model performance metrics")

if __name__ == "__main__":
    main()
    create_visualization()
