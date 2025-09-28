"""
NFL Prediction Interface
=======================

A user-friendly interface for making NFL game predictions.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import os
from nfl_predictor import NFLPredictor
from data_collector import NFLDataCollector
import json

class NFLPredictionInterface:
    """
    Interface for making NFL game predictions with user-friendly methods.
    """
    
    def __init__(self):
        self.predictor = NFLPredictor()
        self.data_collector = NFLDataCollector()
        self.is_trained = False
        
    def train_models(self, data_path=None):
        """
        Train the prediction models.
        
        Args:
            data_path (str): Path to training data CSV file
        """
        print("Training NFL prediction models...")
        
        # Load data
        if data_path and os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            data = self.predictor.load_data()
        
        # Engineer features
        data = self.predictor.engineer_features(data)
        
        # Prepare features
        X, y_spread, y_outcome = self.predictor.prepare_features(data)
        
        # Train models
        self.predictor.train_models(X, y_spread, y_outcome)
        self.is_trained = True
        
        print("Models trained successfully!")
    
    def predict_single_game(self, home_team, away_team, **factors):
        """
        Predict a single game.
        
        Args:
            home_team (str): Home team abbreviation
            away_team (str): Away team abbreviation
            **factors: Game factors (see get_default_factors for options)
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Use default factors if not provided
        if not factors:
            factors = self.get_default_factors()
        
        return self.predictor.predict_game(home_team, away_team, factors)
    
    def predict_multiple_games(self, games):
        """
        Predict multiple games at once.
        
        Args:
            games (list): List of game dictionaries with team and factor info
            
        Returns:
            list: List of prediction results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = []
        for game in games:
            prediction = self.predict_single_game(
                game['home_team'],
                game['away_team'],
                **game.get('factors', {})
            )
            predictions.append(prediction)
        
        return predictions
    
    def predict_week(self, season=2024, week=1):
        """
        Predict all games for a specific week.
        
        Args:
            season (int): NFL season year
            week (int): Week number
            
        Returns:
            list: List of predictions for the week
        """
        # Get upcoming games for the week
        upcoming_games = self.data_collector.get_upcoming_games(season, week)
        
        predictions = []
        for _, game in upcoming_games.iterrows():
            prediction = self.predict_single_game(
                game['home_team'],
                game['away_team']
            )
            predictions.append(prediction)
        
        return predictions
    
    def get_default_factors(self):
        """
        Get default game factors for prediction.
        
        Returns:
            dict: Default game factors
        """
        return {
            'home_offensive_rating': 75,
            'away_offensive_rating': 75,
            'home_defensive_rating': 75,
            'away_defensive_rating': 75,
            'home_win_pct': 0.5,
            'away_win_pct': 0.5,
            'home_avg_points': 25,
            'away_avg_points': 25,
            'home_avg_points_allowed': 25,
            'away_avg_points_allowed': 25,
            'home_rest_days': 7,
            'away_rest_days': 7,
            'weather_impact': 0,
            'is_playoff': 0
        }
    
    def get_team_factors(self, team, season=2023):
        """
        Get historical factors for a specific team.
        
        Args:
            team (str): Team abbreviation
            season (int): Season year
            
        Returns:
            dict: Team factors
        """
        # This would typically fetch real team data
        # For now, return sample data
        return {
            'offensive_rating': np.random.normal(75, 15),
            'defensive_rating': np.random.normal(75, 15),
            'win_pct': np.random.uniform(0.2, 0.8),
            'avg_points': np.random.normal(25, 5),
            'avg_points_allowed': np.random.normal(25, 5)
        }
    
    def create_custom_prediction(self, home_team, away_team, 
                               home_off_rating=None, away_off_rating=None,
                               home_def_rating=None, away_def_rating=None,
                               home_win_pct=None, away_win_pct=None,
                               home_avg_points=None, away_avg_points=None,
                               home_avg_allowed=None, away_avg_allowed=None,
                               home_rest=None, away_rest=None,
                               weather_temp=None, weather_wind=None,
                               is_playoff=False):
        """
        Create a custom prediction with specific factors.
        
        Args:
            home_team (str): Home team abbreviation
            away_team (str): Away team abbreviation
            home_off_rating (float): Home team offensive rating
            away_off_rating (float): Away team offensive rating
            home_def_rating (float): Home team defensive rating
            away_def_rating (float): Away team defensive rating
            home_win_pct (float): Home team win percentage
            away_win_pct (float): Away team win percentage
            home_avg_points (float): Home team average points scored
            away_avg_points (float): Away team average points scored
            home_avg_allowed (float): Home team average points allowed
            away_avg_allowed (float): Away team average points allowed
            home_rest (int): Home team rest days
            away_rest (int): Away team rest days
            weather_temp (float): Weather temperature
            weather_wind (float): Weather wind speed
            is_playoff (bool): Is playoff game
            
        Returns:
            dict: Prediction results
        """
        # Build factors dictionary
        factors = {}
        
        if home_off_rating is not None:
            factors['home_offensive_rating'] = home_off_rating
        if away_off_rating is not None:
            factors['away_offensive_rating'] = away_off_rating
        if home_def_rating is not None:
            factors['home_defensive_rating'] = home_def_rating
        if away_def_rating is not None:
            factors['away_defensive_rating'] = away_def_rating
        if home_win_pct is not None:
            factors['home_win_pct'] = home_win_pct
        if away_win_pct is not None:
            factors['away_win_pct'] = away_win_pct
        if home_avg_points is not None:
            factors['home_avg_points'] = home_avg_points
        if away_avg_points is not None:
            factors['away_avg_points'] = away_avg_points
        if home_avg_allowed is not None:
            factors['home_avg_points_allowed'] = home_avg_allowed
        if away_avg_allowed is not None:
            factors['away_avg_points_allowed'] = away_avg_allowed
        if home_rest is not None:
            factors['home_rest_days'] = home_rest
        if away_rest is not None:
            factors['away_rest_days'] = away_rest
        if weather_temp is not None:
            factors['weather_temp'] = weather_temp
        if weather_wind is not None:
            factors['weather_wind'] = weather_wind
        
        factors['is_playoff'] = 1 if is_playoff else 0
        
        # Calculate weather impact
        weather_impact = 0
        if weather_temp is not None and weather_temp < 32:
            weather_impact -= 2
        if weather_wind is not None and weather_wind > 15:
            weather_impact -= 1
        factors['weather_impact'] = weather_impact
        
        return self.predict_single_game(home_team, away_team, **factors)
    
    def display_prediction(self, prediction):
        """
        Display prediction results in a formatted way.
        
        Args:
            prediction (dict): Prediction results
        """
        print(f"\n{'='*50}")
        print(f"NFL GAME PREDICTION")
        print(f"{'='*50}")
        print(f"Home Team: {prediction['home_team']}")
        print(f"Away Team: {prediction['away_team']}")
        print(f"Predicted Spread: {prediction['predicted_spread']:+.1f}")
        print(f"Predicted Winner: {prediction['predicted_winner']}")
        print(f"Home Win Probability: {prediction['home_win_probability']:.1%}")
        print(f"Away Win Probability: {prediction['away_win_probability']:.1%}")
        print(f"{'='*50}")

# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize interface
    interface = NFLPredictionInterface()
    
    # Train models
    interface.train_models()
    
    # Example 1: Simple prediction
    print("Example 1: Simple prediction")
    prediction1 = interface.predict_single_game("KC", "NE")
    interface.display_prediction(prediction1)
    
    # Example 2: Custom prediction
    print("\nExample 2: Custom prediction")
    prediction2 = interface.create_custom_prediction(
        home_team="KC",
        away_team="NE",
        home_off_rating=85,
        away_off_rating=70,
        home_def_rating=80,
        away_def_rating=75,
        home_win_pct=0.7,
        away_win_pct=0.4,
        weather_temp=25,
        weather_wind=20
    )
    interface.display_prediction(prediction2)
    
    # Example 3: Multiple games
    print("\nExample 3: Multiple games prediction")
    games = [
        {"home_team": "KC", "away_team": "NE"},
        {"home_team": "BUF", "away_team": "MIA"},
        {"home_team": "BAL", "away_team": "CIN"}
    ]
    
    predictions = interface.predict_multiple_games(games)
    for pred in predictions:
        interface.display_prediction(pred)
