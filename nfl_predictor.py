"""
NFL Game Prediction Model
========================

This module provides functionality to predict NFL game spreads and outcomes
based on historical game data and various factors.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class NFLPredictor:
    """
    A machine learning model for predicting NFL game spreads and outcomes.
    """
    
    def __init__(self):
        self.spread_model = None
        self.outcome_model = None
        self.feature_columns = []
        self.scaler = None
        
    def load_data(self, data_path=None):
        """
        Load NFL game data from CSV file or create sample data.
        
        Args:
            data_path (str): Path to CSV file with NFL data
            
        Returns:
            pd.DataFrame: Loaded NFL game data
        """
        if data_path and os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            # Create sample data structure for demonstration
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Create sample NFL data for demonstration purposes.
        In production, this would be replaced with real data collection.
        """
        np.random.seed(42)
        n_games = 1000
        
        data = {
            'home_team': np.random.choice(['NE', 'KC', 'BUF', 'MIA', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT'], n_games),
            'away_team': np.random.choice(['NE', 'KC', 'BUF', 'MIA', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT'], n_games),
            'home_offensive_rating': np.random.normal(75, 15, n_games),
            'away_offensive_rating': np.random.normal(75, 15, n_games),
            'home_defensive_rating': np.random.normal(75, 15, n_games),
            'away_defensive_rating': np.random.normal(75, 15, n_games),
            'home_rest_days': np.random.randint(3, 14, n_games),
            'away_rest_days': np.random.randint(3, 14, n_games),
            'home_win_pct': np.random.uniform(0.2, 0.8, n_games),
            'away_win_pct': np.random.uniform(0.2, 0.8, n_games),
            'home_avg_points': np.random.normal(25, 5, n_games),
            'away_avg_points': np.random.normal(25, 5, n_games),
            'home_avg_points_allowed': np.random.normal(25, 5, n_games),
            'away_avg_points_allowed': np.random.normal(25, 5, n_games),
            'weather_temp': np.random.normal(50, 20, n_games),
            'weather_wind': np.random.exponential(5, n_games),
            'is_playoff': np.random.choice([0, 1], n_games, p=[0.8, 0.2]),
            'home_spread': np.random.normal(0, 7, n_games),
            'home_score': np.random.poisson(25, n_games),
            'away_score': np.random.poisson(25, n_games)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate derived features
        df['point_differential'] = df['home_score'] - df['away_score']
        df['home_win'] = (df['point_differential'] > 0).astype(int)
        df['actual_spread'] = df['point_differential']
        
        # Remove games where home and away teams are the same
        df = df[df['home_team'] != df['away_team']]
        
        return df
    
    def engineer_features(self, df):
        """
        Create additional features for the ML model.
        
        Args:
            df (pd.DataFrame): Raw NFL game data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        # Create feature copies to avoid modifying original
        df = df.copy()
        
        # Offensive/Defensive differentials
        df['offensive_differential'] = df['home_offensive_rating'] - df['away_offensive_rating']
        df['defensive_differential'] = df['away_defensive_rating'] - df['home_defensive_rating']
        
        # Win percentage differential
        df['win_pct_differential'] = df['home_win_pct'] - df['away_win_pct']
        
        # Points per game differentials
        df['points_scored_differential'] = df['home_avg_points'] - df['away_avg_points']
        df['points_allowed_differential'] = df['away_avg_points_allowed'] - df['home_avg_points_allowed']
        
        # Rest advantage
        df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
        
        # Weather impact (simplified)
        df['weather_impact'] = np.where(df['weather_temp'] < 32, -2, 0) + np.where(df['weather_wind'] > 15, -1, 0)
        
        # Team strength composite
        df['home_strength'] = (df['home_offensive_rating'] + df['home_defensive_rating']) / 2
        df['away_strength'] = (df['away_offensive_rating'] + df['away_defensive_rating']) / 2
        df['strength_differential'] = df['home_strength'] - df['away_strength']
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for model training.
        
        Args:
            df (pd.DataFrame): Data with engineered features
            
        Returns:
            tuple: (X, y_spread, y_outcome) feature matrices and targets
        """
        # Define feature columns
        feature_columns = [
            'offensive_differential', 'defensive_differential', 'win_pct_differential',
            'points_scored_differential', 'points_allowed_differential', 'rest_advantage',
            'weather_impact', 'strength_differential', 'is_playoff'
        ]
        
        self.feature_columns = feature_columns
        
        # Prepare features
        X = df[feature_columns].fillna(0)
        
        # Prepare targets
        y_spread = df['actual_spread']
        y_outcome = df['home_win']
        
        return X, y_spread, y_outcome
    
    def train_models(self, X, y_spread, y_outcome):
        """
        Train both spread and outcome prediction models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y_spread (pd.Series): Spread targets
            y_outcome (pd.Series): Outcome targets
        """
        # Split data
        X_train, X_test, y_spread_train, y_spread_test, y_outcome_train, y_outcome_test = train_test_split(
            X, y_spread, y_outcome, test_size=0.2, random_state=42
        )
        
        # Train spread prediction model (regression)
        print("Training spread prediction model...")
        self.spread_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.spread_model.fit(X_train, y_spread_train)
        
        # Train outcome prediction model (classification)
        print("Training outcome prediction model...")
        self.outcome_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.outcome_model.fit(X_train, y_outcome_train)
        
        # Evaluate models
        self._evaluate_models(X_test, y_spread_test, y_outcome_test)
    
    def _evaluate_models(self, X_test, y_spread_test, y_outcome_test):
        """
        Evaluate model performance.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_spread_test (pd.Series): Test spread targets
            y_outcome_test (pd.Series): Test outcome targets
        """
        # Spread model evaluation
        spread_pred = self.spread_model.predict(X_test)
        spread_mse = mean_squared_error(y_spread_test, spread_pred)
        print(f"Spread Model MSE: {spread_mse:.2f}")
        print(f"Spread Model RMSE: {np.sqrt(spread_mse):.2f}")
        
        # Outcome model evaluation
        outcome_pred = self.outcome_model.predict(X_test)
        outcome_accuracy = accuracy_score(y_outcome_test, outcome_pred)
        print(f"Outcome Model Accuracy: {outcome_accuracy:.3f}")
        
        # Feature importance
        print("\nTop 5 Most Important Features for Spread Prediction:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.spread_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head())
    
    def predict_game(self, home_team, away_team, game_factors):
        """
        Predict spread and outcome for a specific game.
        
        Args:
            home_team (str): Home team abbreviation
            away_team (str): Away team abbreviation
            game_factors (dict): Dictionary containing game factors
            
        Returns:
            dict: Prediction results
        """
        if self.spread_model is None or self.outcome_model is None:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare input features
        input_data = pd.DataFrame([{
            'offensive_differential': game_factors.get('home_offensive_rating', 75) - game_factors.get('away_offensive_rating', 75),
            'defensive_differential': game_factors.get('away_defensive_rating', 75) - game_factors.get('home_defensive_rating', 75),
            'win_pct_differential': game_factors.get('home_win_pct', 0.5) - game_factors.get('away_win_pct', 0.5),
            'points_scored_differential': game_factors.get('home_avg_points', 25) - game_factors.get('away_avg_points', 25),
            'points_allowed_differential': game_factors.get('away_avg_points_allowed', 25) - game_factors.get('home_avg_points_allowed', 25),
            'rest_advantage': game_factors.get('home_rest_days', 7) - game_factors.get('away_rest_days', 7),
            'weather_impact': game_factors.get('weather_impact', 0),
            'strength_differential': (game_factors.get('home_offensive_rating', 75) + game_factors.get('home_defensive_rating', 75))/2 - 
                                   (game_factors.get('away_offensive_rating', 75) + game_factors.get('away_defensive_rating', 75))/2,
            'is_playoff': game_factors.get('is_playoff', 0)
        }])
        
        # Make predictions
        predicted_spread = self.spread_model.predict(input_data)[0]
        predicted_outcome_prob = self.outcome_model.predict_proba(input_data)[0]
        predicted_winner = "Home" if predicted_outcome_prob[1] > 0.5 else "Away"
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_spread': round(predicted_spread, 1),
            'predicted_winner': predicted_winner,
            'home_win_probability': round(predicted_outcome_prob[1], 3),
            'away_win_probability': round(predicted_outcome_prob[0], 3)
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = NFLPredictor()
    
    # Load and prepare data
    print("Loading data...")
    data = predictor.load_data()
    
    print("Engineering features...")
    data = predictor.engineer_features(data)
    
    print("Preparing features...")
    X, y_spread, y_outcome = predictor.prepare_features(data)
    
    # Train models
    print("Training models...")
    predictor.train_models(X, y_spread, y_outcome)
    
    # Example prediction
    print("\nExample prediction:")
    game_factors = {
        'home_offensive_rating': 85,
        'away_offensive_rating': 70,
        'home_defensive_rating': 80,
        'away_defensive_rating': 75,
        'home_win_pct': 0.7,
        'away_win_pct': 0.4,
        'home_avg_points': 28,
        'away_avg_points': 22,
        'home_avg_points_allowed': 20,
        'away_avg_points_allowed': 26,
        'home_rest_days': 10,
        'away_rest_days': 7,
        'weather_impact': 0,
        'is_playoff': 0
    }
    
    prediction = predictor.predict_game("KC", "NE", game_factors)
    print(f"Prediction: {prediction}")
