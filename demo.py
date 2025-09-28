#!/usr/bin/env python3
"""
NFL Prediction Model - Interactive Demo
======================================

A simple interactive demo for the NFL prediction model.

Author: AI Assistant
Date: 2024
"""

from prediction_interface import NFLPredictionInterface

def main():
    """Run the interactive demo."""
    print("🏈 NFL Game Prediction Model - Interactive Demo")
    print("=" * 50)
    
    # Initialize the model
    print("Initializing model...")
    interface = NFLPredictionInterface()
    interface.train_models()
    print("✅ Model ready!")
    
    while True:
        print("\n" + "="*50)
        print("Choose an option:")
        print("1. Predict a single game")
        print("2. Predict with custom factors")
        print("3. Predict multiple games")
        print("4. Scenario analysis")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            predict_single_game(interface)
        elif choice == "2":
            predict_custom_game(interface)
        elif choice == "3":
            predict_multiple_games(interface)
        elif choice == "4":
            run_scenario_analysis(interface)
        elif choice == "5":
            print("Thanks for using the NFL Prediction Model! 🏈")
            break
        else:
            print("Invalid choice. Please try again.")

def predict_single_game(interface):
    """Predict a single game with default factors."""
    print("\n--- Single Game Prediction ---")
    
    home_team = input("Enter home team (e.g., KC): ").strip().upper()
    away_team = input("Enter away team (e.g., NE): ").strip().upper()
    
    if not home_team or not away_team:
        print("Invalid team names. Please try again.")
        return
    
    try:
        prediction = interface.predict_single_game(home_team, away_team)
        interface.display_prediction(prediction)
    except Exception as e:
        print(f"Error: {e}")

def predict_custom_game(interface):
    """Predict a game with custom factors."""
    print("\n--- Custom Game Prediction ---")
    
    home_team = input("Enter home team (e.g., KC): ").strip().upper()
    away_team = input("Enter away team (e.g., NE): ").strip().upper()
    
    if not home_team or not away_team:
        print("Invalid team names. Please try again.")
        return
    
    print("\nEnter game factors (press Enter for defaults):")
    
    try:
        home_off = input("Home team offensive rating (default 75): ").strip()
        home_off = float(home_off) if home_off else None
        
        away_off = input("Away team offensive rating (default 75): ").strip()
        away_off = float(away_off) if away_off else None
        
        home_def = input("Home team defensive rating (default 75): ").strip()
        home_def = float(home_def) if home_def else None
        
        away_def = input("Away team defensive rating (default 75): ").strip()
        away_def = float(away_def) if away_def else None
        
        home_win_pct = input("Home team win percentage (default 0.5): ").strip()
        home_win_pct = float(home_win_pct) if home_win_pct else None
        
        away_win_pct = input("Away team win percentage (default 0.5): ").strip()
        away_win_pct = float(away_win_pct) if away_win_pct else None
        
        weather_temp = input("Weather temperature (default 50): ").strip()
        weather_temp = float(weather_temp) if weather_temp else None
        
        weather_wind = input("Weather wind speed (default 5): ").strip()
        weather_wind = float(weather_wind) if weather_wind else None
        
        is_playoff = input("Is playoff game? (y/n, default n): ").strip().lower() == 'y'
        
        prediction = interface.create_custom_prediction(
            home_team=home_team,
            away_team=away_team,
            home_off_rating=home_off,
            away_off_rating=away_off,
            home_def_rating=home_def,
            away_def_rating=away_def,
            home_win_pct=home_win_pct,
            away_win_pct=away_win_pct,
            weather_temp=weather_temp,
            weather_wind=weather_wind,
            is_playoff=is_playoff
        )
        
        interface.display_prediction(prediction)
        
    except ValueError:
        print("Invalid input. Please enter numbers for ratings and percentages.")
    except Exception as e:
        print(f"Error: {e}")

def predict_multiple_games(interface):
    """Predict multiple games."""
    print("\n--- Multiple Games Prediction ---")
    
    games = []
    print("Enter games (press Enter with empty home team to finish):")
    
    while True:
        home_team = input("Home team: ").strip().upper()
        if not home_team:
            break
        
        away_team = input("Away team: ").strip().upper()
        if not away_team:
            print("Away team required. Skipping this game.")
            continue
        
        games.append({"home_team": home_team, "away_team": away_team})
        print(f"Added: {home_team} vs {away_team}")
    
    if not games:
        print("No games to predict.")
        return
    
    try:
        predictions = interface.predict_multiple_games(games)
        for i, pred in enumerate(predictions, 1):
            print(f"\nGame {i}:")
            interface.display_prediction(pred)
    except Exception as e:
        print(f"Error: {e}")

def run_scenario_analysis(interface):
    """Run scenario analysis for a game."""
    print("\n--- Scenario Analysis ---")
    
    home_team = input("Enter home team (e.g., KC): ").strip().upper()
    away_team = input("Enter away team (e.g., NE): ").strip().upper()
    
    if not home_team or not away_team:
        print("Invalid team names. Please try again.")
        return
    
    scenarios = [
        ("Neutral Conditions", {}),
        ("Home Team Advantage", {
            "home_off_rating": 85, "away_off_rating": 70,
            "home_def_rating": 80, "away_def_rating": 75,
            "home_win_pct": 0.7, "away_win_pct": 0.4
        }),
        ("Cold Weather", {
            "weather_temp": 15, "weather_wind": 25
        }),
        ("Playoff Game", {
            "is_playoff": True, "home_off_rating": 90, "away_off_rating": 90
        })
    ]
    
    print(f"\nScenario Analysis: {home_team} vs {away_team}")
    print("-" * 40)
    
    for scenario_name, factors in scenarios:
        try:
            prediction = interface.create_custom_prediction(
                home_team, away_team, **factors
            )
            print(f"\n{scenario_name}:")
            print(f"  Spread: {prediction['predicted_spread']:+.1f}")
            print(f"  Winner: {prediction['predicted_winner']}")
            print(f"  Home Win Prob: {prediction['home_win_probability']:.1%}")
        except Exception as e:
            print(f"Error in {scenario_name}: {e}")

if __name__ == "__main__":
    main()
