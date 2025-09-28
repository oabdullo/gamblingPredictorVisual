"""
Test script for NFL Prediction Model
===================================

This script tests the basic functionality of the NFL prediction model.

Author: AI Assistant
Date: 2024
"""

from prediction_interface import NFLPredictionInterface

def test_basic_functionality():
    """Test basic model functionality."""
    print("Testing NFL Prediction Model...")
    print("=" * 40)
    
    try:
        # Initialize interface
        print("1. Initializing interface...")
        interface = NFLPredictionInterface()
        
        # Train models
        print("2. Training models...")
        interface.train_models()
        
        # Test basic prediction
        print("3. Testing basic prediction...")
        prediction = interface.predict_single_game("KC", "NE")
        print(f"   Basic prediction successful: {prediction['home_team']} vs {prediction['away_team']}")
        
        # Test custom prediction
        print("4. Testing custom prediction...")
        custom_pred = interface.create_custom_prediction(
            home_team="KC",
            away_team="NE",
            home_off_rating=85,
            away_off_rating=70,
            home_win_pct=0.7,
            away_win_pct=0.4
        )
        print(f"   Custom prediction successful: {custom_pred['predicted_spread']:+.1f} spread")
        
        # Test multiple predictions
        print("5. Testing multiple predictions...")
        games = [
            {"home_team": "KC", "away_team": "NE"},
            {"home_team": "BUF", "away_team": "MIA"}
        ]
        multiple_preds = interface.predict_multiple_games(games)
        print(f"   Multiple predictions successful: {len(multiple_preds)} games predicted")
        
        print("\n✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        return False

def display_sample_predictions():
    """Display some sample predictions."""
    print("\nSample Predictions:")
    print("-" * 30)
    
    interface = NFLPredictionInterface()
    interface.train_models()
    
    # Sample games
    sample_games = [
        ("KC", "NE", "Chiefs vs Patriots"),
        ("BUF", "MIA", "Bills vs Dolphins"),
        ("BAL", "CIN", "Ravens vs Bengals"),
        ("LAR", "SF", "Rams vs 49ers")
    ]
    
    for home, away, description in sample_games:
        pred = interface.predict_single_game(home, away)
        print(f"\n{description}:")
        print(f"  Predicted Spread: {pred['predicted_spread']:+.1f}")
        print(f"  Predicted Winner: {pred['predicted_winner']}")
        print(f"  Home Win Probability: {pred['home_win_probability']:.1%}")

if __name__ == "__main__":
    # Run tests
    success = test_basic_functionality()
    
    if success:
        # Display sample predictions
        display_sample_predictions()
    else:
        print("Please fix the errors before running sample predictions.")
