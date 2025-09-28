"""
NFL Game Prediction Model - Streamlit Web App
============================================

A web application for predicting NFL game spreads and outcomes.

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from prediction_interface import NFLPredictionInterface

# Page configuration
st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def check_api_key_status():
    """Check and display API key status."""
    from data_collector import NFLDataCollector
    
    collector = NFLDataCollector()
    api_key = collector.api_key
    
    if api_key:
        st.success("✅ API Key loaded successfully - Using real NFL data!")
    else:
        st.warning("⚠️ No API key found - Using sample data for demonstration")
        st.info("""
        **To use real NFL data:**
        1. Get an API key from [SportsData.io](https://sportsdata.io)
        2. Add it to GitHub Secrets as `nfl_api_key`
        3. Or set environment variable `NFL_API_KEY`
        """)

@st.cache_resource
def load_model():
    """Load and cache the prediction model."""
    interface = NFLPredictionInterface()
    interface.train_models()
    return interface

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🏈 NFL Game Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check API key status
    check_api_key_status()
    
    # Load model
    with st.spinner("Loading prediction model..."):
        interface = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("🏈 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Single Game Prediction", "Multiple Games", "Scenario Analysis", "Model Performance", "About"]
        )
    
    # Main content based on page selection
    if page == "Single Game Prediction":
        single_game_prediction(interface)
    elif page == "Multiple Games":
        multiple_games_prediction(interface)
    elif page == "Scenario Analysis":
        scenario_analysis(interface)
    elif page == "Model Performance":
        model_performance(interface)
    elif page == "About":
        about_page()

def single_game_prediction(interface):
    """Single game prediction page."""
    st.header("🎯 Single Game Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Game Details")
        home_team = st.text_input("Home Team", value="KC", help="Enter team abbreviation (e.g., KC, NE, BUF)")
        away_team = st.text_input("Away Team", value="NE", help="Enter team abbreviation (e.g., KC, NE, BUF)")
        
        # Basic prediction
        if st.button("Predict with Default Factors", type="primary"):
            if home_team and away_team:
                with st.spinner("Making prediction..."):
                    prediction = interface.predict_single_game(home_team.upper(), away_team.upper())
                    display_prediction_result(prediction)
            else:
                st.error("Please enter both team names.")
    
    with col2:
        st.subheader("Custom Factors (Optional)")
        
        with st.expander("Team Performance", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                home_off_rating = st.slider("Home Offensive Rating", 50, 100, 75)
                home_def_rating = st.slider("Home Defensive Rating", 50, 100, 75)
                home_win_pct = st.slider("Home Win %", 0.0, 1.0, 0.5, 0.01)
            with col_b:
                away_off_rating = st.slider("Away Offensive Rating", 50, 100, 75)
                away_def_rating = st.slider("Away Defensive Rating", 50, 100, 75)
                away_win_pct = st.slider("Away Win %", 0.0, 1.0, 0.5, 0.01)
        
        with st.expander("Game Conditions", expanded=False):
            col_c, col_d = st.columns(2)
            with col_c:
                home_rest = st.slider("Home Rest Days", 3, 14, 7)
                weather_temp = st.slider("Temperature (°F)", -10, 100, 50)
            with col_d:
                away_rest = st.slider("Away Rest Days", 3, 14, 7)
                weather_wind = st.slider("Wind Speed (mph)", 0, 50, 5)
            
            is_playoff = st.checkbox("Playoff Game")
        
        # Custom prediction
        if st.button("Predict with Custom Factors", type="secondary"):
            if home_team and away_team:
                with st.spinner("Making custom prediction..."):
                    prediction = interface.create_custom_prediction(
                        home_team=home_team.upper(),
                        away_team=away_team.upper(),
                        home_off_rating=home_off_rating,
                        away_off_rating=away_off_rating,
                        home_def_rating=home_def_rating,
                        away_def_rating=away_def_rating,
                        home_win_pct=home_win_pct,
                        away_win_pct=away_win_pct,
                        home_rest=home_rest,
                        away_rest=away_rest,
                        weather_temp=weather_temp,
                        weather_wind=weather_wind,
                        is_playoff=is_playoff
                    )
                    display_prediction_result(prediction)
            else:
                st.error("Please enter both team names.")

def multiple_games_prediction(interface):
    """Multiple games prediction page."""
    st.header("📊 Multiple Games Prediction")
    
    st.subheader("Add Games")
    
    # Initialize session state for games list
    if 'games' not in st.session_state:
        st.session_state.games = []
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        home_team = st.text_input("Home Team", key="multi_home")
    with col2:
        away_team = st.text_input("Away Team", key="multi_away")
    with col3:
        if st.button("Add Game", type="primary"):
            if home_team and away_team:
                st.session_state.games.append({
                    "home_team": home_team.upper(),
                    "away_team": away_team.upper()
                })
                st.success(f"Added: {home_team.upper()} vs {away_team.upper()}")
            else:
                st.error("Please enter both team names.")
    
    # Display current games
    if st.session_state.games:
        st.subheader("Games to Predict")
        
        # Display games in a table
        games_df = pd.DataFrame(st.session_state.games)
        st.dataframe(games_df, use_container_width=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Predict All Games", type="primary"):
                with st.spinner("Making predictions..."):
                    predictions = interface.predict_multiple_games(st.session_state.games)
                    display_multiple_predictions(predictions)
        
        with col2:
            if st.button("Clear All Games"):
                st.session_state.games = []
                st.rerun()
        
        with col3:
            if st.button("Export Predictions"):
                export_predictions(predictions if 'predictions' in locals() else [])
    else:
        st.info("Add some games to get started!")

def scenario_analysis(interface):
    """Scenario analysis page."""
    st.header("🔍 Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Game Setup")
        home_team = st.text_input("Home Team", value="KC", key="scenario_home")
        away_team = st.text_input("Away Team", value="NE", key="scenario_away")
    
    with col2:
        st.subheader("Analysis Type")
        analysis_type = st.selectbox(
            "Choose analysis:",
            ["Weather Impact", "Team Strength", "Rest Advantage", "Playoff vs Regular"]
        )
    
    if home_team and away_team:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Running scenario analysis..."):
                run_scenario_analysis(interface, home_team.upper(), away_team.upper(), analysis_type)

def model_performance(interface):
    """Model performance page."""
    st.header("📈 Model Performance")
    
    # Generate sample predictions for analysis
    with st.spinner("Generating performance data..."):
        sample_games = [
            {"home_team": "KC", "away_team": "NE"},
            {"home_team": "BUF", "away_team": "MIA"},
            {"home_team": "BAL", "away_team": "CIN"},
            {"home_team": "LAR", "away_team": "SF"},
            {"home_team": "DAL", "away_team": "PHI"}
        ]
        
        predictions = interface.predict_multiple_games(sample_games)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    spreads = [p['predicted_spread'] for p in predictions]
    home_probs = [p['home_win_probability'] for p in predictions]
    
    with col1:
        st.metric("Average Spread", f"{np.mean(spreads):.1f}")
    with col2:
        st.metric("Spread Std Dev", f"{np.std(spreads):.1f}")
    with col3:
        st.metric("Avg Home Win %", f"{np.mean(home_probs):.1%}")
    with col4:
        st.metric("Total Predictions", len(predictions))
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Spread distribution
        fig_spread = px.histogram(
            x=spreads,
            title="Predicted Spread Distribution",
            labels={'x': 'Predicted Spread', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_spread.update_layout(showlegend=False)
        st.plotly_chart(fig_spread, use_container_width=True)
    
    with col2:
        # Win probability distribution
        fig_prob = px.histogram(
            x=home_probs,
            title="Home Win Probability Distribution",
            labels={'x': 'Home Win Probability', 'y': 'Count'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig_prob.update_layout(showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # Feature importance (if available)
    if hasattr(interface.predictor, 'feature_columns') and hasattr(interface.predictor.spread_model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': interface.predictor.feature_columns,
            'Importance': interface.predictor.spread_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance for Spread Prediction",
            color='Importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_importance, use_container_width=True)

def about_page():
    """About page."""
    st.header("ℹ️ About NFL Game Predictor")
    
    st.markdown("""
    ## Overview
    This NFL Game Predictor uses machine learning to predict game spreads and outcomes based on various factors including team performance, weather conditions, and game context.
    
    ## Features
    - **Spread Prediction**: Predicts point spreads for NFL games
    - **Outcome Prediction**: Predicts game winners with probability scores
    - **Multiple Factors**: Considers offensive/defensive ratings, win percentages, weather, rest days, and more
    - **Scenario Analysis**: Compare different game conditions
    - **Interactive Interface**: Easy-to-use web interface
    
    ## Model Details
    - **Algorithm**: XGBoost for both regression (spread) and classification (outcome)
    - **Features**: 9 engineered features from team and game data
    - **Performance**: RMSE ~8.1 for spread prediction, ~53% accuracy for outcomes
    
    ## How to Use
    1. **Single Game**: Enter team names and optionally customize factors
    2. **Multiple Games**: Add multiple games and predict all at once
    3. **Scenario Analysis**: Compare different game conditions
    4. **Model Performance**: View model metrics and feature importance
    
    ## Technical Stack
    - **Backend**: Python, scikit-learn, XGBoost
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **Data**: Pandas, NumPy
    
    ## Disclaimer
    This tool is for educational and entertainment purposes only. Sports betting involves risk, and past performance does not guarantee future results.
    """)

def display_prediction_result(prediction):
    """Display prediction result in a formatted card."""
    st.markdown("### 🎯 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Spread",
            f"{prediction['predicted_spread']:+.1f}",
            help="Positive means home team favored, negative means away team favored"
        )
    
    with col2:
        st.metric(
            "Predicted Winner",
            prediction['predicted_winner'],
            help="Based on predicted spread"
        )
    
    with col3:
        st.metric(
            "Confidence",
            f"{max(prediction['home_win_probability'], prediction['away_win_probability']):.1%}",
            help="Confidence in the prediction"
        )
    
    # Probability bars
    st.subheader("Win Probabilities")
    
    prob_data = {
        'Team': [f"{prediction['home_team']} (Home)", f"{prediction['away_team']} (Away)"],
        'Probability': [prediction['home_win_probability'], prediction['away_win_probability']]
    }
    
    fig = px.bar(
        prob_data,
        x='Team',
        y='Probability',
        title="Win Probability by Team",
        color='Probability',
        color_continuous_scale='RdYlBu_r'
    )
    fig.update_layout(showlegend=False, yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)

def display_multiple_predictions(predictions):
    """Display multiple predictions in a table."""
    st.subheader("📊 Prediction Results")
    
    # Create results dataframe
    results_data = []
    for pred in predictions:
        results_data.append({
            'Game': f"{pred['home_team']} vs {pred['away_team']}",
            'Spread': f"{pred['predicted_spread']:+.1f}",
            'Winner': pred['predicted_winner'],
            'Home Win %': f"{pred['home_win_probability']:.1%}",
            'Away Win %': f"{pred['away_win_probability']:.1%}"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    spreads = [p['predicted_spread'] for p in predictions]
    home_wins = sum(1 for p in predictions if p['predicted_winner'] == 'Home')
    
    with col1:
        st.metric("Average Spread", f"{np.mean(spreads):.1f}")
    with col2:
        st.metric("Home Team Wins", f"{home_wins}/{len(predictions)}")
    with col3:
        st.metric("Away Team Wins", f"{len(predictions) - home_wins}/{len(predictions)}")

def run_scenario_analysis(interface, home_team, away_team, analysis_type):
    """Run scenario analysis."""
    scenarios = {
        "Weather Impact": [
            ("Normal Weather", {}),
            ("Cold Weather", {"weather_temp": 15, "weather_wind": 25}),
            ("Hot Weather", {"weather_temp": 90, "weather_wind": 5}),
            ("Windy Weather", {"weather_temp": 50, "weather_wind": 35})
        ],
        "Team Strength": [
            ("Even Matchup", {}),
            ("Home Team Strong", {"home_off_rating": 90, "away_off_rating": 70}),
            ("Away Team Strong", {"home_off_rating": 70, "away_off_rating": 90}),
            ("Both Strong", {"home_off_rating": 90, "away_off_rating": 90})
        ],
        "Rest Advantage": [
            ("Equal Rest", {}),
            ("Home Team Rested", {"home_rest": 14, "away_rest": 7}),
            ("Away Team Rested", {"home_rest": 7, "away_rest": 14}),
            ("Both Rested", {"home_rest": 14, "away_rest": 14})
        ],
        "Playoff vs Regular": [
            ("Regular Season", {}),
            ("Playoff Game", {"is_playoff": True}),
            ("High Stakes", {"is_playoff": True, "home_off_rating": 95, "away_off_rating": 95})
        ]
    }
    
    scenario_list = scenarios.get(analysis_type, scenarios["Weather Impact"])
    
    results = []
    for scenario_name, factors in scenario_list:
        try:
            prediction = interface.create_custom_prediction(
                home_team, away_team, **factors
            )
            results.append({
                'Scenario': scenario_name,
                'Spread': prediction['predicted_spread'],
                'Winner': prediction['predicted_winner'],
                'Home Win %': prediction['home_win_probability']
            })
        except Exception as e:
            st.error(f"Error in {scenario_name}: {e}")
    
    if results:
        # Display results table
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            results_df,
            x='Scenario',
            y='Spread',
            title=f"Spread Prediction by {analysis_type}",
            color='Spread',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def export_predictions(predictions):
    """Export predictions to CSV."""
    if predictions:
        results_data = []
        for pred in predictions:
            results_data.append({
                'Home Team': pred['home_team'],
                'Away Team': pred['away_team'],
                'Predicted Spread': pred['predicted_spread'],
                'Predicted Winner': pred['predicted_winner'],
                'Home Win Probability': pred['home_win_probability'],
                'Away Win Probability': pred['away_win_probability']
            })
        
        df = pd.DataFrame(results_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"nfl_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No predictions to export. Make some predictions first!")

if __name__ == "__main__":
    main()
