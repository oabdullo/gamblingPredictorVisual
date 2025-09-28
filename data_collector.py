"""
NFL Data Collection Module
=========================

This module handles data collection from various NFL data sources.
In production, this would connect to real APIs and data sources.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import os
import streamlit as st

class NFLDataCollector:
    """
    Collects NFL game data from various sources.
    """
    
    def __init__(self):
        self.base_url = "https://api.sportsdata.io/v3/nfl"
        self.api_key = self._get_api_key()
        
    def _get_api_key(self):
        """
        Get API key from GitHub secrets or environment variables.
        Priority: GitHub Secrets > Environment Variable > None
        """
        try:
            # Try to get from GitHub Secrets (when deployed on Streamlit Cloud)
            if hasattr(st, 'secrets') and 'nfl_api_key' in st.secrets:
                return st.secrets['nfl_api_key']
        except:
            pass
        
        # Try to get from environment variable
        api_key = os.getenv('NFL_API_KEY')
        if api_key:
            return api_key
        
        # Try alternative environment variable names
        api_key = os.getenv('SPORTSDATA_API_KEY')
        if api_key:
            return api_key
        
        # Return None if no API key found
        return None
    
    def set_api_key(self, api_key):
        """Set the API key for data collection."""
        self.api_key = api_key
    
    def get_team_stats(self, season=2023):
        """
        Get team statistics for a given season.
        
        Args:
            season (int): NFL season year
            
        Returns:
            pd.DataFrame: Team statistics
        """
        print(f"Collecting team stats for {season} season...")
        
        # If API key is available, try to get real data
        if self.api_key:
            try:
                return self._get_real_team_stats(season)
            except Exception as e:
                print(f"API call failed, using sample data: {e}")
        
        # Fallback to sample data
        return self._get_sample_team_stats(season)
    
    def _get_real_team_stats(self, season):
        """
        Get real team stats from API.
        
        Args:
            season (int): NFL season year
            
        Returns:
            pd.DataFrame: Real team statistics
        """
        url = f"{self.base_url}/scores/json/TeamSeasonStats/{season}"
        headers = {'Ocp-Apim-Subscription-Key': self.api_key}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Process the API response into our format
        team_stats = []
        for team in data:
            team_stats.append({
                'team': team.get('Team', ''),
                'season': season,
                'wins': team.get('Wins', 0),
                'losses': team.get('Losses', 0),
                'points_for': team.get('PointsFor', 0),
                'points_against': team.get('PointsAgainst', 0),
                'offensive_rating': team.get('OffensiveRanking', 75),
                'defensive_rating': team.get('DefensiveRanking', 75),
                'avg_points_per_game': team.get('PointsFor', 0) / max(team.get('Games', 1), 1),
                'avg_points_allowed_per_game': team.get('PointsAgainst', 0) / max(team.get('Games', 1), 1),
                'turnover_differential': team.get('Takeaways', 0) - team.get('Giveaways', 0),
                'home_record': f"{team.get('HomeWins', 0)}-{team.get('HomeLosses', 0)}",
                'away_record': f"{team.get('AwayWins', 0)}-{team.get('AwayLosses', 0)}"
            })
        
        return pd.DataFrame(team_stats)
    
    def _get_sample_team_stats(self, season):
        """
        Get sample team stats for demonstration.
        
        Args:
            season (int): NFL season year
            
        Returns:
            pd.DataFrame: Sample team statistics
        """
        teams = [
            'NE', 'KC', 'BUF', 'MIA', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT',
            'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'LAC', 'LV', 'DAL', 'PHI',
            'NYG', 'WAS', 'CHI', 'DET', 'GB', 'MIN', 'ATL', 'CAR', 'NO',
            'TB', 'ARI', 'LAR', 'SF', 'SEA'
        ]
        
        team_stats = []
        for team in teams:
            stats = {
                'team': team,
                'season': season,
                'wins': np.random.randint(4, 13),
                'losses': np.random.randint(4, 13),
                'points_for': np.random.randint(250, 450),
                'points_against': np.random.randint(250, 450),
                'offensive_rating': np.random.normal(75, 15),
                'defensive_rating': np.random.normal(75, 15),
                'avg_points_per_game': np.random.normal(25, 5),
                'avg_points_allowed_per_game': np.random.normal(25, 5),
                'turnover_differential': np.random.normal(0, 5),
                'home_record': f"{np.random.randint(2, 8)}-{np.random.randint(2, 8)}",
                'away_record': f"{np.random.randint(2, 8)}-{np.random.randint(2, 8)}"
            }
            team_stats.append(stats)
        
        return pd.DataFrame(team_stats)
    
    def get_game_data(self, season=2023, week=None):
        """
        Get game data for a given season and week.
        
        Args:
            season (int): NFL season year
            week (int): Specific week (None for all weeks)
            
        Returns:
            pd.DataFrame: Game data
        """
        print(f"Collecting game data for {season} season...")
        
        # Sample game data
        games = []
        teams = ['NE', 'KC', 'BUF', 'MIA', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT']
        
        for week_num in range(1, 18):  # Regular season weeks
            if week and week_num != week:
                continue
                
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    home_team = teams[i]
                    away_team = teams[i + 1]
                    
                    # Randomly determine home/away
                    if np.random.random() > 0.5:
                        home_team, away_team = away_team, home_team
                    
                    game = {
                        'season': season,
                        'week': week_num,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': np.random.poisson(25),
                        'away_score': np.random.poisson(25),
                        'spread': np.random.normal(0, 7),
                        'total': np.random.normal(45, 10),
                        'weather_temp': np.random.normal(50, 20),
                        'weather_wind': np.random.exponential(5),
                        'is_playoff': 0,
                        'date': f"{season}-{week_num:02d}-{np.random.randint(1, 8):02d}"
                    }
                    games.append(game)
        
        return pd.DataFrame(games)
    
    def get_historical_data(self, start_season=2020, end_season=2023):
        """
        Get historical data for multiple seasons.
        
        Args:
            start_season (int): Starting season year
            end_season (int): Ending season year
            
        Returns:
            pd.DataFrame: Combined historical data
        """
        all_data = []
        
        for season in range(start_season, end_season + 1):
            print(f"Collecting data for {season} season...")
            season_data = self.get_game_data(season)
            all_data.append(season_data)
            time.sleep(0.1)  # Rate limiting
        
        return pd.concat(all_data, ignore_index=True)
    
    def get_upcoming_games(self, season=2024, week=None):
        """
        Get upcoming games for prediction.
        
        Args:
            season (int): NFL season year
            week (int): Specific week
            
        Returns:
            pd.DataFrame: Upcoming games data
        """
        print(f"Collecting upcoming games for {season} season...")
        
        # Sample upcoming games
        upcoming_games = []
        teams = ['NE', 'KC', 'BUF', 'MIA', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT']
        
        for week_num in range(1, 18):
            if week and week_num != week:
                continue
                
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    home_team = teams[i]
                    away_team = teams[i + 1]
                    
                    if np.random.random() > 0.5:
                        home_team, away_team = away_team, home_team
                    
                    game = {
                        'season': season,
                        'week': week_num,
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_date': f"{season}-{week_num:02d}-{np.random.randint(1, 8):02d}",
                        'is_playoff': 0
                    }
                    upcoming_games.append(game)
        
        return pd.DataFrame(upcoming_games)
    
    def save_data(self, data, filename):
        """
        Save data to CSV file.
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        filepath = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename):
        """
        Load data from CSV file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            pd.DataFrame: Loaded data
        """
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"File {filepath} not found")
            return None

# Example usage
if __name__ == "__main__":
    collector = NFLDataCollector()
    
    # Collect historical data
    print("Collecting historical data...")
    historical_data = collector.get_historical_data(2020, 2023)
    collector.save_data(historical_data, 'historical_games.csv')
    
    # Collect team stats
    print("Collecting team statistics...")
    team_stats = collector.get_team_stats(2023)
    collector.save_data(team_stats, 'team_stats_2023.csv')
    
    # Collect upcoming games
    print("Collecting upcoming games...")
    upcoming = collector.get_upcoming_games(2024, 1)
    collector.save_data(upcoming, 'upcoming_games.csv')
    
    print("Data collection complete!")
