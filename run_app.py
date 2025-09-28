#!/usr/bin/env python3
"""
Run the NFL Prediction Streamlit App
===================================

This script provides an easy way to run the Streamlit app with proper configuration.

Author: AI Assistant
Date: 2024
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app."""
    print("🏈 Starting NFL Game Predictor...")
    print("=" * 40)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this from the project directory.")
        sys.exit(1)
    
    print("🚀 Launching web application...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down NFL Game Predictor. Goodbye!")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
