import os
import subprocess
import time
from connect_db import setup_history_table
from train_models import preprocess_and_train

def run_pipeline():
    print("=== Connect 4 MLOps Pipeline Initializing ===")

    # 1. Database Setup
    print("\n[Step 1/3] Setting up Database...")
    if setup_history_table():
        print("Database initialized successfully.")
    else:
        print("Database initialization failed. Please check your XAMPP connection.")
        return

    # 2. Model Training & Visualization
    # Always run the training to ensure models and plots are up-to-date
    print("\n[Step 2/3] Running training and visualization pipeline...")
    preprocess_and_train()

    # 3. Start Web Server
    print("\n[Step 3/3] Starting Streamlit App...")
    print("The game will be opened in your browser shortly.")
    
    try:
        # Run streamlit app
        subprocess.run(["streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
    except Exception as e:
        print(f"\nError starting server: {e}")

if __name__ == "__main__":
    run_pipeline()
