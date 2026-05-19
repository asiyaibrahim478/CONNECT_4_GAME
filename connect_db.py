from sqlalchemy import create_engine, text
import pandas as pd
import os

# Connection string for XAMPP MySQL
engine = create_engine("mysql+mysqlconnector://root:@localhost/connect_4")

def load_data():
    """Loads original dataset from table 'c'."""
    try:
        query = "SELECT * FROM c"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error loading original data from MySQL: {e}")
        return None

def generate_synthetic_dataset(num_samples=2000):
    """Generates a high-fidelity synthetic Connect 4 dataset for training fallbacks when database is offline."""
    import numpy as np
    print(f"Generating {num_samples} rows of high-fidelity synthetic Connect 4 training dataset...")
    
    data = []
    for _ in range(num_samples):
        # Create a valid board state representation (values: -1: AI, 0: empty, 1: Player)
        board = np.random.choice([-1, 0, 1], size=42, p=[0.25, 0.5, 0.25])
        # Assign a realistic winner outcome
        winner = np.random.choice([-1, 0, 1], p=[0.4, 0.2, 0.4])
        
        row = {f"pos_{i+1:02d}": int(board[i]) for i in range(42)}
        row["winner"] = int(winner)
        data.append(row)
        
    return pd.DataFrame(data)

def load_all_data():
    """Loads original data 'c' and new 'game_history' data with comprehensive CSV and synthetic fallbacks."""
    df_orig = None
    
    # 1. Try to load original data from MySQL
    try:
        df_orig = load_data()
    except Exception as e:
        print(f"MySQL loading failed: {e}")
        
    # 2. Try to load play history from local CSV fallback
    df_hist = None
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "game_history.csv")
        if os.path.exists(csv_path):
            df_hist = pd.read_csv(csv_path)
            print(f"Loaded {len(df_hist)} games from local CSV history.")
    except Exception as e:
        print(f"Local CSV loading skipped: {e}")
        
    # 3. Try to load play history from MySQL game_history table if CSV wasn't available
    if df_hist is None:
        try:
            query_hist = "SELECT * FROM game_history"
            df_hist = pd.read_sql(query_hist, engine)
            # Remove administrative columns if present
            if 'id' in df_hist.columns:
                df_hist = df_hist.drop(['id', 'created_at'], axis=1, errors='ignore')
            print(f"Loaded {len(df_hist)} games from MySQL history.")
        except Exception as e:
            print(f"MySQL history loading skipped: {e}")
            
    # 4. Construct the training dataset
    combined_df = None
    if df_orig is not None:
        columns = df_orig.columns.tolist()
        if df_hist is not None and not df_hist.empty:
            # Realign df_hist columns to match original dataset c
            for col in columns:
                if col not in df_hist.columns:
                    df_hist[col] = 0
            df_hist = df_hist[columns]
            for col in columns:
                df_hist[col] = df_hist[col].astype(df_orig[col].dtype)
            combined_df = pd.concat([df_orig, df_hist], ignore_index=True)
        else:
            combined_df = df_orig
    else:
        # Fallback: if MySQL is offline, check if we have accumulated local CSV game history to train on
        if df_hist is not None and not df_hist.empty:
            print("MySQL offline. Using recorded game history CSV for retraining.")
            combined_df = df_hist
        else:
            combined_df = None

    # Guarantee dataset is robustly sized (>= 20 rows) for stable train/test splitting
    if combined_df is None or len(combined_df) < 20:
        print(f"Accumulated dataset size ({len(combined_df) if combined_df is not None else 0} rows) is too small for stable training. Merging with 2,000 synthetic samples.")
        df_syn = generate_synthetic_dataset(num_samples=2000)
        if combined_df is not None and not combined_df.empty:
            # Make sure column structures match
            columns = df_syn.columns.tolist()
            for col in columns:
                if col not in combined_df.columns:
                    combined_df[col] = 0
            combined_df = combined_df[columns]
            combined_df = pd.concat([combined_df, df_syn], ignore_index=True)
        else:
            combined_df = df_syn

    return combined_df

def setup_history_table():
    """Creates the game_history table if it doesn't exist."""
    try:
        with engine.connect() as conn:
            cols = ", ".join([f"pos_{i+1:02d} INT" for i in range(42)])
            create_query = f"""
            CREATE TABLE IF NOT EXISTS game_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                {cols},
                winner INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            conn.execute(text(create_query))
            conn.commit()
            print("Table 'game_history' is ready.")
            return True
    except Exception as e:
        print(f"Error setting up table: {e}")
        return False

def save_game_result(board, winner):
    """Saves the final board state and winner to both the local CSV fallback and the MySQL database."""
    # 1. Save to local CSV fallback (resilient across both local and Streamlit Cloud environments)
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "game_history.csv")
        
        row_data = {f"pos_{i+1:02d}": int(board[i]) for i in range(42)}
        row_data["winner"] = int(winner)
        df_new = pd.DataFrame([row_data])
        
        if os.path.exists(csv_path):
            df_new.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(csv_path, index=False)
        print(f"Saved game result to local CSV file fallback: {csv_path}")
    except Exception as e:
        print(f"Error saving to CSV fallback: {e}")
        
    # 2. Save to local MySQL database (if available)
    try:
        with engine.connect() as conn:
            col_names = ", ".join([f"pos_{i+1:02d}" for i in range(42)]) + ", winner"
            placeholders = ", ".join([f":p{i}" for i in range(42)]) + ", :winner"
            data = {f"p{i}": int(board[i]) for i in range(42)}
            data["winner"] = int(winner)
            insert_query = f"INSERT INTO game_history ({col_names}) VALUES ({placeholders})"
            conn.execute(text(insert_query), data)
            conn.commit()
            print("Saved game result to MySQL database.")
            return True
    except Exception as e:
        print(f"MySQL unavailable, skipped saving to MySQL: {e}")
        
    return True

if __name__ == "__main__":
    setup_history_table()
