from sqlalchemy import create_engine, text
import pandas as pd

# Connection string for XAMPP MySQL
engine = create_engine("mysql+mysqlconnector://root:@localhost/connect_4")

def load_data():
    """Loads original dataset from table 'c'."""
    try:
        query = "SELECT * FROM c"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error loading original data: {e}")
        return None

def load_all_data():
    """Loads both original data 'c' and new 'game_history' data."""
    try:
        df_orig = load_data()
        if df_orig is None: return None
        
        # Get column names from original data to ensure consistency
        columns = df_orig.columns.tolist()
        
        # Load history
        df_hist = None
        try:
            # We select specifically the columns we need to match original data
            cols_str = ", ".join(columns)
            query_hist = f"SELECT {cols_str} FROM game_history"
            df_hist = pd.read_sql(query_hist, engine)
            print(f"Found {len(df_hist)} new games in history.")
        except Exception as e:
            print(f"No game history found or table empty: {e}")

        if df_hist is not None and not df_hist.empty:
            # Ensure types match before concat
            for col in columns:
                df_hist[col] = df_hist[col].astype(df_orig[col].dtype)
            
            combined_df = pd.concat([df_orig, df_hist], ignore_index=True)
            return combined_df
        
        return df_orig
    except Exception as e:
        print(f"Error combining data: {e}")
        return None

def setup_history_table():
    """Creates the game_history table if it doesn't exist."""
    try:
        with engine.connect() as conn:
            # Create 42 position columns (pos_01 to pos_42)
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
    """Saves the final board state and winner to the database."""
    try:
        with engine.connect() as conn:
            # Map board indices to pos_01...pos_42
            col_names = ", ".join([f"pos_{i+1:02d}" for i in range(42)]) + ", winner"
            placeholders = ", ".join([f":p{i}" for i in range(42)]) + ", :winner"
            data = {f"p{i}": int(board[i]) for i in range(42)} # Ensure int
            data["winner"] = int(winner)
            insert_query = f"INSERT INTO game_history ({col_names}) VALUES ({placeholders})"
            conn.execute(text(insert_query), data)
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving game result: {e}")
        return False

if __name__ == "__main__":
    setup_history_table()
