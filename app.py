from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os
import math
import subprocess
import sys
from connect_db import save_game_result

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# Use absolute paths to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Logistic_Regression.pkl')
model = None

def load_ml_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}. Using fallback Minimax.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_ml_model()

# --- Helper Functions for Minimax ---
def check_win(board, player):
    for r in range(6):
        for c in range(4):
            if all(board[r*7 + c + i] == player for i in range(4)): return True
    for r in range(3):
        for c in range(7):
            if all(board[(r+i)*7 + c] == player for i in range(4)): return True
    for r in range(3):
        for c in range(4):
            if all(board[(r+i)*7 + c + i] == player for i in range(4)): return True
    for r in range(3, 6):
        for c in range(4):
            if all(board[(r-i)*7 + c + i] == player for i in range(4)): return True
    return False

def get_valid_columns(board):
    return [c for c in range(7) if board[c] == 0]

def drop_piece(board, col, player):
    new_board = list(board)
    for r in range(5, -1, -1):
        if new_board[r*7 + col] == 0:
            new_board[r*7 + col] = player
            return new_board
    return None

def evaluate_board(board, model):
    if check_win(board, -1): return 100000
    if check_win(board, 1): return -100000
    if not model: return 0
    try:
        input_data = np.array(board).reshape(1, -1)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            if -1 in classes:
                return probs[classes.index(-1)] * 100
        return 0
    except:
        return 0

def minimax(board, depth, alpha, beta, maximizingPlayer, model):
    valid_cols = get_valid_columns(board)
    is_terminal = len(valid_cols) == 0 or check_win(board, 1) or check_win(board, -1)
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if check_win(board, -1): return 1000000
            elif check_win(board, 1): return -1000000
            else: return 0
        else:
            return evaluate_board(board, model)
            
    if maximizingPlayer:
        value = -math.inf
        ordered_cols = [3, 2, 4, 1, 5, 0, 6]
        for col in [c for c in ordered_cols if c in valid_cols]:
            new_board = drop_piece(board, col, -1)
            value = max(value, minimax(new_board, depth-1, alpha, beta, False, model))
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value
    else:
        value = math.inf
        ordered_cols = [3, 2, 4, 1, 5, 0, 6]
        for col in [c for c in ordered_cols if c in valid_cols]:
            new_board = drop_piece(board, col, 1)
            value = min(value, minimax(new_board, depth-1, alpha, beta, True, model))
            beta = min(beta, value)
            if alpha >= beta: break
        return value

# --- API Endpoints ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    data = request.json
    board = data.get('board')
    level = data.get('level', 'medium')
    depth_map = {'easy': 1, 'medium': 3, 'hard': 5}
    depth = depth_map.get(level, 3)

    if not board or len(board) != 42:
        return jsonify({'error': 'Invalid board state'}), 400

    valid_cols = get_valid_columns(board)
    if not valid_cols:
        return jsonify({'error': 'No moves left'}), 400

    for col in valid_cols:
        temp = drop_piece(board, col, -1)
        if check_win(temp, -1): return jsonify({'column': col})
    for col in valid_cols:
        temp = drop_piece(board, col, 1)
        if check_win(temp, 1): return jsonify({'column': col})

    best_score = -math.inf
    best_move = valid_cols[0]
    for col in valid_cols:
        new_board = drop_piece(board, col, -1)
        score = minimax(new_board, depth, -math.inf, math.inf, False, model)
        if score > best_score:
            best_score = score
            best_move = col
    return jsonify({'column': int(best_move)})

@app.route('/api/save_game', methods=['POST'])
def save_game():
    data = request.json
    board = data.get('board')
    winner = data.get('winner')
    if board and len(board) == 42 and winner is not None:
        save_game_result(board, winner)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 400

@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        script_path = os.path.join(BASE_DIR, "train_models.py")
        print(f"Starting retraining process: {sys.executable} {script_path}", flush=True)
        
        # Run the training script in a subprocess with the correct CWD
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, 
            text=True,
            cwd=BASE_DIR
        )
        
        # Log stdout/stderr for debugging
        if result.stdout: print(f"Retrain Output: {result.stdout}", flush=True)
        if result.stderr: print(f"Retrain Error Output: {result.stderr}", flush=True)

        if result.returncode == 0:
            print("Retraining completed successfully via subprocess. Reloading model...", flush=True)
            load_ml_model()
            return jsonify({'status': 'success', 'message': 'AI Retrained successfully!'})
        else:
            print(f"Retraining script failed with return code {result.returncode}", flush=True)
            # Fallback to direct function call
            print("Attempting fallback retraining via direct import...", flush=True)
            try:
                from train_models import preprocess_and_train
                preprocess_and_train()
                load_ml_model()
                return jsonify({'status': 'success', 'message': 'AI Retrained via fallback method!'})
            except Exception as fe:
                print(f"Fallback retraining also failed: {fe}")
                return jsonify({'status': 'error', 'message': f"Script fail: {result.stderr}. Fallback fail: {str(fe)}"}), 500
    except Exception as e:
        print(f"Retraining endpoint error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
