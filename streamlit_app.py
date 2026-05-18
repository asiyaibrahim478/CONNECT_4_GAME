import streamlit as st
import numpy as np
import joblib
import os
import math
import subprocess
import sys
from connect_db import save_game_result

st.set_page_config(page_title="AI Connect 4", layout="wide")

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Logistic_Regression.pkl')

@st.cache_resource
def load_ml_model():
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

model = load_ml_model()

# --- Game Engine ---
ROWS = 6
COLS = 7

def init_board():
    return [0] * (ROWS * COLS)

def check_win(board, player):
    # horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r*COLS + c + i] == player for i in range(4)): return True
    # vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[(r+i)*COLS + c] == player for i in range(4)): return True
    # diagonal positive
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[(r+i)*COLS + c + i] == player for i in range(4)): return True
    # diagonal negative
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[(r-i)*COLS + c + i] == player for i in range(4)): return True
    return False

def get_valid_columns(board):
    return [c for c in range(COLS) if board[c] == 0]

def drop_piece(board, col, player):
    new_board = list(board)
    for r in range(ROWS - 1, -1, -1):
        if new_board[r*COLS + col] == 0:
            new_board[r*COLS + col] = player
            return new_board
    return None

def evaluate_board(board, model):
    if check_win(board, -1): return 100000
    if check_win(board, 1): return -100000
    if model is None: return 0
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
            if new_board:
                value = max(value, minimax(new_board, depth-1, alpha, beta, False, model))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return value
    else:
        value = math.inf
        ordered_cols = [3, 2, 4, 1, 5, 0, 6]
        for col in [c for c in ordered_cols if c in valid_cols]:
            new_board = drop_piece(board, col, 1)
            if new_board:
                value = min(value, minimax(new_board, depth-1, alpha, beta, True, model))
                beta = min(beta, value)
                if alpha >= beta: break
        return value

# --- State Management ---
if 'board' not in st.session_state:
    st.session_state.board = init_board()
    st.session_state.current_player = 1 # 1 for User, -1 for AI
    st.session_state.game_active = True
    st.session_state.winner = 0

# --- UI Setup ---
st.markdown("""
<style>
.board {
    display: grid;
    grid-template-columns: repeat(7, 60px);
    grid-gap: 10px;
    background: #1e40af;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    margin: 0 auto;
    width: fit-content;
}
.cell {
    width: 60px;
    height: 60px;
    background: #0f172a;
    border-radius: 50%;
    box-shadow: inset 0 5px 10px rgba(0,0,0,0.8);
}
.cell.red {
    background: radial-gradient(circle at 30% 30%, #ff4b4b, #b91c1c);
    box-shadow: 0 4px 8px rgba(0,0,0,0.5), inset 0 -2px 5px rgba(0,0,0,0.4);
}
.cell.yellow {
    background: radial-gradient(circle at 30% 30%, #fcd34d, #b45309);
    box-shadow: 0 4px 8px rgba(0,0,0,0.5), inset 0 -2px 5px rgba(0,0,0,0.4);
}
.stButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    level = st.radio("Difficulty", ['Easy', 'Medium', 'Hard'], index=1)
    
    st.header("Actions")
    if st.button("Reset Game"):
        st.session_state.board = init_board()
        st.session_state.current_player = 1
        st.session_state.game_active = True
        st.session_state.winner = 0
        st.rerun()
        
    if st.button("Retrain AI"):
        with st.spinner("Retraining..."):
            script_path = os.path.join(BASE_DIR, "train_models.py")
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=BASE_DIR)
            if result.returncode == 0:
                st.success("AI Retrained successfully!")
                load_ml_model.clear() # clear cache to reload
            else:
                try:
                    from train_models import preprocess_and_train
                    preprocess_and_train()
                    st.success("AI Retrained successfully (fallback)!")
                    load_ml_model.clear()
                except Exception as e:
                    st.error(f"Retraining failed: {e}")

st.title("AI Connect 4")

if not st.session_state.game_active:
    if st.session_state.winner == 1:
        st.success("Congratulations! You Win! 🎉")
        st.balloons()
    elif st.session_state.winner == -1:
        st.error("Sorry you lost, play best for the next time. 🤖")
    else:
        st.info("It's a Draw! 🤝")
else:
    if st.session_state.current_player == 1:
        st.markdown("### Your Turn (Red)")
    else:
        st.markdown("### AI is thinking... (Yellow)")

def make_move(col):
    if st.session_state.game_active and st.session_state.current_player == 1:
        new_b = drop_piece(st.session_state.board, col, 1)
        if new_b:
            st.session_state.board = new_b
            if check_win(st.session_state.board, 1):
                st.session_state.game_active = False
                st.session_state.winner = 1
                save_game_result(st.session_state.board, 1)
            elif len(get_valid_columns(st.session_state.board)) == 0:
                st.session_state.game_active = False
                st.session_state.winner = 0
                save_game_result(st.session_state.board, 0)
            else:
                st.session_state.current_player = -1
            # Rerun will happen automatically when button is clicked

# AI Move Logic
if st.session_state.game_active and st.session_state.current_player == -1:
    depth_map = {'Easy': 1, 'Medium': 3, 'Hard': 5}
    depth = depth_map.get(level, 3)
    
    valid_cols = get_valid_columns(st.session_state.board)
    best_move = None
    
    # Check immediate wins/losses
    for col in valid_cols:
        temp = drop_piece(st.session_state.board, col, -1)
        if temp and check_win(temp, -1):
            best_move = col
            break
    if best_move is None:
        for col in valid_cols:
            temp = drop_piece(st.session_state.board, col, 1)
            if temp and check_win(temp, 1):
                best_move = col
                break
                
    if best_move is None:
        best_score = -math.inf
        best_move = valid_cols[0]
        for col in valid_cols:
            new_board = drop_piece(st.session_state.board, col, -1)
            score = minimax(new_board, depth, -math.inf, math.inf, False, model)
            if score > best_score:
                best_score = score
                best_move = col
                
    new_b = drop_piece(st.session_state.board, best_move, -1)
    st.session_state.board = new_b
    if check_win(st.session_state.board, -1):
        st.session_state.game_active = False
        st.session_state.winner = -1
        save_game_result(st.session_state.board, -1)
    elif len(get_valid_columns(st.session_state.board)) == 0:
        st.session_state.game_active = False
        st.session_state.winner = 0
        save_game_result(st.session_state.board, 0)
    else:
        st.session_state.current_player = 1
    st.rerun()

# --- Draw Board ---
cols = st.columns([1, 1, 1, 1, 1, 1, 1, 4]) # Add empty columns for spacing
for i in range(7):
    with cols[i]:
        st.button("⬇️", key=f"col_{i}", on_click=make_move, args=(i,), disabled=not st.session_state.game_active)

board_html = '<div class="board">'
for r in range(ROWS):
    for c in range(COLS):
        val = st.session_state.board[r*COLS + c]
        color_class = ""
        if val == 1: color_class = "red"
        elif val == -1: color_class = "yellow"
        board_html += f'<div class="cell {color_class}"></div>'
board_html += '</div>'

st.markdown(board_html, unsafe_allow_html=True)
