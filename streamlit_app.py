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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: #0b1120;
    color: white;
}
[data-testid="stHeader"] {
    background-color: transparent;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}

.main-header {
    text-align: center;
    margin-bottom: 20px;
}
.main-header h1 {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 0;
    letter-spacing: 2px;
}
.main-header h1 span {
    color: #38bdf8;
}

/* Board Styling */
.board-container {
    background: linear-gradient(180deg, #1e40af, #1e3a8a);
    padding: 15px;
    border-radius: 20px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.5), inset 0 4px 6px rgba(255,255,255,0.1);
    margin: 0 auto;
    width: fit-content;
}

.board {
    display: grid;
    grid-template-columns: repeat(7, 50px);
    grid-gap: 10px;
}

.cell {
    width: 50px;
    height: 50px;
    background: #0f172a;
    border-radius: 50%;
    position: relative;
    box-shadow: inset 0 5px 10px rgba(0,0,0,0.8);
}

.cell.red::after {
    content: '';
    position: absolute;
    inset: 2px;
    background: radial-gradient(circle at 35% 35%, #ef4444 0%, #b91c1c 100%);
    border-radius: 50%;
    box-shadow: inset -3px -3px 6px rgba(0,0,0,0.4), inset 3px 3px 6px rgba(255,255,255,0.3), 2px 2px 4px rgba(0,0,0,0.5);
}

.cell.yellow::after {
    content: '';
    position: absolute;
    inset: 2px;
    background: radial-gradient(circle at 35% 35%, #facc15 0%, #b45309 100%);
    border-radius: 50%;
    box-shadow: inset -3px -3px 6px rgba(0,0,0,0.4), inset 3px 3px 6px rgba(255,255,255,0.3), 2px 2px 4px rgba(0,0,0,0.5);
}

.cell.win::before {
    content: '';
    position: absolute;
    inset: -4px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    animation: pulse-win 1.5s infinite;
}

.cell.win::after {
    border: 3px solid white;
    box-shadow: 0 0 15px white;
}

@keyframes pulse-win {
    0% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.2); opacity: 0; }
    100% { transform: scale(1); opacity: 0.5; }
}

.status-badge {
    display: inline-block;
    padding: 10px 30px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 1.2rem;
    letter-spacing: 1px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
.status-turn {
    border: 2px solid #047857; background: #064e3b; color: #34d399;
}
.status-ai {
    border: 2px solid #854d0e; background: #422006; color: #fbbf24;
}
.status-win {
    border: 2px solid #047857; background: #064e3b; color: #34d399;
}
.status-loss {
    border: 2px solid #991b1b; background: #450a0a; color: #f87171;
}
.status-draw {
    border: 2px solid #374151; background: #1f2937; color: #9ca3af;
}

/* Button override to make drop buttons look seamless */
.stButton > button {
    border-radius: 20px;
    font-weight: 800;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    level = st.radio("Difficulty", ['Easy', 'Medium', 'Hard'], index=1)
    
    st.markdown("---")
    st.header("Actions")
    if st.button("🔄 Reset Game", use_container_width=True):
        st.session_state.board = init_board()
        st.session_state.current_player = 1
        st.session_state.game_active = True
        st.session_state.winner = 0
        st.rerun()
        
    if st.button("🧠 Retrain AI", use_container_width=True):
        with st.spinner("Retraining..."):
            script_path = os.path.join(BASE_DIR, "train_models.py")
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=BASE_DIR)
            if result.returncode == 0:
                st.success("AI Retrained successfully!")
                load_ml_model.clear()
            else:
                try:
                    from train_models import preprocess_and_train
                    preprocess_and_train()
                    st.success("AI Retrained successfully (fallback)!")
                    load_ml_model.clear()
                except Exception as e:
                    st.error(f"Retraining failed: {e}")

status_html = ""
if not st.session_state.game_active:
    if st.session_state.winner == 1:
        status_html = '<div class="status-badge status-win">🏆 YOU WIN! 🎉</div>'
        st.balloons()
    elif st.session_state.winner == -1:
        status_html = '<div class="status-badge status-loss">🤖 AI WINS! TRY AGAIN.</div>'
    else:
        status_html = '<div class="status-badge status-draw">🤝 IT\'S A DRAW!</div>'
else:
    if st.session_state.current_player == 1:
        status_html = '<div class="status-badge status-turn">👤 YOUR TURN</div>'
    else:
        status_html = '<div class="status-badge status-ai">🤖 AI IS THINKING...</div>'

st.markdown(f"""
<div class="main-header">
    <h1><span>AI</span> CONNECT 4</h1>
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 15px; margin-bottom: 25px;">
        <hr style="width: 100px; border-color: #374151; margin: 0;">
        <div style="width: 10px; height: 10px; background: #eab308; border-radius: 50%;"></div>
        <hr style="width: 100px; border-color: #374151; margin: 0;">
    </div>
    {status_html}
</div>
""", unsafe_allow_html=True)

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

def get_win_cells(board, player):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r*COLS + c + i] == player for i in range(4)):
                return [(r, c+i) for i in range(4)]
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[(r+i)*COLS + c] == player for i in range(4)):
                return [(r+i, c) for i in range(4)]
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[(r+i)*COLS + c + i] == player for i in range(4)):
                return [(r+i, c+i) for i in range(4)]
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[(r-i)*COLS + c + i] == player for i in range(4)):
                return [(r-i, c+i) for i in range(4)]
    return []

win_cells = []
if not st.session_state.game_active and st.session_state.winner != 0:
    win_cells = get_win_cells(st.session_state.board, st.session_state.winner)

# --- Draw Board ---
spacer_left, center_col, spacer_right = st.columns([1, 1.5, 1])

with center_col:
    # Top buttons to act as droppers
    st.markdown('<div style="width: fit-content; margin: 0 auto; display: grid; grid-template-columns: repeat(7, 50px); grid-gap: 10px; margin-bottom: 5px;">', unsafe_allow_html=True)
    btn_cols = st.columns(7)
    for i in range(7):
        with btn_cols[i]:
            st.button("🔽", key=f"col_{i}", on_click=make_move, args=(i,), disabled=not st.session_state.game_active, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    board_html = '<div class="board-container"><div class="board">'
    for r in range(ROWS):
        for c in range(COLS):
            val = st.session_state.board[r*COLS + c]
            color_class = ""
            win_class = " win" if (r, c) in win_cells else ""
            if val == 1: color_class = "red"
            elif val == -1: color_class = "yellow"
            board_html += f'<div class="cell {color_class}{win_class}"></div>'
    board_html += '</div></div>'

    st.markdown(board_html, unsafe_allow_html=True)
