import streamlit as st
import numpy as np
import joblib
import os
import math
import subprocess
import sys
import time
from connect_db import save_game_result

st.set_page_config(page_title="AI Connect 4 - MLOps Dashboard", layout="wide")

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

def get_win_cells(board, player):
    # horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r*COLS + c + i] == player for i in range(4)):
                return [(r, c+i) for i in range(4)]
    # vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[(r+i)*COLS + c] == player for i in range(4)):
                return [(r+i, c) for i in range(4)]
    # diagonal positive
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[(r+i)*COLS + c + i] == player for i in range(4)):
                return [(r+i, c+i) for i in range(4)]
    # diagonal negative
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[(r-i)*COLS + c + i] == player for i in range(4)):
                return [(r-i, c+i) for i in range(4)]
    return []

# --- State Management ---
if 'board' not in st.session_state:
    st.session_state.board = init_board()
    st.session_state.current_player = 1 # 1 for User, -1 for AI
    st.session_state.game_active = True
    st.session_state.winner = 0
if 'level' not in st.session_state:
    st.session_state.level = 'Medium'
if 'retrain_status' not in st.session_state:
    st.session_state.retrain_status = ""

# --- Native Query Parameters Action Handler ---
params = st.query_params
if params:
    # 1. Column click action
    if "col" in params:
        try:
            col = int(params["col"])
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
        except Exception as e:
            pass
        st.query_params.clear()
        st.rerun()

    # 2. Difficulty selection action
    elif "difficulty" in params:
        lvl = params["difficulty"]
        if lvl in ["easy", "medium", "hard"]:
            st.session_state.level = lvl.capitalize()
            st.session_state.retrain_status = ""
        st.query_params.clear()
        st.rerun()

    # 3. Reset game action
    elif "action" in params and params["action"] == "reset":
        st.session_state.board = init_board()
        st.session_state.current_player = 1
        st.session_state.game_active = True
        st.session_state.winner = 0
        st.session_state.retrain_status = ""
        st.query_params.clear()
        st.rerun()

    # 4. Retrain AI action
    elif "action" in params and params["action"] == "retrain":
        st.session_state.retrain_status = "training"
        st.query_params.clear()
        st.rerun()

# --- AI Turn Handler ---
if st.session_state.game_active and st.session_state.current_player == -1:
    time.sleep(0.3)
    
    depth_map = {'Easy': 1, 'Medium': 3, 'Hard': 5}
    depth = depth_map.get(st.session_state.level, 3)
    
    valid_cols = get_valid_columns(st.session_state.board)
    if valid_cols:
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
        if new_b:
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

# --- Retraining Executor ---
if st.session_state.retrain_status == "training":
    script_path = os.path.join(BASE_DIR, "train_models.py")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=BASE_DIR)
    
    if result.returncode == 0:
        st.session_state.retrain_status = "success"
        load_ml_model.clear()
        model = load_ml_model()
    else:
        try:
            from train_models import preprocess_and_train
            preprocess_and_train()
            st.session_state.retrain_status = "success"
            load_ml_model.clear()
            model = load_ml_model()
        except Exception as e:
            st.session_state.retrain_status = f"failed: {str(e)}"
    st.rerun()

# --- HTML/CSS Compiler & Renderer ---

# Load style.css and append custom Streamlit overrides
try:
    with open(os.path.join(BASE_DIR, 'style.css'), 'r') as f:
        style_css = f.read()
except Exception as e:
    style_css = ""
    st.error(f"Error loading style.css: {e}")

streamlit_overrides = """
/* Streamlit UI overrides */
html, body {
    overflow: hidden !important;
    height: 100vh !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stHeader"] {
    display: none !important;
}
footer {
    display: none !important;
}
[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
div[data-testid="stButton"] {
    display: none !important;
}
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
}
.game-wrapper {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 999999 !important;
    background-color: #0b0f1a !important;
}
/* Style adjustments for HTML links acting as buttons */
.level-btn {
    display: inline-block !important;
    text-decoration: none !important;
    box-sizing: border-box !important;
}
.action-btn {
    display: block !important;
    text-decoration: none !important;
    box-sizing: border-box !important;
}
.cell-link {
    display: block !important;
    width: 100% !important;
    height: 100% !important;
    text-decoration: none !important;
    border-radius: 50% !important;
}
"""

# Strip all blank lines and whitespace from CSS to prevent markdown rendering issues
clean_style_css = "\n".join([line.strip() for line in style_css.split("\n") if line.strip()])
clean_streamlit_overrides = "\n".join([line.strip() for line in streamlit_overrides.split("\n") if line.strip()])
style_tag = f"<style>\n{clean_style_css}\n{clean_streamlit_overrides}\n</style>"
st.markdown(style_tag, unsafe_allow_html=True)

# Build Difficulty Selector
difficulty = st.session_state.level.lower()
btn_easy_active = "active" if difficulty == "easy" else ""
btn_medium_active = "active" if difficulty == "medium" else ""
btn_hard_active = "active" if difficulty == "hard" else ""

raw_difficulty_html = f"""
<div class="difficulty-selector">
<a class="level-btn {btn_easy_active}" href="?difficulty=easy" target="_self">Easy</a>
<a class="level-btn {btn_medium_active}" href="?difficulty=medium" target="_self">Medium</a>
<a class="level-btn {btn_hard_active}" href="?difficulty=hard" target="_self">Hard</a>
</div>
"""
difficulty_selector_html = "\n".join([line.strip() for line in raw_difficulty_html.split("\n") if line.strip()])

# Fetch winning cells (if game is over)
win_cells = []
if not st.session_state.game_active and st.session_state.winner != 0:
    win_cells = get_win_cells(st.session_state.board, st.session_state.winner)
win_cells_flat = [r * COLS + c for r, c in win_cells]

# Build Status Header
if not st.session_state.game_active:
    if st.session_state.winner == 1:
        raw_status_html = '<div class="game-status win-status" id="status">Congratulations! You Win! 🌸</div>'
    elif st.session_state.winner == -1:
        raw_status_html = '<div class="game-status loss-status" id="status">Sorry you lost, play best for the next time. 🤖</div>'
    else:
        raw_status_html = '<div class="game-status" id="status">It\'s a Draw! 🤝</div>'
else:
    if st.session_state.current_player == 1:
        raw_status_html = '<div class="game-status" id="status">Your Turn (Red)</div>'
    else:
        raw_status_html = f'<div class="game-status" id="status"><span style="color: #f59e0b">AI ({st.session_state.level}) is thinking...</span></div>'
status_html = "\n".join([line.strip() for line in raw_status_html.split("\n") if line.strip()])

# Build Board Cells Grid
board_html = ""
for i in range(ROWS * COLS):
    val = st.session_state.board[i]
    c = i % COLS
    classes = ["cell"]
    if val == 1:
        classes.append("red")
    elif val == -1:
        classes.append("yellow")
        
    if i in win_cells_flat:
        classes.append("win")
        
    classes_str = " ".join(classes)
    
    # Empty clickable cells are styled as standard HTML anchor links targets
    if st.session_state.game_active and st.session_state.current_player == 1 and val == 0:
        board_html += f'<a class="{classes_str}" href="?col={c}" target="_self"></a>\n'
    else:
        board_html += f'<div class="{classes_str}"></div>\n'

# Build Retraining Status Alert Message
retrain_message_html = ""
if st.session_state.retrain_status == "success":
    retrain_message_html = '<div class="game-status win-status" style="margin-top: 1rem; font-size: 0.85rem; padding: 0.5rem; text-align: center;">AI Retrained successfully! 🧠</div>'
elif st.session_state.retrain_status.startswith("failed"):
    err = st.session_state.retrain_status.split(":", 1)[1]
    retrain_message_html = f'<div class="game-status loss-status" style="margin-top: 1rem; font-size: 0.85rem; padding: 0.5rem; text-align: center;">Train failed: {err}</div>'

# Render main HTML wrapper
raw_main_html = f"""
<div class="game-wrapper">
<aside class="sidebar">
<div class="sidebar-header">
<h2>Settings</h2>
</div>
<div class="section">
<p class="section-title">Difficulty</p>
{difficulty_selector_html}
</div>
<div class="section">
<p class="section-title">Actions</p>
<a id="reset-btn" class="action-btn" href="?action=reset" target="_self">Reset Game</a>
<a id="retrain-btn" class="action-btn retrain" href="?action=retrain" target="_self" style="margin-top: 0.5rem;">Retrain AI</a>
{retrain_message_html}
</div>
<div class="footer">
<p>MLOps Pipeline v1.0</p>
<p>Logistic Regression AI</p>
</div>
</aside>
<main class="main-content">
<header>
<h1>AI <span>Connect 4</span></h1>
{status_html}
</header>
<div class="game-board-container">
<div class="game-board" id="board">
{board_html}
</div>
</div>
</main>
</div>
"""
main_html = "\n".join([line.strip() for line in raw_main_html.split("\n") if line.strip()])

st.markdown(main_html, unsafe_allow_html=True)
