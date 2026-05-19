import streamlit as st
import numpy as np
import joblib
import os
import math
import subprocess
import sys
import time
from connect_db import save_game_result

st.set_page_config(page_title="AI Connect 4 - MLOps Dashboard", layout="centered")

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
        pass
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

# --- Sidebar Panel ---
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #f43f5e; font-size: 2.2rem; font-weight: 700; margin: 0;">Connect 4</h2>
        <p style="color: #94a3b8; font-size: 0.95rem; margin: 0.2rem 0 0 0;">MLOps Pipeline Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader("Settings")
difficulty = st.sidebar.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"], index=["Easy", "Medium", "Hard"].index(st.session_state.level))
if difficulty != st.session_state.level:
    st.session_state.level = difficulty
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Actions")

if st.sidebar.button("Reset Game", use_container_width=True):
    st.session_state.board = init_board()
    st.session_state.current_player = 1
    st.session_state.game_active = True
    st.session_state.winner = 0
    st.session_state.retrain_status = ""
    st.rerun()

if st.sidebar.button("Retrain AI Model", use_container_width=True):
    st.session_state.retrain_status = "training"
    st.rerun()

# --- Retraining Executor ---
if st.session_state.retrain_status == "training":
    with st.sidebar:
        with st.spinner("Retraining models using play history..."):
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

if st.session_state.retrain_status == "success":
    st.sidebar.success("AI Retrained successfully! It is now smarter. 🧠")
elif st.session_state.retrain_status.startswith("failed"):
    err = st.session_state.retrain_status.split(":", 1)[1] if ":" in st.session_state.retrain_status else st.session_state.retrain_status
    st.sidebar.error(f"Retraining failed: {err}")

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

# --- Main Layout Title & Status ---
st.markdown(
    """
    <div style="text-align: center; margin-top: 1rem; margin-bottom: 2rem;">
        <h1 style="color: #ffffff; font-size: 3rem; font-weight: 800; margin: 0; font-family: 'Outfit', sans-serif;">
            AI <span style="background: linear-gradient(135deg, #ec4899, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Connect 4</span>
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin: 0.5rem 0 0 0;">Beat the Logistic Regression powered AI bot</p>
    </div>
    """,
    unsafe_allow_html=True
)

if not st.session_state.game_active:
    if st.session_state.winner == 1:
        st.balloons()
        st.markdown('<div class="win-banner">Congratulations! You Win! 🌸</div>', unsafe_allow_html=True)
    elif st.session_state.winner == -1:
        st.markdown('<div class="loss-banner">Sorry you lost, play best for the next time. 🤖</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="draw-banner">It\'s a Draw! 🤝</div>', unsafe_allow_html=True)
else:
    if st.session_state.current_player == 1:
        st.markdown('<div class="player-banner">Your Turn (Red)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ai-banner">AI is thinking...</div>', unsafe_allow_html=True)

# Fetch winning cells (if game is over)
win_cells = []
if not st.session_state.game_active and st.session_state.winner != 0:
    win_cells = get_win_cells(st.session_state.board, st.session_state.winner)
win_cells_flat = [r * COLS + c for r, c in win_cells]

# --- Render styled board columns ---
cols = st.columns(COLS)
for c in range(COLS):
    with cols[c]:
        for r in range(ROWS):
            val = st.session_state.board[r*COLS + c]
            cell_key = f"cell_{r}_{c}"
            
            # Identify coin symbol & state
            is_win = (r*COLS + c) in win_cells_flat
            
            if val == 1:
                symbol = "🔴"
                is_disabled = True
            elif val == -1:
                symbol = "🟡"
                is_disabled = True
            else:
                symbol = " "
                is_disabled = not st.session_state.game_active or st.session_state.current_player != 1
                
            # Render native button styled as cell slot
            if st.button(symbol, key=cell_key, disabled=is_disabled):
                new_b = drop_piece(st.session_state.board, c, 1)
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
                    st.rerun()

# --- Custom Styling Engine ---
# Injects pure, sanitization-immune layout and aesthetic tokens into Streamlit's container DOM.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Viewport adjustments */
    body {
        background-color: #0f172a !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Style the columns container to represent the Connect 4 Board */
    div[data-testid="stHorizontalBlock"] {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6) !important;
        padding: 24px !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), 
                    0 0 30px rgba(59, 130, 246, 0.3) !important;
        max-width: 580px !important;
        margin: 2rem auto !important;
        gap: 12px !important;
        display: flex !important;
        justify-content: center !important;
    }
    
    /* Force columns to lay out vertically and align perfectly */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 10px !important;
    }
    
    /* Style cells globally (the native Streamlit buttons) */
    div[data-testid="stHorizontalBlock"] button {
        width: 56px !important;
        height: 56px !important;
        border-radius: 50% !important;
        border: 3px solid #0f172a !important;
        background: radial-gradient(circle at 30% 30%, #1e293b, #0f172a) !important;
        box-shadow: inset 3px 3px 6px rgba(0, 0, 0, 0.8),
                    3px 3px 6px rgba(0, 0, 0, 0.3) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        font-size: 26px !important;
        padding: 0 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Interactive Hover on empty slots during player turn */
    div[data-testid="stHorizontalBlock"] button:not([disabled]):hover {
        transform: scale(1.08) !important;
        border-color: #f43f5e !important; /* Glowing Red hover */
        box-shadow: 0 0 15px rgba(244, 63, 94, 0.5) !important;
    }
    
    /* Keep occupied slots (disabled buttons) beautiful and glowing */
    div[data-testid="stHorizontalBlock"] button[disabled] {
        opacity: 1 !important;
        cursor: default !important;
        background: radial-gradient(circle at 30% 30%, #1e293b, #0f172a) !important;
    }
    
    /* Winning Line Glowing Effect */
    div[data-testid="stHorizontalBlock"] button.win {
        animation: win-glow 1.5s infinite alternate !important;
    }
    
    @keyframes win-glow {
        0% { transform: scale(1); box-shadow: 0 0 10px #f43f5e; border-color: #f43f5e; }
        100% { transform: scale(1.06); box-shadow: 0 0 25px #8b5cf6; border-color: #8b5cf6; }
    }
    
    /* Banner Notifications style */
    .player-banner, .ai-banner, .win-banner, .loss-banner, .draw-banner {
        text-align: center;
        padding: 0.8rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        max-width: 580px;
        margin: 0 auto 1.5rem auto;
    }
    .player-banner {
        background: rgba(244, 63, 94, 0.15);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.25);
    }
    .ai-banner {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    .win-banner {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .loss-banner {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .draw-banner {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    
    /* Hide standard Streamlit header and footer wrappers */
    [data-testid="stHeader"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
