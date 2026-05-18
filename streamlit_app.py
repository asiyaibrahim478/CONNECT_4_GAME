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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

[data-testid="stAppViewContainer"] {
    background-color: #040914 !important;
}

/* Main Layout Gaps */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(2)):not(:has(> div[data-testid="column"]:nth-of-type(3))) {
    gap: 30px;
    padding: 20px;
}

/* Left Panel (Settings) */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(2)):not(:has(> div[data-testid="column"]:nth-of-type(3))) > div[data-testid="column"]:nth-of-type(1) {
    background-color: #0b1221;
    border: 1px solid #1e293b;
    border-radius: 15px;
    padding: 25px;
    height: fit-content;
}

/* Right Panel (Game Area) */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(2)):not(:has(> div[data-testid="column"]:nth-of-type(3))) > div[data-testid="column"]:nth-of-type(2) {
    background-color: #081021;
    border: 1px solid #14223d;
    border-radius: 15px;
    padding: 40px;
}

/* Sidebar Buttons General */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(2)):not(:has(> div[data-testid="column"]:nth-of-type(3))) > div[data-testid="column"]:nth-of-type(1) div[data-testid="stButton"] > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 800;
    font-size: 1rem;
    padding: 12px 20px;
    display: flex;
    justify-content: flex-start;
    align-items: center;
    color: white;
    transition: all 0.2s;
    background-color: transparent;
    margin-bottom: 5px;
}

/* Reset Button */
div[data-testid="column"]:nth-of-type(1) div.element-container:nth-of-type(6) button {
    background: linear-gradient(180deg, #38bdf8, #2563eb) !important;
    border: none !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
    justify-content: center !important;
}

/* Retrain Button */
div[data-testid="column"]:nth-of-type(1) div.element-container:nth-of-type(7) button {
    background: linear-gradient(180deg, #c084fc, #7e22ce) !important;
    border: none !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
    justify-content: center !important;
}

/* Custom Board and Header CSS */

div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(7)) {
    background: linear-gradient(180deg, #0f40a3, #092c73);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 
        10px 10px 25px rgba(0,0,0,0.6), 
        inset 0px 4px 6px rgba(255,255,255,0.2), 
        inset 0px -4px 6px rgba(0,0,0,0.3);
    border: 2px solid #1a56d4;
    width: fit-content !important;
    margin: 0 auto;
    gap: 15px;
    display: flex;
    justify-content: center;
}

div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(7)) > div[data-testid="column"] {
    width: 60px !important;
    min-width: 60px !important;
    flex: none !important;
    gap: 15px;
}

div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(7)) button {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    border: none !important;
    padding: 0 !important;
    transition: transform 0.1s;
    background: #091224; /* deep dark hole */
    box-shadow: 
        inset 5px 5px 10px rgba(0,0,0,0.8),
        inset -2px -2px 5px rgba(255,255,255,0.05) !important;
}

div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(7)) button p {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

if 'level' not in st.session_state:
    st.session_state.level = 'Medium'

def set_level(lvl):
    st.session_state.level = lvl

# Dynamic CSS for Left Panel Settings Buttons
css = f"""
<style>
/* Easy Button */
div[data-testid="column"]:nth-of-type(1) div.element-container:nth-of-type(2) button {{
    border: 2px solid {'#22c55e' if st.session_state.level == 'Easy' else '#166534'} !important;
    background: {'linear-gradient(90deg, rgba(34,197,94,0.3) 0%, transparent 100%)' if st.session_state.level == 'Easy' else '#0b1221'} !important;
}}
/* Medium Button */
div[data-testid="column"]:nth-of-type(1) div.element-container:nth-of-type(3) button {{
    border: 2px solid {'#eab308' if st.session_state.level == 'Medium' else '#854d0e'} !important;
    background: {'linear-gradient(90deg, rgba(234,179,8,0.3) 0%, transparent 100%)' if st.session_state.level == 'Medium' else '#0b1221'} !important;
}}
/* Hard Button */
div[data-testid="column"]:nth-of-type(1) div.element-container:nth-of-type(4) button {{
    border: 2px solid {'#ef4444' if st.session_state.level == 'Hard' else '#991b1b'} !important;
    background: {'linear-gradient(90deg, rgba(239,68,68,0.3) 0%, transparent 100%)' if st.session_state.level == 'Hard' else '#0b1221'} !important;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

left_panel, right_panel = st.columns([1, 2.5])

with left_panel:
    st.markdown('<div style="color: white; font-size: 1.2rem; font-weight: 800; display: flex; align-items: center; gap: 10px; margin-bottom: 15px;"><span style="font-size: 1.5rem;">⚙️</span> SETTINGS</div><hr style="border-color: #1e2a44; margin: 0 0 15px 0;"><div style="color: white; font-weight: 800; font-size: 0.9rem; margin-bottom: 10px; letter-spacing: 1px;">DIFFICULTY</div>', unsafe_allow_html=True)
    
    st.button("🟢  EASY", on_click=set_level, args=('Easy',), key="btn_easy")
    st.button("🟡  MEDIUM", on_click=set_level, args=('Medium',), key="btn_medium")
    st.button("🔴  HARD", on_click=set_level, args=('Hard',), key="btn_hard")
    
    st.markdown('<hr style="border-color: #1e2a44; margin: 20px 0;">', unsafe_allow_html=True)
    if st.button("🔄  RESET GAME", key="btn_reset"):
        st.session_state.board = init_board()
        st.session_state.current_player = 1
        st.session_state.game_active = True
        st.session_state.winner = 0
        st.rerun()
        
    if st.button("🧠  RETRAIN AI", key="btn_retrain"):
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

with right_panel:
    status_html = ""
    if not st.session_state.game_active:
        if st.session_state.winner == 1:
            status_html = '<div style="display: inline-block; padding: 10px 40px; border-radius: 30px; border: 2px solid #047857; background: #064e3b; color: #34d399; font-weight: 800; font-size: 1.2rem; letter-spacing: 1px; box-shadow: 0 4px 6px rgba(0,0,0,0.4);">🏆 CONGRATULATIONS! YOU WIN! 🎉</div>'
            st.balloons()
        elif st.session_state.winner == -1:
            status_html = '<div style="display: inline-block; padding: 10px 40px; border-radius: 30px; border: 2px solid #991b1b; background: #450a0a; color: #f87171; font-weight: 800; font-size: 1.2rem; letter-spacing: 1px; box-shadow: 0 4px 6px rgba(0,0,0,0.4);">🤖 AI WINS! TRY AGAIN.</div>'
        else:
            status_html = '<div style="display: inline-block; padding: 10px 40px; border-radius: 30px; border: 2px solid #374151; background: #1f2937; color: #9ca3af; font-weight: 800; font-size: 1.2rem; letter-spacing: 1px; box-shadow: 0 4px 6px rgba(0,0,0,0.4);">🤝 IT\'S A DRAW!</div>'
    else:
        if st.session_state.current_player == 1:
            status_html = '<div style="display: inline-block; padding: 10px 40px; border-radius: 30px; border: 2px solid #047857; background: #064e3b; color: #34d399; font-weight: 800; font-size: 1.2rem; letter-spacing: 1px; box-shadow: 0 4px 6px rgba(0,0,0,0.4);">👤 YOUR TURN</div>'
        else:
            status_html = '<div style="display: inline-block; padding: 10px 40px; border-radius: 30px; border: 2px solid #854d0e; background: #422006; color: #fbbf24; font-weight: 800; font-size: 1.2rem; letter-spacing: 1px; box-shadow: 0 4px 6px rgba(0,0,0,0.4);">🤖 AI IS THINKING...</div>'

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 4rem; margin: 0; font-weight: 800; letter-spacing: 3px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            <span style="color: #38bdf8;">AI</span> <span style="color: white;">CONNECT 4</span>
        </h1>
        <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 10px; margin-bottom: 25px;">
            <hr style="width: 150px; border-color: #374151; margin: 0;">
            <div style="width: 12px; height: 12px; background: #eab308; border-radius: 50%;"></div>
            <hr style="width: 150px; border-color: #374151; margin: 0;">
        </div>
        {{status_html}}
    </div>
    """.format(status_html=status_html), unsafe_allow_html=True)

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
    depth = depth_map.get(st.session_state.level, 3)
    
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

win_cells = []
if not st.session_state.game_active and st.session_state.winner != 0:
    win_cells = get_win_cells(st.session_state.board, st.session_state.winner)

# --- Dynamic Board Colors ---
css = "<style>\\n"
for c in range(COLS):
    for r in range(ROWS):
        val = st.session_state.board[r*COLS + c]
        is_win = (r, c) in win_cells
        selector = f'div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-of-type(7)) > div[data-testid="column"]:nth-of-type({c+1}) div.element-container:nth-of-type({r+1}) button'
        
        if val == 1:
            css += f'{selector} {{ background: radial-gradient(circle at 35% 35%, #ff7676 0%, #e02a2a 40%, #8b0000 100%) !important; box-shadow: inset -4px -4px 8px rgba(0,0,0,0.6), inset 4px 4px 8px rgba(255,255,255,0.4), 3px 3px 6px rgba(0,0,0,0.6) !important; }}\\n'
        elif val == -1:
            css += f'{selector} {{ background: radial-gradient(circle at 35% 35%, #fff085 0%, #facc15 40%, #b45309 100%) !important; box-shadow: inset -4px -4px 8px rgba(0,0,0,0.6), inset 4px 4px 8px rgba(255,255,255,0.6), 3px 3px 6px rgba(0,0,0,0.6) !important; }}\\n'
            
        if is_win:
            css += f'{selector} {{ border: 3px solid white !important; box-shadow: 0 0 15px white, inset 0 0 10px white !important; }}\\n'
css += "</style>"
st.markdown(css, unsafe_allow_html=True)

# --- Draw Board ---
board_cols = st.columns(7)
for c in range(COLS):
    with board_cols[c]:
        for r in range(ROWS):
            st.button(" ", key=f"btn_{r}_{c}", on_click=make_move, args=(c,), disabled=not st.session_state.game_active)
