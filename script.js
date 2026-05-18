const ROWS = 6;
const COLS = 7;
const boardElement = document.getElementById('board');
const statusElement = document.getElementById('status');
const resetBtn = document.getElementById('reset-btn');
const retrainBtn = document.getElementById('retrain-btn');
const levelBtns = document.querySelectorAll('.level-btn');

let board = Array(ROWS * COLS).fill(0); // 0: empty, 1: red (User), -1: yellow (AI)
let currentPlayer = 1; // 1: Red, -1: Yellow
let gameActive = true;
let currentLevel = 'medium';

// Difficulty Level Handlers
levelBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        levelBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentLevel = btn.dataset.level;
    });
});

function initBoard() {
    boardElement.innerHTML = '';
    for (let i = 0; i < ROWS * COLS; i++) {
        const cell = document.createElement('div');
        cell.classList.add('cell');
        cell.dataset.index = i;
        cell.addEventListener('click', () => handleCellClick(i % COLS));
        boardElement.appendChild(cell);
    }
}

function updateBoard() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell, i) => {
        cell.classList.remove('red', 'yellow');
        if (board[i] === 1) cell.classList.add('red');
        if (board[i] === -1) cell.classList.add('yellow');
    });
}

function handleCellClick(col) {
    if (!gameActive || currentPlayer !== 1) return;

    if (dropPiece(col, 1)) {
        const winResult = checkWin(1);
        if (winResult) {
            highlightWin(winResult);
            endGame("Congratulations! You Win! 🌸", 1);
        } else if (board.every(cell => cell !== 0)) {
            endGame("It's a Draw! 🤝", 0);
        } else {
            currentPlayer = -1;
            statusElement.innerHTML = `<span style="color: #f59e0b">AI (${currentLevel}) is thinking...</span>`;
            setTimeout(makeAiMove, 300);
        }
    }
}

function dropPiece(col, player) {
    for (let r = ROWS - 1; r >= 0; r--) {
        if (board[r * COLS + col] === 0) {
            board[r * COLS + col] = player;
            updateBoard();
            return true;
        }
    }
    return false;
}

async function makeAiMove() {
    try {
        const response = await fetch('/api/ai_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: board, level: currentLevel })
        });
        const data = await response.json();
        
        if (data.column !== undefined) {
            dropPiece(data.column, -1);
            const winResult = checkWin(-1);
            if (winResult) {
                highlightWin(winResult);
                endGame("Sorry you lost, play best for the next time. 🤖", -1);
            } else if (board.every(cell => cell !== 0)) {
                endGame("It's a Draw! 🤝", 0);
            } else {
                currentPlayer = 1;
                statusElement.innerText = "Your Turn (Red)";
            }
        }
    } catch (error) {
        console.error("AI Error:", error);
        statusElement.innerText = "Error. Making random move...";
        const validCols = Array.from({length: COLS}, (_, i) => i).filter(c => board[c] === 0);
        if (validCols.length > 0) {
            dropPiece(validCols[Math.floor(Math.random() * validCols.length)], -1);
            currentPlayer = 1;
            statusElement.innerText = "Your Turn (Red)";
        }
    }
}

function checkWin(player) {
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS - 3; c++) {
            const indices = [r*COLS+c, r*COLS+c+1, r*COLS+c+2, r*COLS+c+3];
            if (indices.every(idx => board[idx] === player)) return indices;
        }
    }
    for (let r = 0; r < ROWS - 3; r++) {
        for (let c = 0; c < COLS; c++) {
            const indices = [r*COLS+c, (r+1)*COLS+c, (r+2)*COLS+c, (r+3)*COLS+c];
            if (indices.every(idx => board[idx] === player)) return indices;
        }
    }
    for (let r = 0; r < ROWS - 3; r++) {
        for (let c = 0; c < COLS - 3; c++) {
            const indices = [r*COLS+c, (r+1)*COLS+c+1, (r+2)*COLS+c+2, (r+3)*COLS+c+3];
            if (indices.every(idx => board[idx] === player)) return indices;
        }
    }
    for (let r = 3; r < ROWS; r++) {
        for (let c = 0; c < COLS - 3; c++) {
            const indices = [r*COLS+c, (r-1)*COLS+c+1, (r-2)*COLS+c+2, (r-3)*COLS+c+3];
            if (indices.every(idx => board[idx] === player)) return indices;
        }
    }
    return null;
}

function highlightWin(indices) {
    const cells = document.querySelectorAll('.cell');
    indices.forEach(idx => {
        cells[idx].classList.add('win');
    });
}

function endGame(message, winner) {
    statusElement.innerText = message;
    gameActive = false;
    
    if (winner === 1) {
        statusElement.classList.add('win-status');
        triggerCelebration();
    } else if (winner === -1) {
        statusElement.classList.add('loss-status');
    }
    
    saveGame(board, winner);
}

function triggerCelebration() {
    const duration = 3 * 1000;
    const end = Date.now() + duration;

    (function frame() {
        confetti({
            particleCount: 7,
            angle: 60,
            spread: 55,
            origin: { x: 0 },
            colors: ['#ff69b4', '#ff1493', '#ff00ff', '#ffffff'] // Floral colors
        });
        confetti({
            particleCount: 7,
            angle: 120,
            spread: 55,
            origin: { x: 1 },
            colors: ['#ff69b4', '#ff1493', '#ff00ff', '#ffffff']
        });

        if (Date.now() < end) {
            requestAnimationFrame(frame);
        }
    }());

    // Extra burst for "flower spreading" feel
    confetti({
        particleCount: 150,
        spread: 100,
        origin: { y: 0.6 },
        colors: ['#ff69b4', '#ff1493', '#ff00ff', '#ffffff', '#ef4444']
    });
}

async function saveGame(finalBoard, winner) {
    try {
        await fetch('/api/save_game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: finalBoard, winner: winner })
        });
        console.log("Game result saved to database.");
    } catch (error) {
        console.error("Error saving game:", error);
    }
}

// --- Retrain Logic ---
retrainBtn.addEventListener('click', async () => {
    if (!confirm("This will retrain all models using your latest game history. Continue?")) return;

    retrainBtn.disabled = true;
    retrainBtn.innerText = "Retraining...";
    statusElement.innerHTML = '<span style="color: #38bdf8">AI is learning from history...</span>';

    try {
        const response = await fetch('/api/retrain', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            alert("AI Retrained successfully! It is now smarter.");
            statusElement.innerText = "AI Updated! Your Turn (Red)";
        } else {
            alert("Error during retraining: " + data.message);
            statusElement.innerText = "Retrain Failed.";
        }
    } catch (error) {
        console.error("Retrain Error:", error);
        alert("Server error during retraining.");
    } finally {
        retrainBtn.disabled = false;
        retrainBtn.innerText = "Retrain AI";
    }
});

resetBtn.addEventListener('click', () => {
    board = Array(ROWS * COLS).fill(0);
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => cell.classList.remove('red', 'yellow', 'win'));
    statusElement.classList.remove('win-status', 'loss-status');
    currentPlayer = 1;
    gameActive = true;
    statusElement.innerText = "Your Turn (Red)";
    updateBoard();
});

initBoard();
