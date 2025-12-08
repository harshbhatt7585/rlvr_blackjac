let socket = null;
let isWebSocketMode = false;

// Try to connect to WebSocket server
if (typeof io !== 'undefined') {
    socket = io();
    isWebSocketMode = true;
    
    socket.on('connect', () => {
        console.log('Connected to render server');
        updateMessage('Connected to training render server. Waiting for game state...', 'info');
        document.getElementById('gameStatus').textContent = 'Connected';
        addLog('Connected to training render server', 'info');
    });
    
    socket.on('game_state', (state) => {
        updateUIFromState(state);
    });
    
    socket.on('log', (logData) => {
        addLog(logData.message, logData.level || 'info', logData.timestamp);
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from render server');
        updateMessage('Disconnected from render server', 'info');
        addLog('Disconnected from render server', 'warning');
    });
} else {
    // Standalone mode - load socket.io library dynamically
    const script = document.createElement('script');
    script.src = 'https://cdn.socket.io/4.5.4/socket.io.min.js';
    script.onload = () => {
        socket = io();
        socket.on('connect', () => {
            isWebSocketMode = true;
            console.log('Connected to render server');
            updateMessage('Connected to training render server. Waiting for game state...', 'info');
            addLog('Connected to training render server', 'info');
        });
        socket.on('game_state', (state) => {
            updateUIFromState(state);
        });
        socket.on('log', (logData) => {
            addLog(logData.message, logData.level || 'info', logData.timestamp);
        });
    };
    document.head.appendChild(script);
}

class BlackjackGame {
    constructor() {
        this.deck = [];
        this.playerCards = [];
        this.dealerCards = [];
        this.playerSum = 0;
        this.dealerSum = 0;
        this.usableAce = false;
        this.done = false;
        this.gamesPlayed = 0;
        this.wins = 0;
        this.losses = 0;
    }

    createDeck() {
        // 1 = Ace, 2-10 = number cards, 11-13 = J, Q, K (all worth 10)
        const deck = [];
        for (let i = 0; i < 4; i++) {
            deck.push(1); // Ace
            for (let j = 2; j <= 10; j++) {
                deck.push(j);
            }
            deck.push(10, 10, 10); // J, Q, K
        }
        // Shuffle
        for (let i = deck.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [deck[i], deck[j]] = [deck[j], deck[i]];
        }
        return deck;
    }

    drawCard() {
        if (this.deck.length === 0) {
            this.deck = this.createDeck();
        }
        return this.deck.pop();
    }

    calculateHand(cards) {
        let total = cards.reduce((sum, card) => {
            if (card === 1) return sum + 1; // Ace as 1
            if (card >= 11) return sum + 10; // Face cards
            return sum + card;
        }, 0);

        let usableAce = false;
        if (cards.includes(1)) {
            if (total + 10 <= 21) {
                total += 10;
                usableAce = true;
            }
        }

        return { total, usableAce };
    }

    isBust(handSum) {
        return handSum > 21;
    }

    cardToString(card) {
        if (card === 1) return 'A';
        if (card >= 11) return '10/J/Q/K';
        return card.toString();
    }

    cardToDisplay(card) {
        if (card === 1) return { value: 'A', suit: '♠' };
        if (card >= 11) {
            const faces = ['J', 'Q', 'K'];
            return { value: faces[(card - 11) % 3], suit: '♠' };
        }
        return { value: card.toString(), suit: '♠' };
    }

    reset() {
        this.deck = this.createDeck();
        this.done = false;
        this.playerCards = [this.drawCard(), this.drawCard()];
        this.dealerCards = [this.drawCard(), this.drawCard()];
        
        const playerCalc = this.calculateHand(this.playerCards);
        this.playerSum = playerCalc.total;
        this.usableAce = playerCalc.usableAce;

        const dealerCalc = this.calculateHand(this.dealerCards);
        this.dealerSum = dealerCalc.total;

        this.gamesPlayed++;
        return this.getObservation();
    }

    getObservation() {
        return {
            playerSum: this.playerSum,
            dealerCard: this.dealerCards[0],
            usableAce: this.usableAce,
            playerCards: [...this.playerCards],
            dealerCards: [...this.dealerCards],
            done: this.done
        };
    }

    step(action) {
        if (this.done) {
            throw new Error("Episode is done. Call reset() to start a new episode.");
        }

        let reward = 0.0;
        let info = {};

        if (action === 1) { // Hit
            const newCard = this.drawCard();
            this.playerCards.push(newCard);
            const playerCalc = this.calculateHand(this.playerCards);
            this.playerSum = playerCalc.total;
            this.usableAce = playerCalc.usableAce;

            if (this.isBust(this.playerSum)) {
                this.done = true;
                reward = -1.0;
                info.result = "bust";
            } else if (this.playerSum === 21) {
                this.done = true;
                reward = 1.0;
            }
        } else { // Stand
            const dealerCalc = this.calculateHand(this.dealerCards);
            this.dealerSum = dealerCalc.total;

            while (this.dealerSum < 17) {
                this.dealerCards.push(this.drawCard());
                const newDealerCalc = this.calculateHand(this.dealerCards);
                this.dealerSum = newDealerCalc.total;
            }

            this.done = true;

            const dealerBust = this.isBust(this.dealerSum);

            if (dealerBust) {
                reward = 1.0;
            } else if (this.playerSum > this.dealerSum) {
                reward = 1.0;
            } else if (this.playerSum === this.dealerSum) {
                reward = 0.0;
            } else {
                reward = -1.0;
            }
        }

        return { observation: this.getObservation(), reward, done: this.done, info };
    }
}

// Game instance (for standalone mode)
const game = new BlackjackGame();

// Track previous card counts for animation
let previousPlayerCardCount = 0;
let previousDealerCardCount = 0;
let previousDealerDone = false;

// UI Update functions
function updateUIFromState(state) {
    // Update from Python environment state
    const oldPlayerCount = game.playerCards.length;
    const oldDealerCount = game.dealerCards.length;
    const oldDone = game.done;
    
    game.playerCards = state.player_cards || [];
    game.dealerCards = state.dealer_cards || [];
    game.playerSum = state.player_sum || 0;
    game.dealerSum = state.dealer_sum || 0;
    game.usableAce = state.usable_ace || false;
    game.done = state.done || false;
    
    // Reset previous counts if cards were reset (new game)
    if (game.playerCards.length < oldPlayerCount || game.dealerCards.length < oldDealerCount) {
        previousPlayerCardCount = 0;
        previousDealerCardCount = 0;
        previousDealerDone = false;
    }
    
    updateUI();
    
    // Show action and reward info if available
    if (state.action !== undefined && state.action !== null) {
        const actionName = state.action === 1 ? 'Hit' : 'Stand';
        let message = `Action: ${actionName}`;
        if (state.reward !== undefined && state.reward !== null) {
            message += ` | Reward: ${state.reward.toFixed(2)}`;
        }
        if (state.done) {
            if (state.info && state.info.result === 'bust') {
                updateMessage(`Bust! ${message}`, 'lose');
            } else if (state.reward > 0) {
                updateMessage(`Win! ${message}`, 'win');
            } else if (state.reward < 0) {
                updateMessage(`Loss! ${message}`, 'lose');
            } else {
                updateMessage(`Draw! ${message}`, 'draw');
            }
        } else {
            updateMessage(message, 'info');
        }
    }
}

function updateUI() {
    // Update player total
    document.getElementById('playerTotal').textContent = game.playerSum;
    document.getElementById('playerTotalDisplay').textContent = 
        `Total: ${game.playerSum}${game.usableAce ? ' (with usable ace)' : ''}`;
    
    // Update dealer visible card
    const dealerVisible = game.dealerCards[0];
    if (dealerVisible !== undefined) {
        document.getElementById('dealerVisible').textContent = game.cardToString(dealerVisible);
    }
    
    // Check if dealer's hidden card should be revealed
    const dealerCardRevealed = game.done && !previousDealerDone && previousDealerCardCount >= 2;
    
    // Update player cards with animation
    const playerCardsContainer = document.getElementById('playerCards');
    const currentPlayerCardCount = game.playerCards.length;
    const isNewPlayerCard = currentPlayerCardCount > previousPlayerCardCount;
    
    // Only rebuild if card count changed
    if (isNewPlayerCard || playerCardsContainer.children.length !== currentPlayerCardCount) {
        playerCardsContainer.innerHTML = '';
        game.playerCards.forEach((card, index) => {
            const cardEl = createCardElement(card, false);
            // Animate new cards
            if (index >= previousPlayerCardCount) {
                cardEl.classList.add('deal');
                // Add flip animation with delay
                setTimeout(() => {
                    cardEl.classList.add('flip-in');
                }, index * 100);
            }
            playerCardsContainer.appendChild(cardEl);
        });
    }
    
    // Update dealer cards with animation
    const dealerCardsContainer = document.getElementById('dealerCards');
    const currentDealerCardCount = game.dealerCards.length;
    const isNewDealerCard = currentDealerCardCount > previousDealerCardCount;
    const shouldRebuildDealer = isNewDealerCard || dealerCardRevealed || 
                                 dealerCardsContainer.children.length !== currentDealerCardCount;
    
    if (shouldRebuildDealer) {
        dealerCardsContainer.innerHTML = '';
        game.dealerCards.forEach((card, index) => {
            // If card is being revealed, start as hidden
            const shouldStartHidden = dealerCardRevealed && index === 1;
            const isHidden = !game.done && index === 1;
            const cardEl = createCardElement(card, shouldStartHidden || isHidden);
            
            // Animate new dealer cards
            if (index >= previousDealerCardCount) {
                cardEl.classList.add('deal');
                setTimeout(() => {
                    cardEl.classList.add('flip-in');
                }, index * 100);
            }
            
            // Animate hidden card reveal when game ends
            if (dealerCardRevealed && index === 1) {
                // Wait a bit, then flip to reveal
                setTimeout(() => {
                    cardEl.classList.add('flip-reveal');
                    // Update card content mid-flip (at 50% of animation)
                    setTimeout(() => {
                        const cardDisplay = game.cardToDisplay(card);
                        cardEl.classList.remove('hidden');
                        cardEl.innerHTML = `
                            <div class="card-value">${cardDisplay.value}</div>
                            <div class="card-suit">${cardDisplay.suit}</div>
                        `;
                    }, 400);
                }, 500);
            }
            
            dealerCardsContainer.appendChild(cardEl);
        });
    }
    
    // Update previous counts
    previousPlayerCardCount = currentPlayerCardCount;
    previousDealerCardCount = currentDealerCardCount;
    previousDealerDone = game.done;
    
    // Update dealer total
    if (game.done && game.dealerSum !== undefined) {
        document.getElementById('dealerTotal').textContent = `Total: ${game.dealerSum}`;
    } else {
        const visibleCard = game.dealerCards[0];
        if (visibleCard !== undefined) {
            const visibleValue = visibleCard === 1 ? 11 : (visibleCard >= 11 ? 10 : visibleCard);
            document.getElementById('dealerTotal').textContent = `Showing: ${visibleValue}`;
        }
    }
    
    // Update game status
    const statusEl = document.getElementById('gameStatus');
    if (game.done) {
        statusEl.textContent = 'Game Over';
    } else {
        statusEl.textContent = isWebSocketMode ? 'Training...' : 'Your Turn';
    }
    
    // Update buttons (disable in WebSocket mode) - only if buttons exist
    if (isWebSocketMode) {
        const hitBtn = document.getElementById('hitBtn');
        const standBtn = document.getElementById('standBtn');
        const newGameBtn = document.getElementById('newGameBtn');
        if (hitBtn) hitBtn.disabled = true;
        if (standBtn) standBtn.disabled = true;
        if (newGameBtn) newGameBtn.disabled = true;
    } else {
        const hitBtn = document.getElementById('hitBtn');
        const standBtn = document.getElementById('standBtn');
        if (hitBtn) hitBtn.disabled = game.done;
        if (standBtn) standBtn.disabled = game.done;
    }
    
    // Update stats (only in standalone mode)
    if (!isWebSocketMode) {
        document.getElementById('gamesPlayed').textContent = game.gamesPlayed;
        document.getElementById('wins').textContent = game.wins;
        document.getElementById('losses').textContent = game.losses;
    }
}

function createCardElement(card, isHidden) {
    const cardDiv = document.createElement('div');
    cardDiv.className = 'card' + (isHidden ? ' hidden' : '');
    
    if (isHidden) {
        cardDiv.innerHTML = '<div class="card-value">?</div><div class="card-suit">?</div>';
    } else {
        const cardDisplay = game.cardToDisplay(card);
        cardDiv.innerHTML = `
            <div class="card-value">${cardDisplay.value}</div>
            <div class="card-suit">${cardDisplay.suit}</div>
        `;
    }
    
    return cardDiv;
}

function updateMessage(text, type = 'info') {
    const messageEl = document.getElementById('message');
    if (messageEl) {
        messageEl.textContent = text;
        messageEl.className = `message ${type}`;
    }
    // Also log important messages
    if (type !== 'info' || text.includes('Connected') || text.includes('Disconnected')) {
        addLog(text, type);
    }
}

function handleHit() {
    if (game.done || isWebSocketMode) return;
    
    const result = game.step(1);
    updateUI();
    
    if (result.done) {
        if (result.info.result === 'bust') {
            updateMessage(`Bust! You went over 21. You lose!`, 'lose');
            game.losses++;
        } else if (game.playerSum === 21) {
            updateMessage(`Blackjack! You win!`, 'win');
            game.wins++;
        }
    } else {
        updateMessage(`You drew a card. Total: ${game.playerSum}`, 'info');
    }
}

function handleStand() {
    if (game.done || isWebSocketMode) return;
    
    const result = game.step(0);
    updateUI();
    
    // Determine winner
    let message = '';
    let messageType = 'info';
    
    if (game.isBust(game.dealerSum)) {
        message = `Dealer busts with ${game.dealerSum}! You win!`;
        messageType = 'win';
        game.wins++;
    } else if (game.playerSum > game.dealerSum) {
        message = `You win! ${game.playerSum} vs ${game.dealerSum}`;
        messageType = 'win';
        game.wins++;
    } else if (game.playerSum === game.dealerSum) {
        message = `Push! Both have ${game.playerSum}`;
        messageType = 'draw';
    } else {
        message = `You lose! ${game.playerSum} vs ${game.dealerSum}`;
        messageType = 'lose';
        game.losses++;
    }
    
    updateMessage(message, messageType);
}

function newGame() {
    if (isWebSocketMode) return;
    // Reset previous counts for animation
    previousPlayerCardCount = 0;
    previousDealerCardCount = 0;
    previousDealerDone = false;
    game.reset();
    updateUI();
    updateMessage('New game started! Make your move.', 'info');
}

// Log management functions
function addLog(message, level = 'info', timestamp = null) {
    const logsContent = document.getElementById('logsContent');
    if (!logsContent) return;
    
    // Remove "Waiting for logs..." if it exists
    if (logsContent.children.length === 1 && logsContent.children[0].textContent.includes('Waiting for logs')) {
        logsContent.innerHTML = '';
    }
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${level}`;
    
    const timeStr = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    logEntry.innerHTML = `<span class="log-timestamp">[${timeStr}]</span>${escapeHtml(message)}`;
    
    logsContent.appendChild(logEntry);
    
    // Auto-scroll to bottom
    logsContent.scrollTop = logsContent.scrollHeight;
    
    // Limit to 1000 log entries to prevent memory issues
    if (logsContent.children.length > 1000) {
        logsContent.removeChild(logsContent.firstChild);
    }
}

function clearLogs() {
    const logsContent = document.getElementById('logsContent');
    if (logsContent) {
        logsContent.innerHTML = '<div class="log-entry">Logs cleared</div>';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize
if (!isWebSocketMode) {
    // Reset previous counts for initial render
    previousPlayerCardCount = 0;
    previousDealerCardCount = 0;
    previousDealerDone = false;
    updateUI();
} else {
    // Reset previous counts for WebSocket mode
    previousPlayerCardCount = 0;
    previousDealerCardCount = 0;
    previousDealerDone = false;
}
