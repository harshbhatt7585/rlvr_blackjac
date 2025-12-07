// Blackjack game implementation - supports both standalone and WebSocket modes

// Check if socket.io is available (WebSocket mode)
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
    });
    
    socket.on('game_state', (state) => {
        updateUIFromState(state);
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from render server');
        updateMessage('Disconnected from render server', 'info');
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
        });
        socket.on('game_state', (state) => {
            updateUIFromState(state);
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

// UI Update functions
function updateUIFromState(state) {
    // Update from Python environment state
    game.playerCards = state.player_cards || [];
    game.dealerCards = state.dealer_cards || [];
    game.playerSum = state.player_sum || 0;
    game.dealerSum = state.dealer_sum || 0;
    game.usableAce = state.usable_ace || false;
    game.done = state.done || false;
    
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
    
    // Update player cards
    const playerCardsContainer = document.getElementById('playerCards');
    playerCardsContainer.innerHTML = '';
    game.playerCards.forEach(card => {
        const cardEl = createCardElement(card, false);
        playerCardsContainer.appendChild(cardEl);
    });
    
    // Update dealer cards
    const dealerCardsContainer = document.getElementById('dealerCards');
    dealerCardsContainer.innerHTML = '';
    game.dealerCards.forEach((card, index) => {
        const isHidden = !game.done && index === 1;
        const cardEl = createCardElement(card, isHidden);
        dealerCardsContainer.appendChild(cardEl);
    });
    
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
    
    // Update buttons (disable in WebSocket mode)
    if (isWebSocketMode) {
        document.getElementById('hitBtn').disabled = true;
        document.getElementById('standBtn').disabled = true;
        document.getElementById('newGameBtn').disabled = true;
    } else {
        document.getElementById('hitBtn').disabled = game.done;
        document.getElementById('standBtn').disabled = game.done;
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
    messageEl.textContent = text;
    messageEl.className = `message ${type}`;
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
    game.reset();
    updateUI();
    updateMessage('New game started! Make your move.', 'info');
}

// Initialize
if (!isWebSocketMode) {
    updateUI();
    updateMessage('Click "New Game" to start playing!', 'info');
} else {
    updateMessage('Connected to training render server. Waiting for game state...', 'info');
}
