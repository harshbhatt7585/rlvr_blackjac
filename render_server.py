"""
Web server for rendering Blackjack environment during training.
Provides WebSocket connection for real-time game state updates.
"""
import os
import json
import logging
import datetime
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from threading import Lock

app = Flask(__name__, static_folder='frontend')
app.config['SECRET_KEY'] = 'blackjack-render-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Thread-safe state management
render_lock = Lock()
current_state = None
render_enabled = False

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory."""
    return send_from_directory('frontend', path)

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("Client connected for rendering")
    emit('connected', {'status': 'connected'})
    
    # Send current state if available
    with render_lock:
        if current_state is not None:
            emit('game_state', current_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")

def update_game_state(state):
    """
    Update the current game state and broadcast to all connected clients.
    
    Args:
        state: Dictionary containing game state information
    """
    global current_state
    with render_lock:
        current_state = state
        socketio.emit('game_state', state)
        socketio.sleep(0)  # Allow other events to be processed

def enable_rendering():
    """Enable rendering mode."""
    global render_enabled
    render_enabled = True
    print("Rendering enabled - WebSocket server ready")

def disable_rendering():
    """Disable rendering mode."""
    global render_enabled
    render_enabled = False

def is_rendering_enabled():
    """Check if rendering is enabled."""
    return render_enabled

def send_log(message, level='info'):
    """
    Send a log message to all connected clients.
    
    Args:
        message: Log message string
        level: Log level (info, warning, error, debug)
    """
    try:
        log_data = {
            'message': message,
            'level': level,
            'timestamp': datetime.datetime.now().isoformat()
        }
        # Emit to all clients (broadcasts by default when no room is specified)
        socketio.emit('log', log_data)
    except Exception as e:
        # Fallback to print if emit fails
        print(f"Failed to send log via WebSocket: {e}")

class WebSocketLogHandler(logging.Handler):
    """Custom logging handler that sends logs to WebSocket clients."""
    
    def emit(self, record):
        """Emit a log record to WebSocket clients."""
        try:
            log_message = self.format(record)
            level = record.levelname.lower()
            if level not in ['info', 'warning', 'error', 'debug']:
                level = 'info'
            send_log(log_message, level)
        except Exception:
            # Ignore errors in logging handler to prevent recursion
            pass

class PrintCapture:
    """Capture print statements and send them to WebSocket clients."""
    
    def __init__(self, original_print):
        self.original_print = original_print
        self.buffer = []
    
    def __call__(self, *args, **kwargs):
        # Call original print
        self.original_print(*args, **kwargs)
        
        # Capture and send to WebSocket
        try:
            message = ' '.join(str(arg) for arg in args)
            if message.strip():  # Only send non-empty messages
                send_log(message, 'info')
        except Exception:
            pass  # Ignore errors to prevent recursion

def setup_logging_handler():
    """Set up the WebSocket logging handler."""
    handler = WebSocketLogHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    handler.setLevel(logging.INFO)
    return handler

def setup_print_capture():
    """Set up print statement capture."""
    import builtins
    original_print = builtins.print
    builtins.print = PrintCapture(original_print)
    return original_print

def run_server(host='127.0.0.1', port=5000, debug=False):
    """Run the Flask-SocketIO server."""
    print(f"Starting render server on http://{host}:{port}")
    print(f"Open http://{host}:{port} in your browser to view the game")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    enable_rendering()
    run_server(debug=True)

