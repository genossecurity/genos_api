# Flask Server

A simple Flask server with basic routes.

## Setup

The Flask server requires a Python virtual environment. One is already set up at `../my_flask_env/`.

## Running the Server

1. Activate the virtual environment:
   ```bash
   source ../my_flask_env/bin/activate
   ```

2. Run the Flask server:
   ```bash
   python app.py
   ```

The server will start on `http://0.0.0.0:5000`

## Endpoints

- `GET /` - Returns a welcome message
- `GET /api/status` - Returns server status
- `GET /api/test` - Returns a test response

## Development

The server runs in debug mode, which means:
- Auto-reloads when code changes
- Enhanced error pages
- Interactive debugger enabled
