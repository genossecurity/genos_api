import sys
import os
import base64
import logging
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv

# Add current directory to module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine import GenosEngine

# Load environment variables
load_dotenv()

# -----------------------
# Flask App Initialization
# -----------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------
# MongoDB Configuration
# -----------------------
MONGO_URI = os.getenv("MONGO_URI")
client = None
db = None
keys_collection = None
usage_collection = None
if MONGO_URI:
    client = MongoClient(MONGO_URI)
    db = client['genos']
    keys_collection = db['api_keys']
    usage_collection = db['usage']

# -----------------------
# Genos Engine Initialization
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.logger.info("Loading Genos engine — this may take a moment...")
engine = GenosEngine(
    t1_path=os.path.join(BASE_DIR, "models/gatekeeper.pt"),
    t2_path=os.path.join(BASE_DIR, "models/specialist_residual_a.pt"),
)

# Warm-up inference
app.logger.info("Running warm-up pass...")
engine.scan("warmup")
app.logger.info("Genos engine ready.")

_engine_ready = True

# -----------------------
# Helper Functions
# -----------------------
def is_valid_key(api_key):
    """Check if the API key exists in MongoDB."""
    if not api_key:
        return False
    return keys_collection.find_one({"key": api_key}) is not None

def safe_base64_decode(command):
    """Attempt to decode Base64; fallback to plain text."""
    try:
        decoded_bytes = base64.b64decode(command, validate=True)
        decoded_text = decoded_bytes.decode('utf-8')
        if decoded_text.strip():
            return decoded_text
    except Exception:
        pass
    return command


def _to_percentage(value):
    """Normalize confidence value to percentage with 2 decimals.

    Accepts either raw probability (0-1) or percentage (>1).
    """
    value = float(value)
    if value <= 1.0:
        value *= 100.0
    return round(value, 2)


def _run_inference(command):
    """Run engine and normalize response payload schema."""
    # --- Try auto-decode Base64 commands ---
    try:
        decoded_bytes = base64.b64decode(command, validate=True)
        command = decoded_bytes.decode('utf-8')
    except Exception:
        pass  # Assume plain text if decode fails

    # --- Run Genos engine ---
    raw_result = engine.scan(command)

    # Support both legacy and updated engine payload keys.
    label = raw_result.get('label', raw_result.get('status'))
    label_conf = raw_result.get('label_confidence', raw_result.get('gatekeeper_confidence'))
    mitre_predictions = raw_result.get('MITRE_codes', raw_result.get('top_mitre', []))

    if label is None or label_conf is None:
        raise ValueError(f"Unexpected engine payload keys: {list(raw_result.keys())}")

    return {
        "label": label,
        # engine.scan() already returns percentages (multiplied by 100); pass through directly.
        "label_confidence": round(float(label_conf), 2),
        "MITRE_codes": [
            {
                "code": t["code"],
                "confidence": round(float(t["confidence"]), 2)
            }
            for t in mitre_predictions
        ]
    }

# -----------------------
# Routes
# -----------------------
@app.route('/health', methods=['GET'])
def health():
    status = "ok" if _engine_ready else "loading"
    return Response(
        json.dumps({"status": status}, indent=2),
        mimetype='application/json'
    )
@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()

    # --- Validate input ---
    if not data:
        return jsonify({"error": "Missing parameters: request body"}), 400
    
    missing_params = []
    if 'api_key' not in data:
        missing_params.append('api_key')
    if 'command' not in data:
        missing_params.append('command')
    
    if missing_params:
        return jsonify({"error": f"Missing parameters: {', '.join(missing_params)}"}), 400

    if keys_collection is None or usage_collection is None:
        return jsonify({"error": "Internal error"}), 400

    # --- Validate API key ---
    user_record = keys_collection.find_one({"key": data['api_key']})
    if not user_record:
        return jsonify({"error": "Key Not Valid"}), 401

    try:
        response = _run_inference(data['command'])

        # --- Log usage ---
        usage_collection.update_one(
            {"user_id": user_record.get('user_id'), "api_key": data['api_key']},
            {
                "$inc": {"req_count": 1},
                "$set": {"updated_at": datetime.utcnow().isoformat()}
            }
        )

        return app.response_class(
            response=json.dumps(response, indent=2),  # Pretty print
            mimetype='application/json'
        )

    except Exception as e:
        app.logger.error(f"Genos Engine Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/scan/free', methods=['POST'])
def scan_free():
    """Run inference without API key validation or usage logging.

    Intended for the public GUI dashboard which enforces its own
    IP-based rate limit (500/day) on the dashboard server side.
    Only accepts requests from localhost.
    """
    # Restrict to loopback so only the dashboard server can call this
    if request.remote_addr not in ('127.0.0.1', '::1'):
        return jsonify({"error": "Forbidden"}), 403

    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({"error": "Missing parameters: command"}), 400

    try:
        response = _run_inference(data['command'])
        return app.response_class(
            response=json.dumps(response, indent=2),
            mimetype='application/json'
        )
    except Exception as e:
        app.logger.error(f"Genos Engine Error (free): {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)