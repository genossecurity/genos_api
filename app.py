import sys
import os
import base64
import logging
import json
from flask import Flask, request, Response
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
    t2_path=os.path.join(BASE_DIR, "models/specialist.pt"),
    map_path=os.path.join(BASE_DIR, "models/specialist_map.json")
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
    if not data or 'api_key' not in data or 'command' not in data:
        return Response(
            json.dumps({"error": "Missing parameters"}, indent=2),
            mimetype='application/json',
            status=400
        )

    # Validate API key
    user_record = keys_collection.find_one({"key": data['api_key']})
    if not user_record:
        return Response(
            json.dumps({"error": "Key Not Valid"}, indent=2),
            mimetype='application/json',
            status=401
        )

    command = safe_base64_decode(data['command'])

    try:
        # Run Genos engine
        raw_result = engine.scan(command)

        # Build pretty response
        response = {
            "label": raw_result['status'],
            "label_confidence": round(raw_result['gatekeeper_confidence'], 4),
            "MITRE_codes": [
                {"code": t["code"], "confidence": round(t["confidence"], 4)}
                for t in raw_result.get('top_mitre', [])
            ]
        }

        # Log usage
        usage_collection.update_one(
            {"user_id": user_record.get('user_id'), "api_key": data['api_key']},
            {
                "$inc": {"req_count": 1},
                "$set": {"updated_at": os.getenv("CURRENT_TIME", "2026-03-17T00:00:00.000+00:00")}
            }
        )

        return Response(
            json.dumps(response, indent=2),
            mimetype='application/json'
        )

    except Exception as e:
        app.logger.error(f"Genos Engine Error: {str(e)}")
        return Response(
            json.dumps({"error": "Internal server error"}, indent=2),
            mimetype='application/json',
            status=500
        )

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)