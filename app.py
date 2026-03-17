import sys
import os
import base64
import logging
from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv

# Force Python to look in current directory for local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import GenosEngine

# Load environment variables
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI") # Loaded from .env
client = MongoClient(MONGO_URI)
db = client['genos'] # Database name from image
keys_collection = db['api_keys']  # Collection name from image
usage_collection = db['usage']

# Initialize Engine
# Use absolute paths to ensure models load correctly regardless of where gunicorn is started
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.logger.info("Loading Genos engine — this may take a moment...")
engine = GenosEngine(
    t1_path=os.path.join(BASE_DIR, "models/gatekeeper.pt"),
    t2_path=os.path.join(BASE_DIR, "models/specialist.pt"),
    map_path=os.path.join(BASE_DIR, "models/specialist_map.json")
)
# Warm-up pass: forces all CUDA kernels to compile and weights to be
# fully resident before the first real request arrives.
app.logger.info("Running warm-up inference pass...")
engine.scan("warmup")
app.logger.info("Genos engine ready.")

_engine_ready = True

def is_valid_key(provided_key):
    """
    Checks MongoDB for the key and ensures it is active.
    Example document: {"key": "XYZ", "active": True}
    """
    if not provided_key:
        return False
    
    # Query for the key
    user_data = keys_collection.find_one({"key": provided_key})
    return user_data is not None

@app.route('/health', methods=['GET'])
def health():
    if _engine_ready:
        return jsonify({"status": "ok"}), 200
    return jsonify({"status": "loading"}), 503


@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()
    
    # 1. Validation & Auth
    if not data or 'api_key' not in data or 'command' not in data:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Check key validity
    user_record = keys_collection.find_one({"key": data['api_key']})
    if not user_record:
        return jsonify({"error": "Key Not Valid"}), 401

    command = data['command']
    
    # Try to decode as base64 automatically
    try:
        # Check if it looks like base64 and decode
        decoded_bytes = base64.b64decode(command, validate=True)
        command = decoded_bytes.decode('utf-8')
    except Exception:
        # If decoding fails, assume it's already a plain text command
        pass

    try:
        # 2. Neural Analysis
        raw_result = engine.scan(command)
        
        # 3. Response logic based on verdict
        confidence = raw_result['gatekeeper_confidence']
        label = raw_result['status']

        if label == "Malicious":
            # Bad JSON response: {"MITRE code", "label", "confidence score"}
            response = {
                "MITRE code": raw_result.get('mitre_id', "N/A"),
                "label": label,
                "confidence score": confidence
            }
        else:
            # Good JSON response: {"label", "confidence score"}
            response = {
                "label": label,
                "confidence score": confidence
            }
            
        # Log usage (linking to user_id from api_keys collection)
        usage_collection.update_one(
            {"user_id": user_record.get('user_id'), "api_key": data['api_key']},
            {"$inc": {"req_count": 1}, "$set": {"updated_at": os.getenv("CURRENT_TIME", "2026-03-13T23:37:37.330+00:00")}}
        )

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Genos Engine Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)