import sys
import os
import time
from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
'''
TODO:
1. Check if key is valid
2. Check if limit is reached
3. Decode base64 if base64
'''
# Load environment variables
load_dotenv()

# Force Python to look in current directory for local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import GenosEngine

app = Flask(__name__)

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI") # Loaded from .env
client = MongoClient(MONGO_URI)
db = client.get_database() # Uses database from URI (genos_db) or specify one
keys_collection = db['api_keys']  # Collection name

# Initialize Engine
engine = GenosEngine(
    t1_path="./models/gatekeeper.pt", 
    t2_path="./models/specialist.pt", 
    map_path="./models/specialist_map.json"
)

def is_valid_key(provided_key):
    """
    Checks MongoDB for the key. 
    Assumes documents look like: {"key": "XYZ", "active": True}
    """
    if not provided_key:
        return False
    
    # Query for the key and ensure it is marked as active
    user_key = keys_collection.find_one({"key": provided_key, "active": True})
    return user_key is not None

@app.route('/scan', methods=['POST'])
def scan():
    start_time = time.perf_counter()  # Fixed: Added missing start_time
    data = request.get_json()
    
    # 1. Validation & Auth
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    
    if 'command' not in data or 'api_key' not in data:
        return jsonify({"error": "Missing 'command' or 'api_key'"}), 400
    
    # MongoDB lookup instead of hardcoded string
    if not is_valid_key(data['api_key']):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # 2. Neural Analysis
        raw_result = engine.scan(data['command'])
        
        # 3. Stop timer and calculate milliseconds
        end_time = time.perf_counter()
        inference_ms = (end_time - start_time) * 1000
        
        # 4. Logic 
        response = {
            "verdict": raw_result['status'],
            "confidence": raw_result['gatekeeper_confidence'],
            "latency_ms": round(inference_ms, 2)
        }

        if raw_result['status'] == "Malicious":
            response["mitre_id"] = raw_result.get('mitre_id')
            
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Genos Engine Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)