import sys
import os
import time  # <--- Added import
from flask import Flask, request, jsonify

# Force Python to look in current directory for local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import GenosEngine

app = Flask(__name__)
VALID_API_KEY = "GENOS_DEV_KEY_2026"

# Initialize Engines with local model paths
engine = GenosEngine(
    t1_path="./models/gatekeeper.pt", 
    t2_path="./models/specialist.pt", 
    map_path="./models/specialist_map.json"
)

@app.route('/scan', methods=['POST'])
def scan():
    # 1. Start the high-resolution timer
    start_time = time.perf_counter()
    
    data = request.get_json()
    
    # Validation & Auth
    if not data or 'command' not in data or 'api_key' not in data:
        return jsonify({"error": "Missing parameters"}), 400
    
    if data['api_key'] != VALID_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # 2. Neural Analysis
        raw_result = engine.scan(data['command'])
        
        # 3. Stop timer and calculate milliseconds
        end_time = time.perf_counter()
        inference_ms = (end_time - start_time) * 1000
        
        # 4. Clean Unified Logic (Now with latency injected)
        if raw_result['status'] == "Malicious":
            response = {
                "verdict": "Malicious",
                "mitre_id": raw_result['mitre_id'],
                "confidence": raw_result['gatekeeper_confidence'],
                "server_inference_ms": round(inference_ms, 2) # <--- Added
            }
        else:
            response = {
                "verdict": "Benign",
                "confidence": raw_result['gatekeeper_confidence'],
                "server_inference_ms": round(inference_ms, 2) # <--- Added
            }
            
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Genos Engine Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Production deployment uses Gunicorn, this is for local testing
    app.run(host='0.0.0.0', port=5000)