from flask import Flask, request, jsonify
from engine import GenosEngine

app = Flask(__name__)
VALID_API_KEY = "GENOS_DEV_KEY_2026"
engine = GenosEngine()

@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()
    
    if not data or 'command' not in data or 'api_key' not in data:
        return jsonify({"error": "Missing parameters"}), 400
    
    if data['api_key'] != VALID_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    result = engine.scan(data['command'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)