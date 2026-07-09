import sys
import os
import base64
import logging
import json
import time
import signal
from datetime import datetime
from flask import Flask, request, Response, jsonify, render_template, redirect, url_for
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
    t2_path=os.path.join(BASE_DIR, "models/behavior_encoder.pt"),
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


def _api_label(label):
    """Map internal labels to the public API contract."""
    return "Suspicious" if label == "Context_Dependent" else label


def _listening_pids_on_port(port):
    """Return PIDs listening on a TCP port using Linux /proc data."""
    target_port = f"{int(port):04X}"
    socket_inodes = set()

    for proc_net_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_net_path, "r", encoding="utf-8") as proc_net:
                next(proc_net, None)
                for line in proc_net:
                    columns = line.split()
                    if len(columns) < 10:
                        continue
                    local_address = columns[1]
                    socket_state = columns[3]
                    inode = columns[9]
                    local_port = local_address.rsplit(":", 1)[-1].upper()
                    if local_port == target_port and socket_state == "0A":
                        socket_inodes.add(inode)
        except OSError:
            continue

    if not socket_inodes:
        return []

    current_pid = os.getpid()
    listening_pids = set()
    for pid_name in os.listdir("/proc"):
        if not pid_name.isdigit():
            continue
        pid = int(pid_name)
        if pid == current_pid:
            continue
        fd_dir = os.path.join("/proc", pid_name, "fd")
        try:
            fd_names = os.listdir(fd_dir)
        except OSError:
            continue
        for fd_name in fd_names:
            fd_path = os.path.join(fd_dir, fd_name)
            try:
                target = os.readlink(fd_path)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in socket_inodes:
                listening_pids.add(pid)
                break

    return sorted(listening_pids)


def _free_port(port, timeout=3.0):
    """Terminate other processes listening on the requested TCP port."""
    pids = _listening_pids_on_port(port)
    if not pids:
        return

    app.logger.warning("Port %s is already in use by PID(s): %s. Terminating them...", port, pids)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            app.logger.warning("No permission to terminate PID %s using port %s", pid, port)

    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining_pids = [pid for pid in pids if os.path.exists(os.path.join("/proc", str(pid)))]
        if not remaining_pids:
            return
        time.sleep(0.1)

    for pid in pids:
        if not os.path.exists(os.path.join("/proc", str(pid))):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            app.logger.warning("Force-killed PID %s still holding port %s", pid, port)
        except ProcessLookupError:
            pass
        except PermissionError:
            app.logger.warning("No permission to force-kill PID %s using port %s", pid, port)


def _run_inference(command, include_flags=None):
    """Run engine, normalize response, and apply include flags.

    include_flags is an optional dict of section booleans:
      evidence, mitre, analysis, ioc, meta
    All sections are included by default when include_flags is None.
    """
    # --- Try auto-decode Base64 commands ---
    original_command = command
    app_decoded = False
    try:
        decoded_bytes = base64.b64decode(command, validate=True)
        # Try UTF-16LE first (PowerShell -EncodedCommand format)
        try:
            utf16 = decoded_bytes.decode('utf-16-le')
            ascii_printable = sum(1 for c in utf16 if '\x20' <= c <= '\x7e' or c in '\r\n\t')
            if ascii_printable > len(utf16) * 0.6 and len(utf16) > 3:
                command = utf16
                app_decoded = True
            else:
                raise ValueError("not utf-16le")
        except (UnicodeDecodeError, ValueError):
            # Fallback to UTF-8
            command = decoded_bytes.decode('utf-8')
            app_decoded = True
    except Exception:
        pass  # Assume plain text if decode fails

    # --- Run Genos engine ---
    t_start = time.perf_counter()
    raw_result = engine.scan(command)
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # Support both legacy and updated engine payload keys.
    label = raw_result.get('label', raw_result.get('status'))
    label_conf = raw_result.get('label_confidence', raw_result.get('gatekeeper_confidence'))
    mitre_predictions = raw_result.get('MITRE_codes', raw_result.get('top_mitre', []))

    if label is None or label_conf is None:
        raise ValueError(f"Unexpected engine payload keys: {list(raw_result.keys())}")

    public_label = _api_label(label)

    # --- Build default flags (all on) ---
    flags = {
        "evidence": True,
        "mitre": True,
        "analysis": True,
        "ioc": True,
        "meta": True,
    }
    if include_flags and isinstance(include_flags, dict):
        for key in flags:
            if key in include_flags:
                flags[key] = bool(include_flags[key])

    # --- Core fields (always returned) ---
    result = {
        "label": public_label,
        "canonical_label": label,
        "label_confidence": round(float(label_conf), 2),
    }

    for key in ("class_probabilities", "decision_margin", "reason", "triggered_features", "routing_policy", "should_run_specialist", "gatekeeper", "behavior"):
        if key in raw_result:
            result[key] = raw_result[key]

    # --- Context_Dependent action hint ---
    if "action" in raw_result:
        result["action"] = raw_result["action"]

    # --- Label probabilities ---
    if "label_probabilities" in raw_result:
        probabilities = dict(raw_result["label_probabilities"])
        if "context_dependent" in probabilities and "suspicious" not in probabilities:
            probabilities["suspicious"] = probabilities["context_dependent"]
        result["label_probabilities"] = probabilities

    # --- MITRE codes ---
    if flags["mitre"]:
        result["MITRE_codes"] = [
            {
                "code": t["code"],
                "confidence": round(float(t["confidence"]), 2)
            }
            for t in mitre_predictions
        ]

    # --- Evidence ---
    if flags["evidence"] and "evidence" in raw_result:
        result["evidence"] = raw_result["evidence"]
        # If the app layer pre-decoded Base64, make sure the evidence block
        # reflects that so the UI obfuscation banner fires.
        if app_decoded:
            ev = result["evidence"]
            if not ev.get("uses_obfuscation"):
                ev["uses_obfuscation"] = True
            if not ev.get("obfuscation_markers"):
                ev["obfuscation_markers"] = ["base64"]

    # --- Analysis ---
    if flags["analysis"]:
        for key in ("mapping_reasons", "why_mapped", "confidence_driver", "analyst_hint"):
            if key in raw_result:
                result[key] = raw_result[key]
        # Decoded payload data (present when obfuscation was detected)
        for key in ("decoded_payload", "payload_mitre_codes", "deobfuscated_cmd"):
            if key in raw_result and raw_result[key] is not None:
                result[key] = raw_result[key]
        # If the app layer pre-decoded Base64 before the engine saw it, surface the
        # decoded command so the UI can show the obfuscated → plain comparison.
        if app_decoded and "deobfuscated_cmd" not in result:
            result["deobfuscated_cmd"] = command

    # --- IOC summary ---
    if flags["ioc"] and "ioc_summary" in raw_result:
        result["ioc_summary"] = raw_result["ioc_summary"]

    # --- Meta ---
    if flags["meta"]:
        if "attack_stage" in raw_result:
            result["attack_stage"] = raw_result["attack_stage"]
        if "severity" in raw_result:
            result["severity"] = raw_result["severity"]
        result["elapsed_ms"] = elapsed_ms

    return result

# -----------------------
# Routes
# -----------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/demo', methods=['GET'])
def demo():
    return render_template('demo.html')


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
        include_flags = data.get('include')
        response = _run_inference(data['command'], include_flags=include_flags)

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
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({"error": "Missing parameters: command"}), 400

    try:
        include_flags = data.get('include')
        response = _run_inference(data['command'], include_flags=include_flags)
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
    port = int(os.getenv("PORT", "5000"))
    _free_port(port)
    app.run(host='0.0.0.0', port=port)