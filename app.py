import os
import sys
import json
import time
import signal
import logging
from threading import Lock
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from genos.engine import GenosEngine
from genos.specialist import FAMILY_LABELS

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# Flask Application Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_inference_lock = Lock()

# Public GET API names map to the engine's evidence fields.
IOC_FIELDS = {
    "urls": "urls",
    "domains": "domains",
    "ips": "ips",
    "ports": "ports",
    "files": "file_paths",
    "registry": "registry_paths",
}


# Load both model weights and the warm-up inference before routes can serve a page.
app.logger.info("Initializing Genos Engine...")
engine = GenosEngine(
    t1_path=os.path.join(BASE_DIR, "models/gatekeeper.pt"),
    t2_path=os.path.join(BASE_DIR, "models/behavior_encoder.pt"),
)
app.logger.info("Running warm-up inference pass...")
engine.scan("warmup")
app.logger.info("Genos Engine ready (11 MITRE tactic families active).")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _api_label(label: str) -> str:
    """Map internal labels to public contract."""
    return "Context_Dependent" if label == "Suspicious" else label


def _listening_pids_on_port(port: int) -> list[int]:
    """Identify PIDs holding a TCP port using Linux /proc filesystem."""
    target_port = f"{int(port):04X}"
    socket_inodes = set()

    for proc_net_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_net_path, "r", encoding="utf-8") as proc_net:
                next(proc_net, None)
                for line in proc_net:
                    cols = line.split()
                    if len(cols) < 10:
                        continue
                    local_address, state, inode = cols[1], cols[3], cols[9]
                    if local_address.rsplit(":", 1)[-1].upper() == target_port and state == "0A":
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
            for fd_name in os.listdir(fd_dir):
                target = os.readlink(os.path.join(fd_dir, fd_name))
                if target.startswith("socket:[") and target[8:-1] in socket_inodes:
                    listening_pids.add(pid)
                    break
        except OSError:
            continue

    return sorted(listening_pids)


def _free_port(port: int, timeout: float = 3.0) -> None:
    """Terminate other processes holding the requested TCP port."""
    pids = _listening_pids_on_port(port)
    if not pids:
        return

    app.logger.warning("Port %s in use by PID(s): %s. Terminating...", port, pids)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = [p for p in pids if os.path.exists(os.path.join("/proc", str(p)))]
        if not remaining:
            return
        time.sleep(0.1)

    for pid in pids:
        if os.path.exists(os.path.join("/proc", str(pid))):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _scan_with_gpu_memory(command: str) -> tuple[dict, dict]:
    """Execute scan while profiling CUDA memory allocations."""
    scan_engine = engine
    import torch

    with _inference_lock:
        device = scan_engine.device
        memory = {
            "status": "cpu" if device.type != "cuda" else "unavailable",
            "device_name": None,
            "baseline_allocated_bytes": None,
            "peak_allocated_bytes": None,
            "command_peak_bytes": None,
            "peak_reserved_bytes": None,
        }
        baseline = None
        if device.type == "cuda":
            try:
                torch.cuda.synchronize(device)
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            except (RuntimeError, AssertionError):
                baseline = None

        result = scan_engine.scan(command)

        if baseline is not None:
            try:
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
                name = torch.cuda.get_device_name(device)
                memory.update({
                    "status": "measured",
                    "device_name": name,
                    "baseline_allocated_bytes": baseline,
                    "peak_allocated_bytes": peak,
                    "command_peak_bytes": max(0, peak - baseline),
                    "peak_reserved_bytes": reserved,
                })
            except (RuntimeError, AssertionError):
                pass

        return result, memory


def _run_inference(command: str, include_flags: dict | None = None) -> dict:
    """Run full Genos inspection pipeline and structure the client payload."""
    t_start = time.perf_counter()
    raw_result, gpu_memory = _scan_with_gpu_memory(command)
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    label = raw_result.get("label", raw_result.get("status"))
    label_conf = raw_result.get("label_confidence", raw_result.get("gatekeeper_confidence", 0.0))
    if label is None or label_conf is None:
        raise ValueError(f"Unexpected engine payload keys: {list(raw_result.keys())}")

    public_label = _api_label(label)

    flags = {
        "evidence": True,
        "mitre": True,
        "families": True,
        "analysis": True,
        "ioc": True,
        "meta": True,
    }
    if include_flags and isinstance(include_flags, dict):
        flags.update({k: bool(v) for k, v in include_flags.items() if k in flags})

    # Core identification fields
    result = {
        "label": public_label,
        "canonical_label": label,
        "label_confidence": round(float(label_conf), 2),
    }

    # Pass-through metadata
    for key in (
        "class_probabilities", "decision_margin", "reason", "triggered_features",
        "routing_policy", "should_run_specialist", "gatekeeper", "behavior",
        "provenance", "score_type", "calibration", "input_truncated",
        "mitre_scope", "specialist_mode", "baseline_status", "seen_count",
    ):
        if key in raw_result:
            result[key] = raw_result[key]

    if "action" in raw_result:
        result["action"] = raw_result["action"]

    # Normalized label probabilities
    if "label_probabilities" in raw_result:
        probs = dict(raw_result["label_probabilities"])
        if "suspicious" in probs:
            probs.setdefault("context_dependent", probs.pop("suspicious"))
        result["label_probabilities"] = probs

    # 11 MITRE Tactic Families
    if flags["families"] and "attack_families" in raw_result:
        result["attack_families"] = raw_result["attack_families"]

    # Legacy technique ranking (active only in GENOS_SPECIALIST_MODE=mitre)
    if flags["mitre"] and "MITRE_codes" in raw_result:
        result["MITRE_codes"] = [
            {
                "code": t["code"],
                "confidence": round(float(t["confidence"]), 2),
                "score_type": t.get("score_type", "uncalibrated_model_estimate"),
            }
            for t in raw_result.get("MITRE_codes", [])
        ]

    # Observable evidence
    if flags["evidence"] and "evidence" in raw_result:
        result["evidence"] = raw_result["evidence"]

    # Analyst explanations and decoded payloads
    if flags["analysis"]:
        for key in ("analyst_hint", "confidence_driver", "decoded_payload", "deobfuscated_cmd", "mapping_reasons", "why_mapped"):
            if key in raw_result and raw_result[key] is not None:
                result[key] = raw_result[key]

    # Network & host IOC summary
    if flags["ioc"] and "ioc_summary" in raw_result:
        result["ioc_summary"] = raw_result["ioc_summary"]

    # Execution telemetry
    if flags["meta"]:
        if "attack_stage" in raw_result:
            result["attack_stage"] = raw_result["attack_stage"]
        if "severity" in raw_result:
            result["severity"] = raw_result["severity"]
        result["elapsed_ms"] = elapsed_ms
        result["gpu_memory"] = gpu_memory

    return result


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Main web scanner interface."""
    return render_template("index.html")


@app.route("/api", methods=["GET"])
def api_builder():
    """Interactive GET request builder."""
    return render_template("api.html", ioc_fields=IOC_FIELDS)


def _get_api_response(payload: dict, status: int = 200):
    response = jsonify(payload)
    response.status_code = status
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/scan", methods=["GET"])
def scan_get():
    """Tier 1 by default; opt into Tier 2 and selected IOC fields."""
    allowed = {"command", "tier2", "iocs"}
    unknown = set(request.args) - allowed
    if unknown:
        return _get_api_response({"error": "Unknown parameter(s): " + ", ".join(sorted(unknown))}, 400)
    if any(len(request.args.getlist(key)) != 1 for key in request.args):
        return _get_api_response({"error": "Each parameter must be supplied only once"}, 400)

    command = request.args.get("command", "").strip()
    if not command:
        return _get_api_response({"error": "Command parameter is required and cannot be empty"}, 400)

    tier2_value = request.args.get("tier2", "false").lower()
    if tier2_value not in {"true", "false", "1", "0"}:
        return _get_api_response({"error": "tier2 must be true or false (1 or 0 also accepted)"}, 400)
    tier2 = tier2_value in {"true", "1"}

    selected_iocs = []
    if "iocs" in request.args:
        ioc_value = request.args["iocs"].strip().lower()
        selected_iocs = list(IOC_FIELDS) if ioc_value == "all" else list(dict.fromkeys(
            field.strip() for field in ioc_value.split(",")
        ))
        if any(field not in IOC_FIELDS for field in selected_iocs):
            return _get_api_response({
                "error": "iocs must be all or a comma-separated list of: " + ", ".join(IOC_FIELDS)
            }, 400)

    try:
        scan_engine = engine
        with _inference_lock:
            raw_result = scan_engine.scan(
                command, run_specialist=tier2, collect_iocs=bool(selected_iocs),
                use_baseline=False,
            )
        result = {
            "label": _api_label(raw_result["label"]),
            "label_confidence": round(float(raw_result["label_confidence"]), 2),
            "deobfuscated_cmd": raw_result.get("deobfuscated_cmd"),
        }
        if tier2:
            behavior = raw_result.get("behavior") or {}
            completed = bool(raw_result.get("should_run_specialist"))
            result["tier2"] = {
                "status": "completed" if completed else (
                    "skipped_benign" if raw_result["label"] == "Benign" else "unavailable"
                ),
                "stage": behavior.get("stage") if completed else None,
                "stage_confidence": behavior.get("stage_confidence") if completed else None,
            }
        if selected_iocs:
            evidence = raw_result.get("evidence") or {}
            result["iocs"] = {
                field: evidence.get(IOC_FIELDS[field], []) for field in selected_iocs
            }
        return _get_api_response(result)
    except Exception:
        app.logger.exception("Inference error during GET command scan")
        return _get_api_response({"error": "Inference failed"}, 500)

@app.route("/health", methods=["GET"])
def health():
    """Health check reporting engine readiness and active specialist heads."""
    return jsonify({
        "status": "ok",
        "engine_ready": True,
        "specialist_mode": engine.specialist_mode,
        "supported_families": len(engine.family_labels),
        "device": engine.device.type,
    })


@app.route("/api/families", methods=["GET"])
def families():
    """Return the 11 MITRE ATT&CK tactic heads/families."""
    family_labels = engine.family_labels
    return jsonify({
        "count": len(family_labels),
        "families": family_labels,
    })


@app.route("/scan", methods=["POST"])
@app.route("/api/scan", methods=["POST"])
def scan():
    """Execute command security scan (JSON or form POST)."""
    command = None
    include_flags = None

    if request.is_json:
        data = request.get_json(silent=True) or {}
        command = data.get("command")
        include_flags = data.get("include") or data.get("include_flags")
    else:
        command = request.form.get("command")

    if not command or not isinstance(command, str) or not command.strip():
        return jsonify({"error": "Command parameter is required and cannot be empty"}), 400

    try:
        result = _run_inference(command.strip(), include_flags=include_flags)
        return jsonify(result), 200
    except Exception as exc:
        app.logger.exception("Inference error during command scan")
        return jsonify({"error": "Inference failed", "details": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "6001"))
    host = os.getenv("HOST", "0.0.0.0")
    _free_port(port)
    app.run(host=host, port=port, debug=False)
