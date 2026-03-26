import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from engine import GenosEngine

def run_shell():
    print("\n🛡️  GENOS GUARDIAN LOCAL SHELL v1.0")
    print("Loading Neural Engines...")
    engine = GenosEngine()
    print("Ready. Type 'exit' to quit.\n")

    while True:
        cmd = input("GENOS > ")
        if cmd.lower() in ['exit', 'quit']: break
        if not cmd.strip(): continue
        reset = "\033[0m"

        res = engine.scan(cmd)
        
        if res['status'] == "Benign":
            color = "\033[92m" # Green
            print(f"{color}[SAFE]{reset} | Gatekeeper: {res['gatekeeper_confidence']:.4f}")
        else:
            color = "\033[91m" # Red
            top = res.get('top_mitre', [])
            top_code = top[0]['code'] if top else 'N/A'
            top_conf = top[0]['confidence'] if top else 0.0
            print(f"{color}[MALICIOUS]{reset} | MITRE: {top_code} ({top_conf:.4f}) | Gatekeeper: {res['gatekeeper_confidence']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    run_shell()