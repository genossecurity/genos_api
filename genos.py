import sys
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

        res = engine.scan(cmd)
        
        if res['status'] == "Benign":
            color = "\033[92m" # Green
            print(f"{color}[SAFE]{'\033[0m'} | Conf: {res['confidence']}")
        else:
            color = "\033[91m" # Red
            print(f"{color}[MALICIOUS]{'\033[0m'} | MITRE: {res['mitre_id']} | Conf: {res['confidence']}")
        print("-" * 50)

if __name__ == "__main__":
    run_shell()