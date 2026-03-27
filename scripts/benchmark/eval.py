import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# Define BASE_DIR to point to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from engine import GenosEngine
except ImportError:
    print("[!] Warning: Could not import GenosEngine. Make sure engine.py is in the root directory.")


def run_offline_audit(sample_size=1000):
    engine = GenosEngine() # Automatically loads models, maps, and max_length=256
    
    # Use the balanced datasets we just created for accurate evaluation!
    try:
        b_df = pd.read_csv(os.path.join(BASE_DIR, "data", "training", "genos-balanced-good.csv")).sample(sample_size, replace=True)
        m_df = pd.read_csv(os.path.join(BASE_DIR, "data", "training", "genos-balanced-bad.csv"))
    except FileNotFoundError:
        # Fallback to archive if balanced data isn't moved yet
        b_df = pd.read_csv(os.path.join(BASE_DIR, "data", "archive", "benign_final.csv")).sample(sample_size)
        m_df = pd.read_csv(os.path.join(BASE_DIR, "data", "archive", "mitre_atlas_raw.csv"))

    print("\n" + "=" * 50 + "\nOFFLINE PIPELINE AUDIT\n" + "=" * 50)

    fps = 0
    for cmd in tqdm(b_df["command"], desc="Checking False Positives"):
        result = engine.scan(str(cmd))
        if result["label"] == "Malicious":
            fps += 1

    hits, exact = 0, 0
    for _, row in tqdm(m_df.iterrows(), total=len(m_df), desc="Checking Attack Attribution"):
        result = engine.scan(str(row["command"]))
        if result["label"] == "Malicious":
            hits += 1
            # Check if the Specialist caught the exact MITRE code
            if result.get("MITRE_codes") and result["MITRE_codes"][0]["code"] == row["mitre_id"]:
                exact += 1

    print("\n" + "=" * 50)
    print("OFFLINE RESULTS")
    print("=" * 50)
    print(f"False Positive Rate   : {(fps / len(b_df)) * 100:.2f}%")
    print(f"Overall Detection Rate: {(hits / len(m_df)) * 100:.2f}%")
    print(f"MITRE Match Accuracy  : {(exact / hits) * 100:.2f}%" if hits else "MITRE Match Accuracy  : 0.00%")
    print("=" * 50)


def run_fuzzing_audit():
    engine = GenosEngine()

    try:
        from scripts.data_augmentation.augment import CommandFuzzer
    except ImportError as exc:
        raise RuntimeError(
            "CommandFuzzer is not available. Ensure scripts/data_augmentation/augment.py "
            "defines CommandFuzzer."
        ) from exc

    fuzzer = CommandFuzzer()

    # Test the fuzzer against the balanced malicious dataset
    try:
        m_df = pd.read_csv(os.path.join(BASE_DIR, "data", "training", "genos-balanced-bad.csv"))
    except FileNotFoundError:
        m_df = pd.read_csv(os.path.join(BASE_DIR, "data", "archive", "mitre_atlas_raw.csv"))
        
    hits, exact = 0, 0

    print("\n" + "!" * 50 + "\nFUZZING RESILIENCE AUDIT\n" + "!" * 50)

    for _, row in tqdm(m_df.iterrows(), total=len(m_df), desc="Attacking the Engine"):
        # 1. Apply Obfuscation
        fuzzed_cmd = fuzzer.apply_cocktail(str(row["command"]))
        actual_mitre = row["mitre_id"]

        # 2. Let the Engine peel it and scan it
        result = engine.scan(fuzzed_cmd)
        
        if result["label"] == "Malicious":
            hits += 1
            if result.get("MITRE_codes") and result["MITRE_codes"][0]["code"] == actual_mitre:
                exact += 1

    print("\n" + "=" * 50)
    print("FUZZING RESULTS")
    print("=" * 50)
    print(f"Overall Detection   : {(hits / len(m_df)) * 100:.2f}%")
    print(f"MITRE Match Accuracy: {(exact / hits) * 100:.2f}%" if hits else "MITRE Match Accuracy: 0.00%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Run GENOS benchmark audits")
    parser.add_argument(
        "--mode",
        choices=["offline", "fuzzing", "both"],
        default="offline",
        help="Which audit(s) to run",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size for benign dataset in offline audit",
    )
    args = parser.parse_args()

    if args.mode in ("offline", "both"):
        run_offline_audit(sample_size=args.sample_size)

    if args.mode in ("fuzzing", "both"):
        run_fuzzing_audit()


if __name__ == "__main__":
    main()