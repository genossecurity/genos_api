import pandas as pd
import os

# --- PATH CONFIGURATION ---
# Based on your Genos directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_data_path(filename: str) -> str:
    """Resolve training data path with fallback to archive."""
    candidates = [
        os.path.join(BASE_DIR, "data", "training", filename),
        os.path.join(BASE_DIR, "data", "archive", "training", filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} in expected locations: " + ", ".join(candidates)
    )


TRAIN_DATA_PATH = resolve_data_path("trainer1-bad.csv")

def inspect_labels():
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"[!] File not found: {TRAIN_DATA_PATH}")
        return

    df = pd.read_csv(TRAIN_DATA_PATH)
    
    print("="*60)
    print("GENOS LABEL REVEAL: trainer1-bad.csv")
    print("="*60)
    print(f"[*] Total Rows: {len(df)}")
    
    # 1. Show the Distribution of IDs
    unique_ids = df['mitre_id'].unique()
    print(f"[*] Total Unique IDs: {len(unique_ids)}")
    
    # 2. Specifically look for ATLAS/AML prefixes
    # We check for 'AML', 'T', and any numeric-only entries
    aml_samples = df[df['mitre_id'].str.contains('AML', na=False, case=False)]
    t_samples = df[df['mitre_id'].str.contains('^T[0-9]', na=False, case=False)]
    
    print("-" * 60)
    print(f"[+] ATLAS (AML) Samples Found: {len(aml_samples)}")
    print(f"[+] Enterprise (T-prefix) Samples Found: {len(t_samples)}")
    
    # 3. List actual unique AML IDs if they exist
    if not aml_samples.empty:
        print("\n--- UNIQUE ATLAS IDs PRESENT ---")
        print(aml_samples['mitre_id'].unique())
    else:
        print("\n[!] WARNING: No AML/ATLAS IDs detected in 'mitre_id' column.")
        print("Checking first 5 raw IDs in file for format check:")
        print(df['mitre_id'].head(5).tolist())

    print("-" * 60)
    print("TOP 10 MOST FREQUENT IDs:")
    print(df['mitre_id'].value_counts().head(10))

if __name__ == "__main__":
    inspect_labels()