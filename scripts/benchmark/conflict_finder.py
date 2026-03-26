import pandas as pd
import os

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


def resolve_archive_path(filename: str) -> str:
    """Resolve archive data path."""
    candidates = [
        os.path.join(BASE_DIR, "data", "archive", "art", filename),
        os.path.join(BASE_DIR, "data", "archive", filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} in archive: " + ", ".join(candidates)
    )


TRAINING_DATA = resolve_data_path("trainer1-bad.csv")
ATLAS_RAW = resolve_archive_path("mitre_atlas_raw.csv")

def find_missing_atlas_labels():
    # Load your current training labels
    train_df = pd.read_csv(TRAINING_DATA)
    # Get unique labels from your current dataset (likely mixture of Txxxx and AML.Txxxx)
    current_labels = set(train_df['mitre_id'].unique())

    # Load the official ATLAS technique list
    atlas_df = pd.read_csv(ATLAS_RAW)
    
    # ATLAS_RAW has columns: mitre_id, technique_name
    all_atlas_ids = set(atlas_df['mitre_id'].unique())
    
    missing_from_training = all_atlas_ids - current_labels
    coverage_pct = (len(all_atlas_ids) - len(missing_from_training)) / len(all_atlas_ids) * 100

    print("=" * 50)
    print(f"ATLAS COVERAGE ANALYSIS")
    print("=" * 50)
    print(f"[*] Total ATLAS Techniques: {len(all_atlas_ids)}")
    print(f"[*] Current Coverage: {coverage_pct:.2f}%")
    print(f"[*] Missing IDs: {len(missing_from_training)}")
    print("-" * 50)

    if missing_from_training:
        print("TOP MISSING ATLAS TECHNIQUES:")
        # Join with atlas_df to get the names for better visibility
        missing_details = atlas_df[atlas_df['mitre_id'].isin(missing_from_training)]
        for _, row in missing_details.head(15).iterrows():
            print(f"- {row['mitre_id']}: {row['technique_name']}")
    else:
        print("[+] Perfect Coverage! All ATLAS techniques are represented.")

if __name__ == "__main__":
    try:
        find_missing_atlas_labels()
    except FileNotFoundError as e:
        print(f"[!] {e}")