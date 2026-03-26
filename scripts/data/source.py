import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Force a strictly sorted, unique list of MITRE IDs
df = pd.read_csv(os.path.join(DATA_DIR, "mitre_atlas_raw.csv"))
unique_ids = sorted(df['mitre_id'].unique().tolist())
label_map = {mid: i for i, mid in enumerate(unique_ids)}

with open(os.path.join(CONFIG_DIR, "definitive_mitre_map.json"), "w") as f:
    json.dump(label_map, f)

print(f"[+] Definitive Map Created: {len(label_map)} Classes")