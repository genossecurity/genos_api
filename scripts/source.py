import pandas as pd
import json

# Force a strictly sorted, unique list of MITRE IDs
df = pd.read_csv("mitre_atlas_raw.csv")
unique_ids = sorted(df['mitre_id'].unique().tolist())
label_map = {mid: i for i, mid in enumerate(unique_ids)}

with open("definitive_mitre_map.json", "w") as f:
    json.dump(label_map, f)

print(f"[+] Definitive Map Created: {len(label_map)} Classes")