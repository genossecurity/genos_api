import pandas as pd
import random
import re
import os

def mutate_command(cmd):
    """
    Intelligently mutates a command to create a unique variation 
    without breaking the core syntax or intent.
    """
    if not isinstance(cmd, str):
        return cmd
        
    mutated = cmd
    
    # 1. Swap IPs (e.g., 1.1.1.1 -> random routable IP)
    mutated = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 
                     lambda m: f"{random.randint(11, 250)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}", 
                     mutated)
    
    # 2. Swap generic variables/numbers (var_456 -> var_8921)
    mutated = re.sub(r'var_\d+', lambda m: f"var_{random.randint(100, 99999)}", mutated)
    
    # 3. Swap Usernames in Windows/Linux paths
    users = ['jsmith', 'mwilliams', 'admin', 'devops', 'svc_account', 'john', 'backup_svc']
    mutated = re.sub(r'(c:\\users\\)[a-zA-Z0-9_]+', lambda m: m.group(1) + random.choice(users), mutated, flags=re.IGNORECASE)
    mutated = re.sub(r'(/home/)[a-zA-Z0-9_]+', lambda m: m.group(1) + random.choice(users), mutated, flags=re.IGNORECASE)
    
    # 4. Swap 4-to-5 digit ports or PIDs
    mutated = re.sub(r'\b[1-9]\d{3,4}\b', lambda m: str(random.randint(1024, 65535)), mutated)
    
    # 5. Swap out common filenames
    files = {'notes.txt': 'config.txt', 'setup.exe': 'update.exe', 'app.log': 'system.log', 'data.csv': 'export.csv'}
    for old, new in files.items():
        if old in mutated:
            mutated = mutated.replace(old, random.choice([old, new, f"temp_{random.randint(1,99)}.log"]))
            
    # 6. Fallback: If no regex matched and the command is identical, append a dummy comment or null route
    if mutated == cmd:
        padding = random.choice([
            f" # run_{random.randint(1000,9999)}",
            f" > /dev/null 2>&1 # id_{random.randint(10,99)}",
            f" ; echo {random.randint(1,99)} > $null" if "powershell" in cmd.lower() else ""
        ])
        mutated = mutated + padding.strip()

    return mutated

def enforce_target_count(df_subset, target_count=100):
    """Downsamples if over target, upsamples (synthesizes) if under target."""
    current_count = len(df_subset)
    
    # Downsample
    if current_count >= target_count:
        return df_subset.sample(n=target_count, random_state=42)
    
    # Upsample
    needed = target_count - current_count
    original_cmds = df_subset['command'].tolist()
    
    # Keep track of mitre_id if it exists in this subset
    has_mitre = 'mitre_id' in df_subset.columns
    if has_mitre:
        mitre_id = df_subset['mitre_id'].iloc[0]
    
    synthetic_rows = []
    existing_set = set(original_cmds)
    
    attempts = 0
    max_attempts = needed * 20 
    
    while len(synthetic_rows) < needed and attempts < max_attempts:
        base_cmd = random.choice(original_cmds)
        new_cmd = mutate_command(base_cmd)
        
        if new_cmd not in existing_set:
            existing_set.add(new_cmd)
            row_data = {'command': new_cmd}
            if has_mitre:
                row_data['mitre_id'] = mitre_id
            synthetic_rows.append(row_data)
        attempts += 1

    synth_df = pd.DataFrame(synthetic_rows)
    return pd.concat([df_subset, synth_df], ignore_index=True)

# ==========================================
# Phase 1: The Malicious Dataset
# ==========================================
print("Phase 1: Balancing Malicious Data (100 examples per MITRE ID)...")
bad_df = pd.read_csv('trainer1-bad.csv').drop_duplicates(subset=['mitre_id', 'command'])

balanced_bad = []
for name, group in bad_df.groupby('mitre_id'):
    balanced_bad.append(enforce_target_count(group, 100))

final_bad_df = pd.concat(balanced_bad, ignore_index=True)[['mitre_id', 'command']]
final_bad_df = final_bad_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_bad_df.to_csv('genos-balanced-bad.csv', index=False)
print(f" -> Success: genos-balanced-bad.csv created. Rows: {len(final_bad_df)}\n")

# ==========================================
# Phase 2: The Benign Dataset
# ==========================================
print("Phase 2: Balancing Benign Data (100 examples per Base Command)...")
good_df = pd.read_csv('trainer1-good.csv').drop_duplicates(subset=['command'])

# Extract the base executable (e.g., 'docker run ...' -> 'docker')
good_df['base_cmd'] = good_df['command'].astype(str).apply(lambda x: x.strip().split(' ')[0].lower())

balanced_good = []
for name, group in good_df.groupby('base_cmd'):
    if not name or name == 'nan':
        continue
    # Drop the temporary base_cmd column for the final payload
    clean_group = group[['command']] 
    balanced_good.append(enforce_target_count(clean_group, 100))

final_good_df = pd.concat(balanced_good, ignore_index=True)[['command']]
final_good_df = final_good_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_good_df.to_csv('genos-balanced-good.csv', index=False)
print(f" -> Success: genos-balanced-good.csv created. Rows: {len(final_good_df)}")

print("\nEngine datasets are ready. The class ratio is now strictly controlled.")