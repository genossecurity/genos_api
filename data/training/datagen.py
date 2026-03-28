import os
import random
import re
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "training")

# Ensure output directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration
TARGET_SAMPLES_PER_CLASS = 100
BENIGN_TOTAL_SAMPLES = 15000  # Scales to match the roughly ~14k malicious samples

# ==========================================
# 1. BENIGN DATA GENERATOR (Gatekeeper)
# Target: Fix False Positives (DevOps, Pipes)
# ==========================================
def generate_benign_samples(num_samples=BENIGN_TOTAL_SAMPLES):
    print(f"[*] Generating {num_samples} Benign (Gatekeeper) samples...")
    commands = []
    
    # Template Pools for "Hard Benign" Noise
    k8s_namespaces = ['production', 'staging', 'dev', 'kube-system', 'monitoring']
    k8s_resources = ['pods', 'deployments', 'services', 'nodes', 'configmaps']
    tf_actions = ['apply -auto-approve', 'plan -out=tfplan', 'init -upgrade', 'destroy -auto-approve']
    ps_cmdlets = ['Get-Service', 'Get-Process', 'Get-EventLog -LogName System', 'Get-Content', 'Get-ADUser']
    ps_pipes = ['| Where-Object {$_.Status -eq \'Running\'}', '| Select-Object Name, Id', '| Sort-Object -Descending', '| % { $_.Name }']
    restricted_flags = ['-ExecutionPolicy Restricted', '-ep Restricted', '-ExecutionPolicy AllSigned']
    linux_cmds = ['tar -czf /tmp/backup.tar.gz /var/log', 'systemctl status nginx', 'grep -r "ERROR" /var/log/']
    
    for _ in range(num_samples):
        category = random.choice(['k8s', 'terraform', 'git', 'piped_ps', 'restricted_ps', 'verbose_log', 'linux_admin'])
        
        if category == 'k8s':
            cmd = f"kubectl get {random.choice(k8s_resources)} -n {random.choice(k8s_namespaces)} {'--watch' if random.random() > 0.5 else '-o wide'}"
        elif category == 'terraform':
            cmd = f"terraform {random.choice(tf_actions)}"
        elif category == 'git':
            hashes = f"{random.randint(1000,9999)}"
            cmd = f"git commit -m 'Fixed weight loading issue in engine.py - issue #{hashes}'"
        elif category == 'piped_ps':
            cmd = f"{random.choice(ps_cmdlets)} {random.choice(ps_pipes)}"
        elif category == 'restricted_ps':
            # Contrast training for the bypass FN
            cmd = f"powershell {random.choice(restricted_flags)} -File C:\\Scripts\\maintenance_{random.randint(1,100)}.ps1"
        elif category == 'linux_admin':
            cmd = f"{random.choice(linux_cmds)} > /dev/null 2>&1 # task_{random.randint(100,999)}"
        else:
            cmd = f"cat /var/log/syslog | grep 'systemd' | tail -n {random.randint(10, 100)}"
            
        commands.append({"command": cmd, "mitre_id": "Benign"})
        
    return pd.DataFrame(commands)

# ==========================================
# 2. MALICIOUS DATA BALANCER & GENERATOR
# Target: 100 per class + T1003/T1059 Fixes
# ==========================================
def mutate_command(cmd):
    """Slightly mutates duplicated commands to prevent perfect overfitting."""
    cmd = str(cmd)
    mutation_type = random.choice(['trailing_space', 'comment', 'casing', 'none'])
    
    if mutation_type == 'trailing_space':
        return cmd + (" " * random.randint(1, 3))
    elif mutation_type == 'comment':
        # Append a harmless bash/powershell style comment ID
        return cmd + f" # id_{random.randint(1000, 9999)}"
    elif mutation_type == 'casing':
        # Randomly lower or upper case the entire string 10% of the time
        return cmd.lower() if random.random() > 0.5 else cmd.upper()
    return cmd

def generate_custom_malicious_samples(mitre_id, count):
    """Generates our custom synthesized False Negative fixes on demand."""
    custom_cmds = []
    
    paths = ['C:\\windows\\temp\\', 'C:\\Users\\Public\\', 'C:\\PerfLogs\\', 'C:\\ProgramData\\', '']
    extensions = ['.save', '.bak', '.tmp', '.txt', '.dat', '.hiv', '']
    names = ['sam', 'registry', 'backup', 'dump', 'cache']
    ps_exec_flags = ['-ExecutionPolicy Bypass', '-ep bypass', '-exec bypass', '-ExecutionPolicy Unrestricted']
    unc_hosts = ['192.168.1.15', '10.0.0.5', 'DC-01', 'FILESHARE', '172.16.0.2']
    unc_shares = ['SYSVOL', 'C$', 'IPC$', 'public', 'payloads']
    
    for _ in range(count):
        if mitre_id == 'T1003':  # Registry Dump FN fix
            target_path = f"{random.choice(paths)}{random.choice(names)}{random.choice(extensions)}"
            if random.random() > 0.5: target_path = target_path.lower()
            if random.random() > 0.8: target_path = target_path.upper()
            custom_cmds.append(f"reg save HKLM\\SAM {target_path}")
            
        elif mitre_id == 'T1059.001':  # UNC PowerShell Bypass FN fix
            unc_path = f"\\\\{random.choice(unc_hosts)}\\{random.choice(unc_shares)}\\script_{random.randint(1,99)}.ps1"
            custom_cmds.append(f"powershell {random.choice(ps_exec_flags)} -WindowStyle Hidden -File {unc_path}")
    
    return custom_cmds

def process_malicious_data(raw_csv_path):
    print(f"[*] Processing and Balancing Malicious Data from {raw_csv_path}...")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"[!] Could not find raw data at {raw_csv_path}")
        
    df_raw = pd.read_csv(raw_csv_path)
    balanced_records = []
    
    # Group by MITRE ID
    grouped = df_raw.groupby('mitre_id')
    
    for mitre_id, group in grouped:
        cmds = group['command'].tolist()
        
        # Inject our custom fixes if applicable
        if mitre_id in ['T1003', 'T1059.001']:
            cmds.extend(generate_custom_malicious_samples(mitre_id, 50))
            
        current_count = len(cmds)
        
        if current_count >= TARGET_SAMPLES_PER_CLASS:
            # Downsample without replacement
            sampled_cmds = random.sample(cmds, TARGET_SAMPLES_PER_CLASS)
        else:
            # Upsample with mutation to prevent perfect duplicates
            sampled_cmds = list(cmds)
            deficit = TARGET_SAMPLES_PER_CLASS - current_count
            for _ in range(deficit):
                base_cmd = random.choice(cmds)
                sampled_cmds.append(mutate_command(base_cmd))
                
        for cmd in sampled_cmds:
            balanced_records.append({"mitre_id": mitre_id, "command": cmd})
            
    df_balanced = pd.DataFrame(balanced_records)
    print(f"[+] Balanced Malicious Data: {len(df_balanced)} rows across {len(grouped)} classes.")
    return df_balanced

# ==========================================
# 3. SPLIT & EXPORT ENGINE
# ==========================================
def export_splits(df, base_filename):
    print(f"[*] Exporting splits for: {base_filename}")
    
    # Shuffle the dataframe thoroughly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 80% Train, 20% Temp (Val + Test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Split the 20% Temp into 10% Val and 10% Test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df.to_csv(os.path.join(DATA_DIR, f"{base_filename}_train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, f"{base_filename}_val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, f"{base_filename}_test.csv"), index=False)
    
    print(f"    -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    print("=====================================================")
    print("GENOS DATAGEN: NORMALIZATION & FIX SYNTHESIS")
    print("=====================================================")
    
    # 1. Generate Benign Data
    df_benign = generate_benign_samples(BENIGN_TOTAL_SAMPLES)
    
    # 2. Process, Balance, and Mutate Malicious Data
    raw_mitre_path = "data/art/mitre_atlas_raw.csv" # Ensure this is in the same directory or update path
    df_malicious = process_malicious_data(raw_mitre_path)
    
    # 3. Export splits
    export_splits(df_benign, "gatekeeper")
    export_splits(df_malicious, "specialist")
    
    print("=====================================================")
    print("[SUCCESS] All files generated successfully in data/training/")