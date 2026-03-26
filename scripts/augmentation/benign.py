import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

def normalize(cmd):
    return str(cmd).lower().strip()

def run_benign_prep():
    '''
    This script takes the synthetic benign baseline and augments it with 1,000 samples of "Admin Noise" commands.
    The final output is a cleaned and shuffled dataset of 10,000 benign commands saved as "benign_final.csv".
    '''
    # 1. Load your original benign data
    df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_benign_baseline.csv"))
    df['command'] = df['command'].apply(normalize)
    df = df.drop_duplicates()
    
    # 2. Inject "Admin Noise" (Common clashing commands)
    # We add 1000 samples of "Normal Admin Activity" that looks like Recon
    admin_noise = [
        "whoami", "hostname", "ipconfig /all", "net user", "net group \"domain admins\" /domain",
        "systeminfo", "tasklist", "netstat -ano", "qwinsta", "gpresult /r", "dir c:\\ /s /b"
    ]
    noise_rows = []
    for _ in range(100): # Create 100 variants of each
        for cmd in admin_noise:
            noise_rows.append(normalize(cmd))
            
    # Combine and cap at 10,000
    all_cmds = df['command'].tolist() + noise_rows
    final_df = pd.DataFrame({'command': all_cmds}).sample(frac=1).reset_index(drop=True).head(10000)
    
    final_df.to_csv(os.path.join(DATA_DIR, "benign_final.csv"), index=False)
    print(f"[+] Created benign_final.csv with {len(final_df)} samples (including Admin Noise).")

if __name__ == "__main__":
    run_benign_prep()