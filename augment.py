import pandas as pd
import random

def run_benign_prep():
    df = pd.read_csv("synthetic_benign_baseline.csv")
    df['command'] = df['command'].str.lower().str.strip()
    df = df.drop_duplicates()
    
    # Inject 1,000 samples of "Admin Noise"
    noise = ["whoami", "hostname", "ipconfig /all", "net user", "systeminfo", "tasklist", "netstat -ano"]
    noise_rows = [random.choice(noise).lower() for _ in range(1000)]
    
    all_cmds = df['command'].tolist() + noise_rows
    final_df = pd.DataFrame({'command': all_cmds}).sample(frac=1).reset_index(drop=True).head(10000)
    final_df.to_csv("benign_final.csv", index=False)
    print(f"[+] Created benign_final.csv (10,000 samples)")

if __name__ == "__main__":
    run_benign_prep()