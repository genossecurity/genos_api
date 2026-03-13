import torch
import torch.nn as nn
import pandas as pd
import json
import random
import re
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

# --- 1. THE FUZZING ENGINE ---
class CommandFuzzer:
    @staticmethod
    def random_case(cmd):
        return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in cmd)

    @staticmethod
    def backtick_injection(cmd):
        # Specifically for PowerShell evasion
        words = cmd.split()
        new_words = []
        for word in words:
            if len(word) > 3 and word.isalpha() and random.random() > 0.7:
                # Insert a backtick e.g. "powershell" -> "p`owersh`ell"
                idx = random.randint(1, len(word)-1)
                word = word[:idx] + "`" + word[idx:]
            new_words.append(word)
        return " ".join(new_words)

    @staticmethod
    def path_flip(cmd):
        return cmd.replace("/", "\\") if "/" in cmd else cmd.replace("\\", "/")

    @staticmethod
    def whitespace_chaos(cmd):
        return cmd.replace(" ", "  " if random.random() > 0.5 else "   ")

    @staticmethod
    def env_swap(cmd):
        subs = {
            "c:\\windows\\system32": "%windir%\\system32",
            "c:\\users": "%homedrive%\\users",
            "powershell.exe": "pwsh.exe"
        }
        for k, v in subs.items():
            if k in cmd.lower():
                # Using a lambda here prevents re.sub from interpreting backslashes in 'v'
                cmd = re.sub(re.escape(k), lambda _: v, cmd, flags=re.IGNORECASE)
        return cmd

    def apply_cocktail(self, cmd):
        # Apply 2-3 random fuzzing techniques
        funcs = [self.random_case, self.backtick_injection, self.path_flip, self.whitespace_chaos, self.env_swap]
        for f in random.sample(funcs, random.randint(1, 3)):
            cmd = f(cmd)
        return cmd

# --- 2. ARCHITECTURES ---
class GatekeeperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, 2))
    def forward(self, ids, mask):
        return self.classifier(self.encoder(ids, mask).last_hidden_state[:, 0, :])

class SpecialistModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, num_classes)
        )
    def forward(self, ids, mask):
        return self.classifier(self.encoder(ids, mask).last_hidden_state[:, 0, :])

# --- 3. THE STRESS TEST ---
def run_stress_test():
    device = torch.device("cuda")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    fuzzer = CommandFuzzer()

    # Load Models & Map
    with open("specialist_map.json", "r") as f: s_map = {int(v): k for k, v in json.load(f).items()}
    t1 = GatekeeperModel().to(device)
    t1.load_state_dict(torch.load("gatekeeper.pt", map_location=device))
    t2 = SpecialistModel(num_classes=141).to(device)
    t2.load_state_dict(torch.load("specialist.pt", map_location=device))
    t1.eval(); t2.eval()

    m_df = pd.read_csv("mitre_atlas_raw.csv")
    hits, exact = 0, 0
    
    print("\n" + "!"*50 + "\n🔥 INITIATING ADVERSARIAL STRESS TEST\n" + "!"*50)

    for _, row in tqdm(m_df.iterrows(), total=len(m_df), desc="Attacking the Engine"):
        # Fuzz the original malicious command
        fuzzed_cmd = fuzzer.apply_cocktail(row['command'])
        actual_mitre = row['mitre_id']

        inputs = tokenizer(fuzzed_cmd.lower().strip(), return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(device)
        
        with torch.no_grad():
            # Tier 1 Gatekeeper
            if torch.argmax(t1(inputs['input_ids'], inputs['attention_mask']), dim=1).item() == 1:
                hits += 1
                # Tier 2 Specialist
                idx = torch.argmax(t2(inputs['input_ids'], inputs['attention_mask']), dim=1).item()
                if s_map[idx] == actual_mitre:
                    exact += 1

    print(f"\n" + "="*50)
    print(f"📊 STRESS TEST RESULTS (Fuzzed Data)")
    print(f"="*50)
    print(f"Overall Detection: {(hits/len(m_df))*100:.2f}%")
    print(f"MITRE Match Accuracy: {(exact/hits)*100:.2f}%")
    print(f"Resilience Factor: {((exact/hits) / 0.9859)*100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_stress_test()