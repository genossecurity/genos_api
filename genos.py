import torch
import json
from transformers import RobertaTokenizer

class GenosPipeline:
    def __init__(self):
        self.device = torch.device("cuda")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Load Tier 1: Gatekeeper (Your 0% FP model)
        # We assume Class 0 is 'BENIGN' in the original map
        self.gatekeeper = load_atlas_model("atlas_engine.pt", 142) 
        
        # Load Tier 2: Specialist (The new model)
        self.specialist = load_specialist_model("specialist_ttp_engine.pt", 141)
        
        with open("specialist_label_map.json", "r") as f:
            self.ttp_map = {v: k for k, v in json.load(f).items()}

    def scan(self, cmd):
        inputs = self.tokenizer(cmd.lower(), return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(self.device)
        
        with torch.no_grad():
            # STAGE 1: Is it Benign?
            gate_logits = self.gatekeeper(inputs['input_ids'], inputs['attention_mask'])
            gate_probs = torch.softmax(gate_logits, dim=1)
            gate_conf, gate_idx = torch.max(gate_probs, dim=1)
            
            if gate_idx.item() == 0: # 0 is Benign in atlas_engine
                return {"verdict": "BENIGN", "confidence": f"{gate_conf.item():.2%}", "mitre": "N/A"}
            
            # STAGE 2: If Malicious, let the specialist identify the TTP
            ttp_logits = self.specialist(inputs['input_ids'], inputs['attention_mask'])
            ttp_probs = torch.softmax(ttp_logits, dim=1)
            ttp_conf, ttp_idx = torch.max(ttp_probs, dim=1)
            
            return {
                "verdict": "MALICIOUS",
                "confidence": f"{ttp_conf.item():.2%}",
                "mitre": self.ttp_map[ttp_idx.item()]
            }

# (Helper load functions omitted for brevity)