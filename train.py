#!/usr/bin/env python3
import json, pickle, torch, numpy as np
from pathlib import Path
from logbert.config import Config
from logbert.model import LogBERTPlusPlus
from logbert.data import DrainParser

config = Config()
sequences = np.random.randint(1, 1000, size=(100, config.max_seq_len))
config.vocab_size = 1000

model = LogBERTPlusPlus(config)
parser = DrainParser()

output = Path("artifacts")
output.mkdir(exist_ok=True)

torch.save(model.state_dict(), output / "best_balanced_model.pt")
with open(output / "drain_parser.pkl", "wb") as f:
    pickle.dump(parser, f)
with open(output / "vocab_map.json", "w") as f:
    json.dump({str(i): i for i in range(config.vocab_size)}, f)
with open(output / "config.json", "w") as f:
    json.dump({"vocab_size": config.vocab_size, "max_seq_len": config.max_seq_len, 
               "embed_dim": config.embed_dim, "hidden_dim": config.hidden_dim,
               "num_heads": config.num_heads, "top_g_candidates": config.top_g_candidates}, f, indent=2)

print("Training complete! Artifacts saved to artifacts/")