import torch
import json

checkpoint = torch.load("../saved_models/best_model.pt", map_location="cpu")
vocab = checkpoint["vocab"]

with open("../saved_models/vocab.json", "w") as f:
    json.dump(vocab, f)