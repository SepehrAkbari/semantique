import torch
import json

# Load model checkpoint
checkpoint = torch.load("../saved_models/best_model.pt", map_location="cpu")
vocab = checkpoint["vocab"]

# Save vocab to JSON
with open("../saved_models/vocab.json", "w") as f:
    json.dump(vocab, f)

print("Vocab successfully saved to saved_models/vocab.json")