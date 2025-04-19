import torch
import sys
from preprocess import preprocess, encode, pad_sequence, load_vocab
from train import ReviewRegressor

vocab = load_vocab("models/vocab.json")
vocab_size = len(vocab)
model_params = {"embed_dim": 128, "hidden_dim": 128, "num_layers": 2}
model = ReviewRegressor(vocab_size=vocab_size, **model_params)

checkpoint = torch.load("models/lstm_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

if len(sys.argv) < 2:
    print("Usage: python predict.py 'Your review here'")
    exit(1)

review = sys.argv[1]
tokens = preprocess(review)
x = torch.tensor([pad_sequence(encode(tokens, vocab))])

with torch.no_grad():
    pred = model(x).item()

def convert_to_stars(pred_10scale):
    pred_10scale = round(min(max(pred_10scale, 1), 10))
    return pred_10scale / 2

stars = convert_to_stars(pred)
print(f"Predicted rating: {stars:.1f} / 5 stars")