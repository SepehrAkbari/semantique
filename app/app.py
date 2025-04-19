import sys
import os
from flask import Flask, render_template, request
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from preprocess import preprocess, encode, pad_sequence, load_vocab
from train import ReviewRegressor

app = Flask(__name__)

vocab = load_vocab("../models/vocab.json")
model_params = {"embed_dim": 128, "hidden_dim": 128, "num_layers": 2}
model = ReviewRegressor(vocab_size=len(vocab), **model_params)
checkpoint = torch.load("../models/lstm_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def convert_to_stars(pred_10scale):
    pred_10scale = round(min(max(pred_10scale, 1), 10))
    return pred_10scale / 2

@app.route("/", methods=["GET", "POST"])
def index():
    stars = None
    review_text = ""
    if request.method == "POST":
        review_text = request.form["review"]
        tokens = preprocess(review_text)
        x = torch.tensor([pad_sequence(encode(tokens, vocab))])
        with torch.no_grad():
            pred = model(x).item()
        stars = convert_to_stars(pred)
    return render_template("index.html", stars=stars, review_text=review_text)
