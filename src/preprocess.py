import os
import tarfile
import urllib.request
import glob
import re
from collections import Counter
from random import shuffle
import torch
from torch.utils.data import DataLoader, TensorDataset
import json

def regex_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def download_imdb():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = "aclImdb_v1.tar.gz"
    foldername = "../data/aclImdb"

    if not os.path.exists(foldername):
        print("Downloading IMDb dataset...")
        urllib.request.urlretrieve(url, filename)
        print("Extracting...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        print("Done!")
    else:
        print("IMDb dataset already exists.")

def load_reviews(path):
    reviews = []
    for filepath in glob.glob(path + "/*.txt"):
        with open(filepath, encoding='utf8') as f:
            text = f.read()
        rating = int(filepath.split("_")[-1].split(".")[0])
        reviews.append((text, rating))
    return reviews

def preprocess(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return regex_tokenize(text)

def build_vocab(tokenized_data, vocab_size=20000):
    counter = Counter()
    for tokens, _ in tokenized_data:
        counter.update(tokens)
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(vocab_size))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def encode(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_sequence(seq, max_len=300):
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    else:
        return seq[:max_len]

def get_dataloaders(train_data, test_data, vocab, max_len=300, batch_size=64):
    X_train = [pad_sequence(encode(tokens, vocab), max_len) for tokens, _ in train_data]
    y_train = [rating for _, rating in train_data]

    X_test = [pad_sequence(encode(tokens, vocab), max_len) for tokens, _ in test_data]
    y_test = [rating for _, rating in test_data]

    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def prepare_data():
    download_imdb()
    train_pos = load_reviews("aclImdb/train/pos")
    train_neg = load_reviews("aclImdb/train/neg")
    test_pos = load_reviews("aclImdb/test/pos")
    test_neg = load_reviews("aclImdb/test/neg")

    train_data = [(preprocess(text), rating) for text, rating in train_pos + train_neg]
    test_data = [(preprocess(text), rating) for text, rating in test_pos + test_neg]
    shuffle(train_data)
    shuffle(test_data)

    vocab = build_vocab(train_data)
    train_loader, test_loader = get_dataloaders(train_data, test_data, vocab)
    return train_loader, test_loader, vocab

def load_vocab(path="../models/vocab.json"):
    with open(path, "r") as f:
        vocab = json.load(f)
    return {k: int(v) for k, v in vocab.items()}
