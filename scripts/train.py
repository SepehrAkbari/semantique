import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import prepare_data
from torch.utils.data import random_split
import os

# Define the model
class ReviewRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=0.3, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)              # (B, T) -> (B, T, E)
        lstm_out, _ = self.lstm(embedded)         # (B, T, H)
        last_hidden = lstm_out[:, -1, :]          # (B, H)
        out = self.dropout(last_hidden)
        out = self.fc(out).squeeze(1)             # (B,)
        return out

if __name__ == "__main__":
    # Load data and vocab
    train_loader, test_loader, vocab = prepare_data()

    # Split off a validation set from training data
    full_train_dataset = train_loader.dataset
    val_size = int(0.1 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, optimizer
    model = ReviewRegressor(vocab_size=len(vocab)).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop with validation monitoring and early stopping
    epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    save_dir = "../models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'params': {
                    'embed_dim': 128,
                    'hidden_dim': 128,
                    'num_layers': 2
                }
            }, os.path.join(save_dir, "lstm_model.pt"))
            print("Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete. Best model saved to models/lstm_model.pt")