import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset_loader import ASLDataset, get_dataloader
import argparse

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ASLSignTransformer(nn.Module):
    def __init__(self, input_dim=126, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, num_classes=50, dropout=0.1):
        super(ASLSignTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, 126)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        # Global mean pooling
        h_pooled = h.mean(dim=1)
        logits = self.classifier(h_pooled)
        return logits

def train_model(json_path='static/json/reference.json', epochs=20, batch_size=16, lr=1e-3, save_path='asl_transformer_model.pt', holistic=False):
    print(f"Loading ASL dataset from {json_path}...")
    dataset = ASLDataset(json_path=json_path, max_seq_len=60, holistic=holistic)
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    print(f"Loaded {num_samples} samples across {num_classes} classes.")

    if num_samples < 2:
        print("Not enough dataset samples to train. Please run ASLCoordinateDictionary.py first.")
        return

    val_size = max(1, int(num_samples * 0.15))
    train_size = num_samples - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    input_dim = 144 if holistic else 126
    model = ASLSignTransformer(input_dim=input_dim, d_model=128, nhead=4, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += x_batch.size(0)

        train_acc = correct / total if total > 0 else 0.0
        train_loss = train_loss / total if total > 0 else 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += x_batch.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss / val_total if val_total > 0 else 0.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'word_to_idx': dataset.word_to_idx,
                'idx_to_word': dataset.idx_to_word,
                'num_classes': num_classes
            }, save_path)

    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}. Saved checkpoint to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ASL Sign Transformer")
    parser.add_argument("--holistic", action="store_true", help="Train on holistic dataset (144 dims)")
    args = parser.parse_args()
    
    if args.holistic:
        train_model(json_path='static/json/reference_holistic.json', save_path='asl_transformer_model_holistic.pt', holistic=True)
    else:
        train_model()
