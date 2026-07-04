import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "sonuc_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# 1. VERİ SETİ (DATASET)
# =====================
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64):
        df = pd.read_csv(file_path)
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["label"] = torch.tensor(self.labels[idx])
        return item

# =====================
# 2. MODELLER
# =====================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=256, hidden_dim=128, dropout_rate=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.emb_drop(x)
        lstm_out, _ = self.lstm(x)
        x, _ = torch.max(lstm_out, dim=1) 
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.fc(cls)

# =====================
# 3. EĞİTİM DÖNGÜSÜ
# =====================
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, model_type, scheduler=None):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN - {model_type.upper()}]")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device).view(-1)

            if model_type == "bert":
                outputs = model(input_ids, batch["attention_mask"].to(device)).view(-1)
            else:
                outputs = model(input_ids).view(-1)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device).view(-1)
                outputs = model(input_ids, batch["attention_mask"].to(device)).view(-1) if model_type == "bert" else model(input_ids).view(-1)
                total_val_loss += criterion(outputs, labels).item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"-> Epoch {epoch+1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
        if scheduler: scheduler.step(avg_val_loss)

    return train_losses, val_losses

# =====================
# 4. GÖRSELLEŞTİRME VE RAPOR
# =====================
def plot_learning_curve(train_loss, val_loss, filename, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label="Train Loss", color='blue')
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker='x', label="Validation Loss", color='red')
    plt.title(f"{model_name} - Öğrenme ve Ezberleme Analizi")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Kayıp)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def write_train_report(results, hp_config):
    with open(os.path.join(OUTPUT_DIR, "train_raporu.txt"), "w", encoding="utf-8") as f:
        f.write("DERİN ÖĞRENME PROJESİ - EĞİTİM AŞAMASI RAPORU\n" + "="*60 + "\n\n")
        for key, val in hp_config.items(): f.write(f"{key}: {val}\n")
        f.write("\nOVERFITTING ANALİZİ\n" + "-" * 40 + "\n")
        for model_name, data in results.items():
            f.write(f"=== {model_name} MODELİ ===\n")
            f.write(f"Bitiş Train / Val Loss: {data['train'][-1]:.4f} / {data['val'][-1]:.4f}\n")
            f.write("YORUM: " + ("Ezberleme (Overfitting) riski." if data['val'][-1] > data['train'][-1] * 1.5 else "Sağlıklı öğrenme.\n\n"))

if __name__ == "__main__":
    HP_CONFIG = {"batch_size": 32, "lstm_lr": 1e-3, "bert_lr": 2e-5, "lstm_epochs": 10, "bert_epochs": 3, "weight_decay": 1e-4}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_loader = DataLoader(TextDataset("dataset/train.csv", tokenizer), batch_size=HP_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(TextDataset("dataset/val.csv", tokenizer), batch_size=HP_CONFIG["batch_size"], shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    train_results = {}

    # LSTM EĞİTİMİ
    print("--- LSTM MODELİ EĞİTİMİ BAŞLIYOR ---")
    lstm = LSTMClassifier().to(device)
    lstm_opt = torch.optim.AdamW(lstm.parameters(), lr=HP_CONFIG["lstm_lr"], weight_decay=HP_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(lstm_opt, mode='min', patience=1, factor=0.5)
    t_loss, v_loss = train_model(lstm, train_loader, val_loader, lstm_opt, criterion, HP_CONFIG["lstm_epochs"], "lstm", scheduler)
    train_results["LSTM"] = {"train": t_loss, "val": v_loss}
    plot_learning_curve(t_loss, v_loss, "lstm_overfitting_curve.png", "LSTM")
    torch.save(lstm.state_dict(), os.path.join(OUTPUT_DIR, "lstm_model.pth"))

    # BERT EĞİTİMİ
    print("--- BERT MODELİ EĞİTİMİ BAŞLIYOR ---")
    bert = BERTClassifier().to(device)
    bert_opt = torch.optim.Adam(bert.parameters(), lr=HP_CONFIG["bert_lr"])
    bt_loss, bv_loss = train_model(bert, train_loader, val_loader, bert_opt, criterion, HP_CONFIG["bert_epochs"], "bert")
    train_results["BERT"] = {"train": bt_loss, "val": bv_loss}
    plot_learning_curve(bt_loss, bv_loss, "bert_overfitting_curve.png", "BERT")
    torch.save(bert.state_dict(), os.path.join(OUTPUT_DIR, "bert_model.pth"))

    write_train_report(train_results, HP_CONFIG)
    print(f"İşlemler tamamlandı! Ağırlıklar '{OUTPUT_DIR}/' klasörüne kaydedildi.")