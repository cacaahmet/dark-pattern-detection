import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR, WEIGHTS_DIR = "sonuc_test", "sonuc_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset ve Model Sınıfları
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64):
        df = pd.read_csv(file_path)
        self.texts, self.labels = df["text"].astype(str).tolist(), df["label"].astype(float).tolist()
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(self.labels[idx])
        return item

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
        x = self.emb_drop(self.embedding(input_ids))
        lstm_out, _ = self.lstm(x)
        x, _ = torch.max(lstm_out, dim=1) 
        return self.fc2(self.dropout(self.relu(self.fc1(self.bn1(x)))))

class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 1)
    def forward(self, input_ids, attention_mask):
        return self.fc(self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :])

def evaluate_test(model, dataloader, model_type, desc):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids, labels = batch["input_ids"].to(device), batch["label"].to(device).view(-1)
            outputs = model(input_ids, batch["attention_mask"].to(device)).view(-1) if model_type == "bert" else model(input_ids).view(-1)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).long().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    return acc, prec, rec, f1, confusion_matrix(all_labels, all_preds), fpr, tpr, auc(fpr, tpr), classification_report(all_labels, all_preds, target_names=["Normal (0)", "Dark Pattern (1)"], digits=4)

def plot_cm(cm, filename, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=True, xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title(f"{model_name} Nihai Test Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_roc(fpr, tpr, roc_auc, filename, model_name):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='tab:blue', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='tab:orange', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.05])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Nihai Test (Gerçek Dünya Simülasyonu) başlatılıyor...\n")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_loader = DataLoader(TextDataset("dataset/test.csv", tokenizer), batch_size=32, shuffle=False)
    results = {}

    for name, ModelClass, file, m_type in [("LSTM", LSTMClassifier, "lstm_model.pth", "lstm"), ("BERT", BERTClassifier, "bert_model.pth", "bert")]:
        print(f"[{name}] Modeli Test Ediliyor...")
        model = ModelClass().to(device)
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, file), weights_only=True, map_location=device))
        
        acc, prec, rec, f1, cm, fpr, tpr, roc_auc, cls_rep = evaluate_test(model, test_loader, m_type, f"{name} Test")
        plot_cm(cm, f"{name.lower()}_test_cm.png", name)
        plot_roc(fpr, tpr, roc_auc, f"{name.lower()}_roc_curve.png", name)
        results[name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": roc_auc, "report": cls_rep}

    with open(os.path.join(OUTPUT_DIR, "nihai_test_raporu.txt"), "w", encoding="utf-8") as f:
        f.write("NİHAİ TEST (TESTING) KAPANIŞ RAPORU\n" + "="*75 + "\n")
        f.write(f"{'Metrik':<15} | {'LSTM Modeli':<15} | {'BERT Modeli':<15}\n" + "-" * 50 + "\n")
        for metric, key in zip(["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], ["acc", "prec", "rec", "f1", "auc"]):
            f.write(f"{metric:<15} | {results['LSTM'][key]:.4f} {'':<8} | {results['BERT'][key]:.4f}\n")
            
        f.write("\nDETAYLI SINIFLANDIRMA RAPORLARI\n" + "-" * 50 + "\n")
        f.write("=== LSTM ===\n" + results['LSTM']['report'] + "\n=== BERT ===\n" + results['BERT']['report'] + "\n")
        
    print(f"\n✅ Tüm test işlemleri başarıyla tamamlandı! Çıktılar '{OUTPUT_DIR}/' klasöründedir.")