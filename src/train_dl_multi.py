# -*- coding: utf-8 -*-
"""
CICIDS2017 多分类深度学习模型(8 类)
==================================
类别: BENIGN / DDoS / PortScan / Botnet / BruteForce / WebAttack / DoS / Infiltration
模型: CNN-LSTM / Transformer(输出维度 = 8,逆频类别权重)
预处理与二分类一致: 分层抽样 30 万 -> StandardScaler -> 裁剪 ±10 -> 补零成序列 [N,128,1]
训练策略: 训练集内部再划分验证集,最多 50 epoch,验证损失早停并回滚最佳权重
保存为完整 checkpoint,便于推理:
  models/ids_cnn_lstm.pt
  models/ids_transformer.pt
  (含 state_dict / class_names / feature_cols / scaler / clip / seq_len)
输出:
  reports/dl_multi_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from label_utils import CLASS_NAMES, group_labels, label_to_idx

PROCESSED = ROOT / "processed" / "cicids2017_clean.parquet"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPORT = ROOT / "reports" / "dl_multi_results.json"

RANDOM_STATE = 42
SEQUENCE_LEN = 128
MAX_EPOCHS = 50
PATIENCE = 8
MIN_DELTA = 1e-4
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_VALUE = 10.0
TRAIN_SAMPLE = 300_000
VAL_SIZE = 0.1
N_CLASSES = len(CLASS_NAMES)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[设备] CPU")
    return torch.device("cpu")


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(PROCESSED)
    y_group = group_labels(df["label"])
    y = label_to_idx(y_group).values
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values
    feature_cols = list(df.drop(columns=["label"]).select_dtypes(include=[np.number]).columns)
    print(f"[加载] {len(X):,} 行 × {X.shape[1]} 特征")

    idx, _ = train_test_split(np.arange(len(y)), train_size=TRAIN_SAMPLE,
                              random_state=RANDOM_STATE, stratify=y)
    X, y = X[idx], y[idx]
    print(f"[抽样] {len(X):,} 条;类别分布: {np.bincount(y).tolist()}")
    return X, y, feature_cols


def split_standardize(X: np.ndarray, y: np.ndarray) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # 长尾离群值裁剪,抑制梯度爆炸
    X_train = np.clip(X_train, -CLIP_VALUE, CLIP_VALUE)
    X_test = np.clip(X_test, -CLIP_VALUE, CLIP_VALUE)
    return X_train, X_test, y_train, y_test, sc


def to_sequence(X: np.ndarray, seq_len: int) -> np.ndarray:
    """把 [N, F] 展平,不足 seq_len 补零/超出截断,统一返回 [N, seq_len, 1]。"""
    n, f = X.shape
    flat = X.reshape(n, -1)
    if f < seq_len:
        flat = np.pad(flat, ((0, 0), (0, seq_len - f)), mode="constant")
    else:
        flat = flat[:, :seq_len]
    return flat.reshape(n, seq_len, 1)


class CNNLSTM(nn.Module):
    def __init__(self, n_features, seq_len, num_classes, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):  # x: [B, seq_len, 1]
        x = x.permute(0, 2, 1)                 # [B, 1, seq_len]
        x = self.conv(x)                        # [B, 32, seq_len//2]
        x = x.permute(0, 2, 1)                 # [B, seq_len//2, 32]
        out, _ = self.lstm(x)                   # [B, seq_len//2, hidden*2]
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, n_features, seq_len, num_classes, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=256,
            dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, seq_len, 1]
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        return self.cls(x.mean(dim=1))


def evaluate_loss(model: nn.Module, loader, criterion, dev) -> float:
    model.eval()
    total_loss, n_batch = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            total_loss += criterion(model(xb), yb).item()
            n_batch += 1
    return total_loss / max(n_batch, 1)


def train_model(name: str, model: nn.Module, train_loader, val_loader, test_loader,
                dev, class_weight: torch.Tensor) -> dict:
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(dev))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    model.to(dev)
    print(f"\n========== 训练 {name}({N_CLASSES} 类) ==========")

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss, n_batch = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        train_loss = total_loss / max(n_batch, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, dev)
        train_losses.append(round(float(train_loss), 6))
        val_losses.append(round(float(val_loss), 6))
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{MAX_EPOCHS}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  best_epoch={best_epoch}")
        if wait >= PATIENCE:
            print(f"  [早停] 验证损失连续 {PATIENCE} 轮无显著改善,最佳 epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            preds.extend(model(xb).argmax(dim=1).cpu().numpy())
            labels.extend(yb.numpy())
    y_true, y_pred = np.array(labels), np.array(preds)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"  {name}: Accuracy={acc:.4f}  Macro-F1={macro:.4f}  Weighted-F1={weighted:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                zero_division=0, digits=4))
    return {
        "model": name,
        "task": "多分类(8类)",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro), 4),
        "weighted_f1": round(float(weighted), 4),
        "best_epoch": best_epoch,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "best_val_loss": round(float(best_val), 6),
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    set_seed()
    dev = device()
    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test, sc = split_standardize(X, y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=y_train)

    seq = to_sequence(X_train, SEQUENCE_LEN)
    seq_val = to_sequence(X_val, SEQUENCE_LEN)
    seq_test = to_sequence(X_test, SEQUENCE_LEN)
    print(f"[序列] 训练 {seq.shape}  验证 {seq_val.shape}  测试 {seq_test.shape}")

    train_ds = TensorDataset(torch.FloatTensor(seq), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(seq_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(seq_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 逆频类别权重:每类总权重相等,缓解严重失衡
    counts = np.bincount(y_train)
    total = counts.sum()
    cw = torch.tensor([total / (N_CLASSES * counts[c]) for c in range(N_CLASSES)],
                      dtype=torch.float32)
    print(f"[权重] 类别权重: {cw.tolist()}")

    results = []
    for name, build in [("cnn_lstm", CNNLSTM), ("transformer", TransformerModel)]:
        model = build(seq.shape[2], SEQUENCE_LEN, N_CLASSES)
        res = train_model(name, model, train_loader, val_loader, test_loader, dev, cw)
        results.append(res)
        torch.save({
            "state_dict": model.state_dict(),
            "arch": name,
            "class_names": CLASS_NAMES,
            "feature_cols": feature_cols,
            "scaler": sc,
            "clip": CLIP_VALUE,
            "seq_len": SEQUENCE_LEN,
            "n_classes": N_CLASSES,
        }, MODELS_DIR / f"ids_{name}.pt")
        print(f"[保存] models/ids_{name}.pt (含 class_names/feature_cols/scaler)")

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"task": "多分类(8类)", "class_names": CLASS_NAMES,
                   "results": results,
                   "config": {
                       "train_sample": TRAIN_SAMPLE,
                       "test_size": 0.2,
                       "val_size_from_train": VAL_SIZE,
                       "max_epochs": MAX_EPOCHS,
                       "patience": PATIENCE,
                       "learning_rate": LEARNING_RATE,
                       "weight_decay": WEIGHT_DECAY,
                       "batch_size": BATCH_SIZE,
                       "sequence_len": SEQUENCE_LEN,
                   }}, f, ensure_ascii=False, indent=2)
    print(f"\n[结果] {REPORT}")


if __name__ == "__main__":
    main()
