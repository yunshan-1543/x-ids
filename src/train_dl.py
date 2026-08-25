# -*- coding: utf-8 -*-
"""
CICIDS2017 深度学习检测模型
===========================
模型: CNN-LSTM / Transformer(二分类 BENIGN vs Attack)
  特征 -> 归一化 -> 展平为一维序列 -> CNN 局部特征 -> LSTM/Transformer 时序 -> 分类

步骤:
  1. 加载清洗后数据,分层抽样训练集
  2. 特征标准化 + 维度重塑 + 验证集早停
  3. 训练 CNN-LSTM
  4. 训练 Transformer(可选,受数据量/设备控制)
  5. 保存模型与评估结果

输出:
  models/cnn_lstm.pt / models/transformer.pt
  reports/dl_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed" / "cicids2017_clean.parquet"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPORT = ROOT / "reports" / "dl_results.json"

LABEL_COL = "label"  # 清洗后 parquet 列名为小写
RANDOM_STATE = 42
SEQUENCE_LEN = 128      # 把特征切块当时间步用(LSTM/Transformer 输入)
MAX_EPOCHS = 50
PATIENCE = 8
MIN_DELTA = 1e-4
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SAMPLE = 300_000  # 分层抽样上限
VAL_SIZE = 0.1


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


def load_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(PROCESSED)
    y = (df[LABEL_COL] != "BENIGN").astype(int).values
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).values
    print(f"[加载] {len(X):,} 行 × {X.shape[1]} 特征")

    # 分层抽样
    idx, _ = train_test_split(np.arange(len(y)), train_size=TRAIN_SAMPLE,
                              random_state=RANDOM_STATE, stratify=y)
    X, y = X[idx], y[idx]
    print(f"[抽样] 训练集 {len(X):,} 条,Attack 占比 {y.mean()*100:.1f}%")
    return X, y


def split_standardize(X: np.ndarray, y: np.ndarray) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # CICIDS2017 特征长尾分布,标准化后仍有极端离群值(实测 max|z|≈535)。
    # 裁剪到 ±10 抑制梯度爆炸(仅影响 ~0.08% 单元格),避免训练 loss 变 NaN。
    clip = 10.0
    X_train = np.clip(X_train, -clip, clip)
    X_test = np.clip(X_test, -clip, clip)
    return X_train, X_test, y_train, y_test, sc


def to_sequence(X: np.ndarray, seq_len: int) -> np.ndarray:
    """把 [N, F] 展平,不足 seq_len 补零/超出截断,统一返回 [N, seq_len, 1]。"""
    n, f = X.shape
    flat = X.reshape(n, -1)
    if f < seq_len:
        flat = np.pad(flat, ((0, 0), (0, seq_len - f)), mode="constant")
    else:
        flat = flat[:, :seq_len]  # 超出则截断到 seq_len
    return flat.reshape(n, seq_len, 1)


class CNNLSTM(nn.Module):
    def __init__(self, n_features, seq_len, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):  # x: [B, seq_len, 1]
        x = x.permute(0, 2, 1)                 # [B, 1, seq_len]
        x = self.conv(x)                        # [B, 32, seq_len//2]
        x = x.permute(0, 2, 1)                 # [B, seq_len//2, 32]
        out, _ = self.lstm(x)                   # [B, seq_len//2, hidden*2]
        x = out[:, -1, :]                       # 取最后时间步
        return self.fc(x)


class TransformerModel(nn.Module):
    def __init__(self, n_features, seq_len, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=256,
            dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, 2)

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
                dev, max_epochs=MAX_EPOCHS, class_weight=None) -> dict:
    # 类别不均衡:传入类别权重(如逆频)可显著提升少数类(Attack)召回
    if class_weight is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(dev))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    model.to(dev)
    print(f"\n========== 训练 {name} ==========")

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
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
            print(f"  epoch {epoch+1}/{max_epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  best_epoch={best_epoch}")
        if wait >= PATIENCE:
            print(f"  [早停] 验证损失连续 {PATIENCE} 轮无显著改善,最佳 epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 评估
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            logits = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(yb.numpy())
    y_true = np.array(labels)
    y_pred = np.array(preds)
    res = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "best_epoch": best_epoch,
        "max_epochs": max_epochs,
        "early_stopping_patience": PATIENCE,
        "best_val_loss": round(float(best_val), 6),
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
    }
    print(f"  {name}: Accuracy={res['accuracy']:.4f} Precision={res['precision']:.4f} "
          f"Recall={res['recall']:.4f} F1={res['f1']:.4f}")
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    print(f"  [保存] models/{name}.pt")
    return res


def main():
    set_seed()
    dev = device()
    X, y = load_data()
    X_train, X_test, y_train, y_test, sc = split_standardize(X, y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=y_train)

    seq = to_sequence(X_train, SEQUENCE_LEN)
    seq_val = to_sequence(X_val, SEQUENCE_LEN)
    seq_test = to_sequence(X_test, SEQUENCE_LEN)
    n_feat = seq.shape[2]
    print(f"[序列] 训练 {seq.shape}  验证 {seq_val.shape}  测试 {seq_test.shape}")

    train_ds = TensorDataset(torch.FloatTensor(seq), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(seq_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(seq_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    results = []

    # 类别权重:逆频(Attack 占比约 17%),缓解失衡导致的"全预测 BENIGN"
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weight = torch.tensor([1.0, n_neg / n_pos])

    cnn_lstm = CNNLSTM(n_feat, SEQUENCE_LEN)
    res = train_model("cnn_lstm", cnn_lstm, train_loader, val_loader, test_loader, dev,
                      class_weight=class_weight)
    results.append(res)

    transformer = TransformerModel(n_feat, SEQUENCE_LEN)
    res = train_model("transformer", transformer, train_loader, val_loader, test_loader, dev,
                      class_weight=class_weight)
    results.append(res)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "note": "二分类,CNN-LSTM/Transformer; 训练集内部划分验证集并使用早停",
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
            }
        }, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[结果] {REPORT}")


if __name__ == "__main__":
    main()
