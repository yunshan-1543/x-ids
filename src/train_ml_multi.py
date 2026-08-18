# -*- coding: utf-8 -*-
"""
CICIDS2017 多分类机器学习基线(8 类)
===================================
类别: BENIGN / DDoS / PortScan / Botnet / BruteForce / WebAttack / DoS / Infiltration
模型: RandomForest(class_weight='balanced_subsample')
      XGBoost(拟合时传逆频 sample_weight)
步骤:
  1. 加载清洗 parquet,按 label_utils 分组为 8 大类
  2. 分层抽样 30 万,切 80/20(与二分类同一随机种子,可对比)
  3. 训练 + 全类别评估(Accuracy / Macro-F1 / Weighted-F1 / per-class 报告)
  4. 保存模型(含类别顺序与特征列名,便于推理时对齐)

输出:
  models/ids_rf_multi.pkl            (dict: model + class_names + feature_cols)
  models/ids_xgb_multi.json          (XGBoost 原生格式)
  models/ids_xgb_multi.meta.json     (class_names + feature_cols)
  reports/ml_multi_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from label_utils import CLASS_NAMES, group_labels, label_to_idx

PROCESSED = ROOT / "processed" / "cicids2017_clean.parquet"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPORT = ROOT / "reports" / "ml_multi_results.json"

TRAIN_SAMPLE = 300_000
RANDOM_STATE = 42


def evaluate(name: str, model, X_test, y_test, seconds: float) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"\n===== {name} =====")
    print(f"Accuracy={acc:.4f}  Macro-F1={macro:.4f}  Weighted-F1={weighted:.4f}  "
          f"({seconds:.1f}s)")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                zero_division=0, digits=4))
    return {
        "model": name,
        "task": "多分类(8类)",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro), 4),
        "weighted_f1": round(float(weighted), 4),
        "train_seconds": round(float(seconds), 1),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = pd.read_parquet(PROCESSED)
    print(f"[加载] {len(df):,} 行 × {df.shape[1]} 列")

    y_group = group_labels(df["label"])
    print(f"[分组] 大类分布:\n{y_group.value_counts().to_string()}")
    y = label_to_idx(y_group).values
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).fillna(0)
    feature_cols = list(X.columns)

    # 分层抽样 + 切分(与二分类同一 42 种子)
    idx, _ = train_test_split(np.arange(len(y)), train_size=TRAIN_SAMPLE,
                              random_state=RANDOM_STATE, stratify=y)
    X, y = X.iloc[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"[切分] 训练 {len(X_train):,} / 测试 {len(X_test):,}")

    # 逆频样本权重(用于 XGBoost 缓解类别失衡)
    counts = np.bincount(y_train)
    sw = np.array([1.0 / counts[c] for c in y_train])
    sw = sw / sw.sum() * len(y_train)

    results = []

    # ---------- RandomForest(内置类别平衡) ----------
    print("\n[训练] RandomForest(多分类) ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1,
        random_state=RANDOM_STATE, min_samples_leaf=2,
        class_weight="balanced_subsample")
    rf.fit(X_train, y_train)
    results.append(evaluate("RandomForest", rf, X_test, y_test, time.time() - t0))
    pd.to_pickle({"model": rf, "class_names": CLASS_NAMES,
                  "feature_cols": feature_cols}, MODELS_DIR / "ids_rf_multi.pkl")
    print("[保存] models/ids_rf_multi.pkl")

    # ---------- XGBoost ----------
    try:
        import xgboost as xgb
        print("\n[训练] XGBoost(多分类) ...")
        t0 = time.time()
        xg = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            tree_method="hist")
        xg.fit(X_train, y_train, sample_weight=sw)
        results.append(evaluate("XGBoost", xg, X_test, y_test, time.time() - t0))
        xg.save_model(str(MODELS_DIR / "ids_xgb_multi.json"))
        with open(MODELS_DIR / "ids_xgb_multi.meta.json", "w", encoding="utf-8") as f:
            json.dump({"class_names": CLASS_NAMES, "feature_cols": feature_cols},
                      f, ensure_ascii=False, indent=2)
        print("[保存] models/ids_xgb_multi.json + meta")
    except ImportError:
        print("\n[跳过] xgboost 未安装,跳过 XGBoost")

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"task": "多分类(8类)", "class_names": CLASS_NAMES,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[结果] {REPORT}")


if __name__ == "__main__":
    main()
