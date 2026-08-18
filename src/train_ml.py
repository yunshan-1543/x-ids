# -*- coding: utf-8 -*-
"""
CICIDS2017 机器学习基线模型
===========================
模型: RandomForest / XGBoost (二分类 BENIGN vs Attack)
步骤:
  1. 加载清洗后的 Parquet
  2. 用全部特征(数值化),二分类标签
  3. 按类别分层抽样构造平衡训练集(控制内存与训练时间)
  4. 训练 + 评估(Accuracy / Precision / Recall / F1)
  5. 保存模型与评估结果

输出:
  models/rf_baseline.pkl, models/xgb_baseline.pkl
  reports/ml_baseline_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed" / "cicids2017_clean.parquet"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPORT = ROOT / "reports" / "ml_baseline_results.json"

LABEL_COL = "label"  # 清洗后 parquet 列名为小写
TRAIN_SAMPLE = 300_000   # 每折总体样本上限(分层抽样)
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED)
    print(f"[加载] {len(df):,} 行 × {df.shape[1]} 列")
    return df


def make_binary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = (df[LABEL_COL] != "BENIGN").astype(int)
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    # 全 0 常量列已在上游删除;此处仅兜底
    X = X.fillna(0)
    return X, y


def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int) -> tuple[pd.DataFrame, pd.Series]:
    """分层抽样:按正负类比例抽 n 条,保证两折都有。"""
    idx, _ = train_test_split(
        np.arange(len(y)), train_size=n, random_state=RANDOM_STATE, stratify=y)
    return X.iloc[idx], y.iloc[idx]


def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    res = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
    }
    print(f"\n===== {name} =====")
    print(f"Accuracy ={res['accuracy']:.4f}  Precision={res['precision']:.4f}  "
          f"Recall={res['recall']:.4f}  F1={res['f1']:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵(行=真实, 列=预测):")
    print(f"           预测BENIGN  预测Attack")
    print(f"真实BENIGN  {cm[0,0]:>8d}   {cm[0,1]:>8d}")
    print(f"真实Attack  {cm[1,0]:>8d}   {cm[1,1]:>8d}")
    return res


def main():
    df = load_data()
    X, y = make_binary(df)
    print(f"[特征] {X.shape[1]} 个数值特征;Attack 占比 {y.mean()*100:.1f}%")

    # 分层抽样训练子集(数据规模大,先控制到 30 万)
    X_s, y_s = stratified_sample(X, y, TRAIN_SAMPLE)
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.2, random_state=RANDOM_STATE, stratify=y_s)
    print(f"[切分] 训练 {len(X_train):,} / 测试 {len(X_test):,}")

    results = []

    # ---------- RandomForest ----------
    print("\n[训练] RandomForest ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1,
        random_state=RANDOM_STATE, min_samples_leaf=2)
    rf.fit(X_train, y_train)
    res = evaluate("RandomForest", rf, X_test, y_test)
    res["train_seconds"] = round(time.time() - t0, 1)
    results.append(res)
    pd.to_pickle(rf, MODELS_DIR / "rf_baseline.pkl")
    print(f"  [保存] models/rf_baseline.pkl")

    # ---------- XGBoost ----------
    try:
        import xgboost as xgb
        print("\n[训练] XGBoost ...")
        t0 = time.time()
        xg = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_STATE, eval_metric="logloss",
            tree_method="hist")
        xg.fit(X_train, y_train)
        res = evaluate("XGBoost", xg, X_test, y_test)
        res["train_seconds"] = round(time.time() - t0, 1)
        results.append(res)
        xg.save_model(str(MODELS_DIR / "xgb_baseline.json"))
        print(f"  [保存] models/xgb_baseline.json")
    except ImportError:
        print("\n[跳过] xgboost 未安装,跳过 XGBoost")

    # 汇总保存
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"results": results, "note": "二分类 BENIGN vs Attack"}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[结果] {REPORT}")


if __name__ == "__main__":
    main()
