# -*- coding: utf-8 -*-
"""
CICIDS2017 XGBoost 多分类推理模块
=================================
职责:加载已训练的多分类模型(ids_xgb_multi.json),把任意键名的 CICIDS 特征字典
归一化到模型的特征顺序(69 维),输出攻击类别 + 置信度 + 全类别概率。

兼容输入:
  - 原始 CICIDS 列名,如 {"Flow Duration": 1234, "Total Fwd Packets": 20, ...}
  - 清洗后小写列名,如 {"flow_duration": 1234, "total_fwd_packets": 20, ...}
  - 按模型顺序排列的 69 元素列表
缺失特征按 0 填充(与训练清洗逻辑一致:未定义统计值记 0)。

用法:
    from inference import predict
    r = predict({"Flow Duration": 1234, "Total Fwd Packets": 20})
    # -> {"attack": "DDoS", "confidence": 0.99, "probabilities": {...}}
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
# 模型目录可用环境变量覆盖(供 Docker),默认 ../models/
_MODEL_DIR = Path(os.environ.get("MODELS_DIR", str(ROOT / "models")))
MODEL_PATH = _MODEL_DIR / "ids_xgb_multi.json"
META_PATH = _MODEL_DIR / "ids_xgb_multi.meta.json"

_class_names: list[str] = []
_feature_cols: list[str] = []
_booster: xgb.Booster | None = None


def _normalize_key(key: str) -> str:
    """与 data_clean.standardize_columns 一致的列名归一化,兼容原始 CICIDS 写法。"""
    s = key.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"\.+", "_", s)
    return s


def load_model() -> tuple[list[str], list[str]]:
    """加载 XGBoost 模型与元信息(类别顺序 / 特征顺序)。"""
    global _class_names, _feature_cols, _booster
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    _class_names = meta["class_names"]
    _feature_cols = meta["feature_cols"]
    _booster = xgb.Booster()
    _booster.load_model(str(MODEL_PATH))
    return _class_names, _feature_cols


def _ensure_loaded() -> None:
    if _booster is None:
        load_model()


def _to_row(features) -> np.ndarray:
    """把 dict / 列表规整为 69 维行向量(缺失补 0)。"""
    row = np.zeros(len(_feature_cols), dtype=np.float32)
    if isinstance(features, (list, tuple)):
        vals = list(features)
        if len(vals) != len(_feature_cols):
            raise ValueError(
                f"特征数量应为 {len(_feature_cols)},实际收到 {len(vals)}。"
                "若传 dict,请使用 CICIDS 特征名。")
        row[:] = vals
        return row
    if isinstance(features, dict):
        norm = {_normalize_key(str(k)): v for k, v in features.items()}
        for i, col in enumerate(_feature_cols):
            if col in norm:
                try:
                    row[i] = float(norm[col])
                except (TypeError, ValueError):
                    pass  # 非法值按 0 处理
        return row
    raise TypeError("features 应为 dict 或 list")


def predict(features) -> dict:
    """推理单条流特征,返回 {attack, confidence, probabilities}。"""
    _ensure_loaded()
    row = _to_row(features)
    dmat = xgb.DMatrix(row.reshape(1, -1), feature_names=_feature_cols)
    prob = _booster.predict(dmat)[0]          # shape=(n_classes,),softmax 概率
    idx = int(np.argmax(prob))
    return {
        "attack": _class_names[idx],
        "confidence": round(float(prob[idx]), 4),
        "probabilities": {name: round(float(prob[i]), 4)
                          for i, name in enumerate(_class_names)},
    }


if __name__ == "__main__":
    # 自检:输出模型信息
    names, cols = load_model()
    print(f"模型加载成功: {len(cols)} 个特征, {len(names)} 个类别")
    print(f"类别: {names}")
