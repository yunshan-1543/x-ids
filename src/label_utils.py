# -*- coding: utf-8 -*-
"""
CICIDS2017 标签分组工具
======================
原始 15 类标签 -> 8 大类(用于多分类):

    BENIGN / DDoS / PortScan / Botnet / BruteForce / WebAttack / DoS / Infiltration

注意:本版本 CICIDS2017 数据中 Web Attack 标签的横线字符已被损坏成
U+FFFD 替换字符(�),因此用 _normalize_label 把所有横线变体(– — �)归一为
ASCII '-' 再做匹配,避免硬编码损坏字符。

唯一来源:后续多分类脚本都从这里取分组与类别顺序。
"""
from __future__ import annotations

import pandas as pd

# 8 大类,顺序即类别索引顺序(0=BENIGN)。改动前请确认与模型输出维度一致。
CLASS_NAMES = [
    "BENIGN", "DDoS", "PortScan", "Botnet",
    "BruteForce", "WebAttack", "DoS", "Infiltration",
]

# 原始 15 类 -> 大类(键用 ASCII '-' 写法,匹配前会先归一化)
ATTACK_GROUPS = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Botnet",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "Heartbleed": "DoS",
    "Infiltration": "Infiltration",
}

# 8 大类内部的原始标签 -> 细类名(用于多分类报告的可读性)
GROUP_MEMBERS = {
    "BENIGN": ["BENIGN"],
    "DDoS": ["DDoS"],
    "PortScan": ["PortScan"],
    "Botnet": ["Bot"],
    "BruteForce": ["FTP-Patator", "SSH-Patator"],
    "WebAttack": ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"],
    "DoS": ["DoS slowloris", "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye", "Heartbleed"],
    "Infiltration": ["Infiltration"],
}


def _normalize_label(s: str) -> str:
    """把横线变体(– — �)与首尾空白归一为 ASCII '-',统一匹配口径。"""
    return (s.strip()
            .replace("�", "-")
            .replace("–", "-")   # en-dash
            .replace("—", "-"))  # em-dash


def group_labels(series: pd.Series) -> pd.Series:
    """原始 Label 列 -> 大类。未知标签归为 'Unknown'(正常不应出现)。"""
    norm = series.astype(str).map(_normalize_label)
    return norm.map(ATTACK_GROUPS).fillna("Unknown")


def label_to_idx(series: pd.Series) -> pd.Series:
    """大类 -> 整数索引(顺序由 CLASS_NAMES 决定)。"""
    mapping = {name: i for i, name in enumerate(CLASS_NAMES)}
    return series.map(mapping).astype(int)
