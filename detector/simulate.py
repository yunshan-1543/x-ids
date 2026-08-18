# -*- coding: utf-8 -*-
"""
模拟流量事件源
==============
从清洗后的真实 CICIDS2017 数据按 8 大类各抽样少量样本作为"特征池",
按可配置速率随机取样本(加少量噪声避免重复),走真实模型推理,
生成带置信度的检测事件。

用途:在没有 Npcap / 真实攻击流量时,也能完整演示
"流量 -> 特征 -> 模型 -> 攻击类型+置信度 -> 面板" 整条链路。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "detector"))

from label_utils import CLASS_NAMES, group_labels
import inference as inf   # 模块引用,避免 from-import 在 load_model 前捕获空列表
from inference import predict

PROCESSED = ROOT / "processed" / "cicids2017_clean.parquet"
N_PER_CLASS = 60  # 每个大类抽 60 条真实样本进池

_pool: dict[str, np.ndarray] | None = None

# 事件类别权重:演示时让攻击更醒目(30% 正常 + 70% 攻击)
WEIGHTS = np.array([0.30] + [0.70 / (len(CLASS_NAMES) - 1)] * (len(CLASS_NAMES) - 1))


def build_pool() -> dict[str, np.ndarray]:
    """从真实数据构建 {大类: (k, 69) 特征矩阵}。只加载一次并缓存。"""
    global _pool
    if _pool is not None:
        return _pool
    df = pd.read_parquet(PROCESSED)
    g = group_labels(df["label"])
    df = df[g.isin(CLASS_NAMES)]
    df = df.assign(group=g[g.isin(CLASS_NAMES)].values)
    pool = {}
    for name in CLASS_NAMES:
        sub = df[df["group"] == name]
        if len(sub) > N_PER_CLASS:
            sub = sub.sample(N_PER_CLASS, random_state=42)
        pool[name] = sub[inf._feature_cols].to_numpy(dtype=np.float32)
    _pool = pool
    return pool


# 攻击类型 -> 典型目标端口 / 常用源-目标 IP(用于告警详情的"谁攻击谁")
_ATTACK_PORTS = {"DDoS": 80, "PortScan": 80, "Botnet": 4444, "BruteForce": 22,
                 "WebAttack": 80, "DoS": 80, "Infiltration": 445, "BENIGN": 443}
_SRC_IPS = ["192.168.1.8", "192.168.1.105", "172.16.3.21", "10.10.2.7", "192.168.1.15"]
_DST_IPS = ["10.0.0.20", "10.0.0.5", "192.168.1.1", "172.16.0.10"]


def _make_meta(attack: str, rng: np.random.Generator) -> dict:
    return {"src_ip": rng.choice(_SRC_IPS), "dst_ip": rng.choice(_DST_IPS),
            "dst_port": int(_ATTACK_PORTS[attack]), "proto": "TCP"}


def next_event(rng: np.random.Generator) -> tuple[str, float, np.ndarray, dict]:
    """随机取一条真实样本 -> 模型推理 -> (attack, confidence, 特征向量, 流元数据)。"""
    pool = build_pool()
    name = rng.choice(CLASS_NAMES, p=WEIGHTS)
    vec = pool[name][rng.integers(len(pool[name]))].copy()
    # 给每个特征按其自身量级加少量相对噪声,避免反复出现完全相同的事件。
    # 注意:不能用跨特征的全局 std 当噪声尺度 —— 它会被量级巨大的特征
    # (flow_duration / fwd_iat_total 等可达 1e7~1e8)主导,使小量级特征
    # (端口/标志位/包长等)叠加数万级噪声,导致模型几乎把所有攻击样本误判为 BENIGN。
    scale = np.abs(vec) * 0.001 + 1e-4
    vec = vec + rng.normal(0, scale, size=vec.shape).astype(np.float32)
    r = predict(vec.tolist())
    return r["attack"], r["confidence"], vec, _make_meta(r["attack"], rng)


def run_simulator(rate_hz: float, emit, stop_event: threading.Event) -> None:
    """后台线程:按 rate_hz 持续生成事件,回调 emit(attack, conf, src, features, meta)。
    单次生成失败仅打印错误并继续,避免整个模拟线程静默退出。"""
    rng = np.random.default_rng(42)
    interval = 1.0 / rate_hz
    while not stop_event.is_set():
        try:
            attack, conf, vec, meta = next_event(rng)
            emit(attack, conf, src="simulate", features=vec, meta=meta)
        except Exception as exc:  # noqa: BLE001
            print(f"[simulate] 生成失败: {exc}", flush=True)
        stop_event.wait(interval)
