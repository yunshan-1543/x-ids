# -*- coding: utf-8 -*-
"""
CICIDS2017 数据清洗脚本
======================
流程:
  1. 逐文件读取原始 CSV(标准表头,共 78 特征 + Label)
  2. 列名标准化(去首尾空格,空格转下划线,处理重复列)
  3. 处理 Inf -> NaN
  4. 删除全空列 / 常量列
  5. 标签规范化(去空白、统一大小写、归类攻击大类)
  6. 删除含 NaN 的行(记录删除量)
  7. 拼接全部日期文件,保存清洗后的 Parquet 与统计摘要

输出:
  processed/cicids2017_clean.parquet   (合并清洗后全量数据)
  processed/cicids2017_clean_summary.json  (清洗前后统计)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "MachineLearningCVE"
PROCESSED_DIR = ROOT / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# 8 个日期文件,按时间顺序
FILE_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# 攻击大类分组已抽到 src/label_utils.py(含 U+FFFD 归一化匹配),这里不再重复。

LABEL_COL = "label"  # 列名经 standardize_columns 后已转小写(原始表头为 ' Label')


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名标准化:去首尾空格,连续空格/点转下划线,转小写。"""
    mapping = {}
    for c in df.columns:
        c2 = c.strip().lower()
        c2 = re.sub(r"\s+", "_", c2)
        c2 = re.sub(r"\.+", "_", c2)
        mapping[c] = c2
    df = df.rename(columns=mapping)
    # 去重列名:保留第一个,其余丢弃。pandas 给重复列自动加 .1 后缀,
    # 经上面规范化后变为 _1 形式(如 fwd_header_length_1),先去掉数字后缀再比较。
    seen: dict[str, str] = {}
    drop_cols: list[str] = []
    for c in df.columns:
        base = re.sub(r"_\d+$", "", c)
        if base in seen:
            drop_cols.append(c)
        else:
            seen[base] = c
    return df.drop(columns=drop_cols)


def clean_label(series: pd.Series) -> pd.Series:
    """标签规范化:去空白,保持原始大小写,CICIDS 标签都是标准写法。"""
    return series.astype(str).str.strip()


def is_constant(col: pd.Series) -> bool:
    """判断一列是否常量(全 0 或无变化,对分类没有信息量)。"""
    if col.nunique(dropna=True) <= 1:
        return True
    return False


def process_file(path: Path, stats: dict) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    stats["files"][path.name] = {"raw_rows": len(df)}

    # 1) 列名标准化
    df = standardize_columns(df)

    # 2) 特征列与标签列分离
    if LABEL_COL not in df.columns:
        raise RuntimeError(f"{path.name}: 找不到 Label 列")
    label = clean_label(df[LABEL_COL])
    features = df.drop(columns=[LABEL_COL])

    # 3) 全部特征转 float
    features = features.apply(pd.to_numeric, errors="coerce")
    # 3.5) Inf / -Inf -> NaN(CICIDS2017 的 Flow Bytes/s 等列含 'Infinity' 字符串,
    #      pd.to_numeric 会转成 inf 而非 NaN,需显式替换,否则模型会收到 inf 特征)
    features = features.replace([np.inf, -np.inf], np.nan)

    # 4) 注意:全空/常量列不在单文件里删,统一放到合并后全局判断。
    #    否则跨文件列集合不一致(如 Infiltration 文件的 fwd_urg_flags/cwe_flag_count
    #    非常量而其他 7 个文件为常量),合并后会给缺失方灌入 NaN。

    # 5) 删除标签为空的行
    valid_label = label.notna() & (label != "") & (label != "nan")
    n_dropped_label = int((~valid_label).sum())

    # 7) 删除含 NaN 的行
    mask = features.notna().all(axis=1) & valid_label.values
    n_dropped_nan = int((~mask).sum()) - n_dropped_label

    features = features[mask]
    label = label[mask]
    df_clean = features.copy()
    df_clean[LABEL_COL] = label

    stats["files"][path.name].update(
        {
            "dropped_bad_label": n_dropped_label,
            "dropped_nan": n_dropped_nan,
            "clean_rows": len(df_clean),
            "seconds": round(time.time() - t0, 1),
        }
    )
    return df_clean


def main() -> None:
    # Windows 控制台默认编码非 UTF-8,统一输出编码避免中文乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    all_parts: list[pd.DataFrame] = []
    stats: dict = {"files": {}, "total_raw": 0, "total_clean": 0}

    for name in FILE_ORDER:
        path = DATA_DIR / name
        if not path.exists():
            print(f"[跳过] 不存在: {path.name}")
            continue
        print(f"[处理] {path.name} ...")
        part = process_file(path, stats)
        all_parts.append(part)
        print(f"        -> {len(part):,} 行")

    # 汇总行数(重新统计 raw)
    for f in stats["files"].values():
        stats["total_raw"] += f["raw_rows"]
    stats["total_clean"] = int(sum(len(p) for p in all_parts))

    if not all_parts:
        raise RuntimeError("没有读取到任何数据文件")

    merged = pd.concat(all_parts, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)
    stats["after_dedup"] = len(merged)

    # 合并后统一删除全空列 / 常量列(保证跨文件列集合一致,消除 per-file 删除导致的 NaN)
    feat_cols = [c for c in merged.columns if c != LABEL_COL]
    empty_cols = [c for c in feat_cols if merged[c].isna().all()]
    const_cols = [c for c in feat_cols if is_constant(merged[c])]
    merged = merged.drop(columns=empty_cols + const_cols)
    stats["removed_columns"] = {"empty": empty_cols, "constant": const_cols}

    # 删除残余含 NaN 的行(应极少,防御性)
    n_nan_after_merge = int(merged.isna().any(axis=1).sum())
    if n_nan_after_merge:
        merged = merged.dropna().reset_index(drop=True)
    stats["dropped_nan_rows_after_merge"] = n_nan_after_merge

    # 标签分布
    label_counts = merged[LABEL_COL].value_counts()
    stats["label_distribution"] = {
        str(k): int(v) for k, v in label_counts.items()
    }
    stats["n_attack_types"] = int((label_counts.index != "BENIGN").sum())
    stats["benign_rows"] = int((merged[LABEL_COL] == "BENIGN").sum())
    stats["attack_rows"] = stats["after_dedup"] - stats["benign_rows"]
    stats["columns"] = [c for c in merged.columns if c != LABEL_COL]

    # 保存 Parquet
    out_path = PROCESSED_DIR / "cicids2017_clean.parquet"
    merged.to_parquet(out_path, index=False)

    # 保存统计
    with open(PROCESSED_DIR / "cicids2017_clean_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n========== 清洗完成 ==========")
    print(f"原始行数(含脏数据): {stats['total_raw']:,}")
    print(f"清洗后行数(去重前): {stats['total_clean']:,}")
    print(f"去重后行数:        {stats['after_dedup']:,}")
    print(f"BENIGN:            {stats['benign_rows']:,}")
    print(f"Attack:            {stats['attack_rows']:,} ({stats['n_attack_types']} 种攻击)")
    print(f"特征列数:          {len(stats['columns'])}")
    print(f"保存至:            {out_path}")
    print(f"标签分布: {label_counts.to_dict()}")


if __name__ == "__main__":
    main()
