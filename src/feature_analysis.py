# -*- coding: utf-8 -*-
"""
CICIDS2017 特征分析与可视化
==========================
输入: processed/cicids2017_clean.parquet (清洗后的全量数据)
输出:
  reports/figures/*.png
  reports/feature_analysis_report.md

分析内容:
  1. 类别分布
  2. 特征统计概要(缺失/唯一值/取值范围)
  3. 标签分布可视化
  4. 特征相关性热力图(抽样)
  5. 特征与标签的区分度初探(Top 特征)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端,避免 Windows GUI 阻塞
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------- 中文字体处理 ----------
plt.rcParams["axes.unicode_minus"] = False
# Windows 常见中文字体
import matplotlib.font_manager as fm

_chinese_fonts = [f.name for f in fm.fontManager.ttflist
                  if f.name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi")]
if _chinese_fonts:
    plt.rcParams["font.sans-serif"] = [_chinese_fonts[0]] + plt.rcParams["font.sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed" / "cicids2017_clean.parquet"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT = ROOT / "reports" / "feature_analysis_report.md"

LABEL_COL = "label"  # 清洗后 parquet 列名为小写
SAMPLE_SIZE = 100_000  # 相关性/可视化抽样


def load() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED)
    print(f"[加载] {len(df):,} 行 × {df.shape[1]} 列")
    return df


def save(fig, name: str):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] reports/figures/{name}")


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = df[LABEL_COL].value_counts().reset_index()
    dist.columns = ["label", "count"]
    dist["percent"] = dist["count"] / dist["count"].sum() * 100
    return dist


def plot_class_distribution(dist: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=dist, x="label", y="count", ax=ax, color="#4C72B0")
    ax.set_title("CICIDS2017 清洗后各类别样本数")
    ax.set_ylabel("样本数")
    ax.set_xlabel("类别")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i, row in dist.iterrows():
        ax.text(i, row["count"], f"{row['count']:,}", ha="center", va="bottom", fontsize=8)
    save(fig, "class_distribution.png")


def binary_balance(df: pd.DataFrame, dist: pd.DataFrame):
    benign = int(dist.loc[dist["label"] == "BENIGN", "count"].sum())
    attack = int(dist.loc[dist["label"] != "BENIGN", "count"].sum())
    print(f"\n[二元] BENIGN={benign:,}  Attack={attack:,}  失衡比=1:{attack/max(benign,1):.1f}")
    return benign, attack


def plot_correlation(df: pd.DataFrame):
    """抽样后计算特征相关矩阵并画热力图(阈值过滤以保持可读)。"""
    feat = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    sample = feat.sample(min(SAMPLE_SIZE, len(feat)), random_state=42)
    corr = sample.corr()

    # 只画 |r|>=0.7 的部分,否则 78×78 热力图看不清
    mask = np.abs(corr) < 0.7
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title(f"特征相关性(|r|>=0.7,抽样 {SAMPLE_SIZE:,})")
    save(fig, "feature_correlation.png")

    # 输出高度相关对,用于去冗余
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= 0.95:
                pairs.append((cols[i], cols[j], round(float(r), 3)))
    pairs.sort(key=lambda x: -abs(x[2]))
    print(f"\n[相关] |r|>=0.95 的特征对: {len(pairs)} 个(可考虑去冗余)")
    for a, b, r in pairs[:15]:
        print(f"   {a}  <->  {b}  r={r}")
    return pairs


def plot_top_features(df: pd.DataFrame, top_n: int = 20):
    """用树模型重要性初探(Train test 小样本),给出 Top 特征。"""
    from sklearn.ensemble import RandomForestClassifier

    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)
    y = (sample[LABEL_COL] != "BENIGN").astype(int)
    X = sample.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    X = X.fillna(0)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                random_state=42, min_samples_leaf=5)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 12))
    top = imp.head(top_n)[::-1]
    ax.barh(top.index, top.values, color="#55A868")
    ax.set_title("Top 特征重要性(RandomForest, 二分类)")
    ax.set_xlabel("importance")
    save(fig, "feature_importance_top.png")
    return imp


def main():
    df = load()
    dist = class_distribution(df)
    print("\n========== 类别分布 ==========")
    print(dist.to_string(index=False))

    plot_class_distribution(dist)
    binary_balance(df, dist)
    pairs = plot_correlation(df)
    imp = plot_top_features(df)

    # 汇总报告
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# CICIDS2017 特征分析报告\n\n")
        f.write(f"- 数据规模: {len(df):,} 行 × {df.shape[1]} 列\n")
        f.write(f"- 类别数: {len(dist)} 种(BENIGN + {len(dist)-1} 种攻击)\n\n")
        f.write("## 类别分布\n\n")
        f.write("| 类别 | 样本数 | 占比 |\n|---|---|---|\n")
        for _, r in dist.iterrows():
            f.write(f"| {r['label']} | {r['count']:,} | {r['percent']:.2f}% |\n")
        f.write(f"\n## 高度相关特征对(|r|>=0.95,共 {len(pairs)} 对)\n\n")
        for a, b, r in pairs[:20]:
            f.write(f"- {a} ↔ {b} (r={r})\n")
        f.write("\n## Top 20 特征重要性\n\n")
        for name, v in imp.head(20).items():
            f.write(f"- {name}: {v:.4f}\n")
    print(f"\n[报告] {REPORT}")


if __name__ == "__main__":
    main()
