# -*- coding: utf-8 -*-
"""
SHAP 模型解释
=============
用 shap.TreeExplainer 对 XGBoost 多分类模型做特征归因,
输出预测类别下"影响最大的特征"及贡献方向,供告警详情页展示。

兼容性处理:
  1) xgboost 多分类把 base_score 序列化为数组字符串 '[-8.6e-5,...]',
     SHAP 加载器 float() 解析失败 -> monkeypatch decode_ubjson_buffer 改为标量
  2) shap_values 对单样本多分类可能返回 (1, n_features, n_classes),
     统一提取预测类的 (n_features,) 贡献

示例响应:
  {"attack":"DDoS","confidence":0.9989,
   "top_features":[
     {"feature":"flow_packets/s","contribution":3.21,"impact":"push_toward"}, ...]}

依赖: pip install shap(shap 为懒加载,未安装时服务照常运行,/explain 返回提示)
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

import inference as inf

_explainer = None
_PATCHED = False


def _patch_ubjson_base_score() -> None:
    """xgboost 多分类 base_score 序列化兼容补丁(进程内只打一次)。"""
    global _PATCHED
    if _PATCHED:
        return
    import shap.explainers._tree as _stree

    _orig = _stree.decode_ubjson_buffer

    def _patched(fd, *a, **kw):
        j = _orig(fd, *a, **kw)
        try:
            lm = j["learner"]["learner_model_param"]
            if isinstance(lm.get("base_score"), str) and lm["base_score"].startswith("["):
                lm["base_score"] = "0.5"
        except Exception:  # noqa: BLE001
            pass
        return j

    _stree.decode_ubjson_buffer = _patched
    _PATCHED = True


def _get_explainer():
    global _explainer
    if _explainer is None:
        import shap  # 懒加载,避免 shap 未安装时阻塞整个服务启动
        _patch_ubjson_base_score()
        _explainer = shap.TreeExplainer(inf._booster, feature_names=inf._feature_cols)
    return _explainer


def explain(features, top_n: int = 5) -> dict:
    """特征归因:返回 {attack, confidence, top_features}(Top5, 含影响占比 share_pct)。"""
    row = inf._to_row(features)
    prob = inf._booster.predict(
        xgb.DMatrix(row.reshape(1, -1), feature_names=inf._feature_cols))[0]
    pred = int(np.argmax(prob))

    sv = _get_explainer().shap_values(row.reshape(1, -1))
    if isinstance(sv, list):                      # 每类一个贡献数组
        contrib = np.asarray(sv[pred]).reshape(-1)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:                         # 兼容两种布局
            if arr.shape[0] == 1:                 # (1, n_features, n_classes)
                contrib = arr[0, :, pred]
            else:                                 # (n_classes, 1, n_features)
                contrib = arr[pred].reshape(-1)
        else:                                     # 单类
            contrib = arr.reshape(-1)

    order = np.argsort(-np.abs(contrib))
    total_mag = float(np.abs(contrib).sum()) or 1.0   # 总影响幅度,归一化为占比
    top = [{
        "feature": inf._feature_cols[i],
        "contribution": round(float(contrib[i]), 4),
        "share_pct": round(float(abs(contrib[i])) / total_mag * 100, 1),
        "impact": "push_toward" if contrib[i] > 0 else "push_against",
    } for i in order[:top_n]]
    return {
        "attack": inf._class_names[pred],
        "confidence": round(float(prob[pred]), 4),
        "top_features": top,
    }
