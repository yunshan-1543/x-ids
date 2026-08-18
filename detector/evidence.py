# -*- coding: utf-8 -*-
"""
告警证据与处置建议
==================
把模型结论(XGBoost 判定某攻击类型)+ 流特征翻译成安全运营可读的:
  - 风险等级(HIGH / MEDIUM / LOW)
  - 检测依据(基于流特征的启发式证据,与模型结论互补)
  - 处置建议(按攻击类型的操作建议)

说明:检测依据是"规则化解释",与 SHAP 特征归因(/explain)互补——
前者讲"流的哪些行为异常",后者讲"模型的哪些特征驱动了判定"。
"""
from __future__ import annotations

RISK_LEVEL: dict[str, str] = {
    "DDoS": "HIGH", "DoS": "HIGH", "Infiltration": "HIGH", "Botnet": "HIGH",
    "PortScan": "MEDIUM", "BruteForce": "MEDIUM", "WebAttack": "MEDIUM",
    "BENIGN": "LOW",
}

RECOMMENDATIONS: dict[str, list[str]] = {
    "DDoS": [
        "限制来源 IP 访问速率,必要时封禁来源",
        "检查异常流量特征与来源分布",
        "启用 DDoS 防护清洗(限流 / 黑洞路由)",
    ],
    "DoS": [
        "提升目标服务的连接数/带宽上限",
        "启用流量清洗或 CDN 缓解",
        "核查攻击来源,按需封禁",
    ],
    "PortScan": [
        "检查扫描源 IP 的访问行为",
        "增加访问控制(防火墙规则),限制连接频率",
        "关闭未使用的服务端口,收敛暴露面",
    ],
    "Botnet": [
        "隔离疑似受控主机,排查恶意软件",
        "检查与 C2 服务器的外联通信",
        "重置相关主机凭证并更新补丁",
    ],
    "BruteForce": [
        "检查登录失败次数与来源 IP",
        "启用登录失败锁定与二次验证",
        "更换弱口令,实施强密码策略",
    ],
    "WebAttack": [
        "调取 Web 日志定位攻击载荷",
        "升级/调整 WAF 防护规则",
        "及时修补 Web 组件已知漏洞",
    ],
    "Infiltration": [
        "立即隔离涉事主机,阻断横向移动",
        "审计关键账号与敏感数据访问记录",
        "全网排查同类入侵痕迹",
    ],
    "BENIGN": [
        "保持现有安全策略,持续监控",
    ],
}


def risk_level(attack: str) -> str:
    return RISK_LEVEL.get(attack, "LOW")


def evidence(features: dict, attack: str) -> list[str]:
    """基于流特征的规则化检测依据(与模型结论互补)。"""
    out: list[str] = []

    def f(name: str, default=0.0):
        v = features.get(name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    fps = f("flow_packets/s")
    syn = int(f("syn_flag_count"))
    n_fwd = int(f("total_fwd_packets"))
    n_bwd = int(f("total_backward_packets"))
    n_pkt = n_fwd + n_bwd
    dur_us = f("flow_duration")
    syn_ratio = syn / n_pkt if n_pkt else 0.0

    if fps > 1000:
        out.append(f"单位时间包数量异常({fps:.0f} pkt/s,疑似流量突增)")
    elif fps > 300:
        out.append(f"包速率偏高({fps:.0f} pkt/s)")
    if syn_ratio > 0.3:
        out.append(f"SYN 请求比例过高({syn_ratio:.0%},疑似洪泛)")
    if n_pkt > 200 and 0 < dur_us < 2e6:
        out.append("短时高频大量数据包(符合突发攻击特征)")
    if attack == "PortScan" and n_bwd == 0 and n_pkt > 10:
        out.append("大量单向探测包且无回包")
    if attack == "BruteForce" and n_pkt > 50:
        out.append("长连接大量重复请求(疑似暴力尝试)")
    if attack == "DDoS" and fps > 800:
        out.append("高包速率叠加高 SYN 比例,符合洪泛特征")
    if attack == "BENIGN":
        out.append("包速率与字节量处于正常基线范围")

    if not out:
        out.append(f"XGBoost 多分类模型判定为「{attack}」且概率最高")
    return out


def recommendations(attack: str) -> list[str]:
    return RECOMMENDATIONS.get(attack, ["核查事件来源,结合现场日志进一步研判"])


def is_syn_flood(features: dict, meta: dict | None = None) -> bool:
    """行为启发式:单流大量 SYN 包、高包速率且无回包 -> 疑似 SYN 洪泛。

    纯单向 SYN 洪泛不在模型训练的 CICIDS 分布内,靠该规则补足(与模型互补)。
    """
    n_fwd = int(features.get("total_fwd_packets", 0) or 0)
    syn = int(features.get("syn_flag_count", 0) or 0)
    fps = float(features.get("flow_packets/s", 0) or 0)
    return n_fwd >= 50 and syn >= n_fwd * 0.9 and fps > 200
