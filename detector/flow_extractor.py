# -*- coding: utf-8 -*-
"""
Scapy 实时流特征提取器 (可选模块)
=================================
把网卡上抓到的原始包按五元组聚合成"流",增量计算 CICIDS2017 的 69 个模型特征,
流超时后交给模型分类。对应架构图中的 "Scapy" 一层。

依赖:scapy + Npcap(Windows 实时抓包)。未安装时本模块 import 会给出提示,
服务其余功能(API / 模拟 / 面板)不受影响。

注意(近似实现):
  - fwd/bwd_header_length:IP 头(L3)+ L4 头(TCP/UDP/ICMP)之和
  - subflow_*:短流近似等于总包数/总字节(CICFlowMeter 需 bulk 检测,此处简化)
  - active/idle:以连续包间隔 <= 2s 划为同一活跃突发,再统计突发时长/空闲间隔
  - min_seg_size_forward:近似取正向包长最小值
  生产级特征对齐建议改用 CICFlowMeter / Zeek。
"""
from __future__ import annotations

import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

try:
    from scapy.all import IP, IPv6, TCP, UDP, ICMP, Raw
    _SCAPY_OK = True
except Exception:  # noqa: BLE001
    _SCAPY_OK = False

import inference as inf   # 模块引用,避免 from-import 在 load_model 前捕获空列表

IDLE_TIMEOUT = 120.0     # 流空闲超时(秒),超时即判定
ACTIVE_GAP = 2.0         # 活跃突发判定阈值(秒)

# TCP 标志位位掩码(Scapy FlagValue 属性是大写字母,统一用位掩码统计)
_FLAG_BITS = {"FIN": 0x01, "SYN": 0x02, "RST": 0x04, "PSH": 0x08,
              "ACK": 0x10, "URG": 0x20, "ECE": 0x40, "CWE": 0x80}

# IP 协议号 -> 可读名称
_PROTO_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP", 47: "GRE", 50: "ESP"}


def _flag_value(tcp) -> int:
    try:
        return int(tcp.flags)
    except Exception:  # noqa: BLE001
        return int(tcp.flags.value)


def _std(vals) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals))


def _header_len(pkt) -> int:
    hl = 0
    if IP in pkt:
        hl += int(pkt[IP].ihl) * 4
    elif IPv6 in pkt:
        hl += 40
    if TCP in pkt:
        hl += int(pkt[TCP].dataofs) * 4
    elif UDP in pkt:
        hl += 8
    elif ICMP in pkt:
        hl += 8
    return hl


class Flow:
    """单条流的增量统计。"""

    def __init__(self, key, src_ip, dst_ip, sport, dport, proto):
        self.key = key
        self.src_ip, self.dst_ip = src_ip, dst_ip
        self.sport, self.dport, self.proto = sport, dport, proto
        self.fwd_len, self.bwd_len = [], []
        self.fwd_ts, self.bwd_ts = [], []
        self.fwd_hdr = self.bwd_hdr = 0
        self.fwd_psh = self.bwd_psh = 0
        self.fwd_urg = self.bwd_urg = 0
        self.flags = {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0,
                      "ACK": 0, "URG": 0, "CWE": 0, "ECE": 0}
        self.init_win_fwd = 0      # SYN 窗口(fwd)
        self.init_win_bwd = 0      # SYN-ACK 窗口(bwd)
        self.act_data_fwd = 0      # 含数据负载的正向包数
        self.min_seg_fwd = None    # 正向包长最小值
        self.all_ts = []
        self.start = None
        self.last = None
        self.fwd_total = 0
        self.bwd_total = 0

    def add(self, pkt, ts, is_fwd):
        if self.start is None:
            self.start = ts
        self.last = ts
        self.all_ts.append(ts)
        size = len(pkt)
        hl = _header_len(pkt)

        tcp = pkt[TCP] if TCP in pkt else None
        if tcp is not None:
            fv = _flag_value(tcp)
            for name, bit in _FLAG_BITS.items():
                if fv & bit:
                    self.flags[name] += 1
            if self.flags["SYN"] == 1 and self.init_win_fwd == 0 and is_fwd:
                self.init_win_fwd = int(tcp.window)
            if not is_fwd and self.init_win_bwd == 0:
                self.init_win_bwd = int(tcp.window)  # 首个反向包(通常为 SYN-ACK)的窗口

        if is_fwd:
            self.fwd_len.append(size)
            self.fwd_ts.append(ts)
            self.fwd_total += size
            self.fwd_hdr += hl
            if tcp is not None:
                if tcp.flags.P:
                    self.fwd_psh += 1
                if tcp.flags.U:
                    self.fwd_urg += 1
                if tcp.payload and Raw in pkt:
                    self.act_data_fwd += 1
                self.min_seg_fwd = size if self.min_seg_fwd is None else min(self.min_seg_fwd, size)
        else:
            self.bwd_len.append(size)
            self.bwd_ts.append(ts)
            self.bwd_total += size
            self.bwd_hdr += hl
            if tcp is not None:
                if tcp.flags.P:
                    self.bwd_psh += 1
                if tcp.flags.U:
                    self.bwd_urg += 1

    def to_meta(self) -> dict:
        """流级元数据:来源/目标 IP、端口、协议(用于告警详情与溯源)。"""
        return {
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "dst_port": self.dport,
            "proto": _PROTO_NAMES.get(self.proto, str(self.proto)),
        }

    def _iat(self, ts_list):
        return [b - a for a, b in zip(ts_list, ts_list[1:])]

    def _active_idle(self):
        ts = sorted(self.all_ts)
        if len(ts) < 2:
            return [0.0], [0.0]
        active, idle = [], []
        burst_start = ts[0]
        prev = ts[0]
        for t in ts[1:]:
            if t - prev <= ACTIVE_GAP:
                prev = t
            else:
                active.append(prev - burst_start)
                idle.append(t - prev)
                burst_start = t
                prev = t
        active.append(prev - burst_start)
        return (active if active else [0.0]), (idle if idle else [0.0])

    def to_features(self) -> dict:
        """按模型特征顺序输出 69 维特征(值域对齐 CICIDS:时长微秒、比值等)。"""
        n_fwd, n_bwd = len(self.fwd_len), len(self.bwd_len)
        n_pkt = n_fwd + n_bwd
        dur = (self.last - self.start) if (self.start and self.last) else 0.0
        dur_us = dur * 1e6
        total = self.fwd_total + self.bwd_total
        all_len = self.fwd_len + self.bwd_len

        flow_iat = self._iat(self.all_ts)
        fwd_iat = self._iat(self.fwd_ts)
        bwd_iat = self._iat(self.bwd_ts)
        act, idl = self._active_idle()

        def stat(seq, op="mean"):
            if not seq:
                return 0.0
            return {"mean": np.mean, "std": _std, "max": max, "min": min, "sum": sum}[op](seq)

        f = {
            "destination_port": self.dport,
            "flow_duration": dur_us,
            "total_fwd_packets": n_fwd,
            "total_backward_packets": n_bwd,
            "total_length_of_fwd_packets": self.fwd_total,
            "total_length_of_bwd_packets": self.bwd_total,
            "fwd_packet_length_max": max(self.fwd_len) if n_fwd else 0,
            "fwd_packet_length_min": min(self.fwd_len) if n_fwd else 0,
            "fwd_packet_length_mean": stat(self.fwd_len, "mean"),
            "fwd_packet_length_std": _std(self.fwd_len),
            "bwd_packet_length_max": max(self.bwd_len) if n_bwd else 0,
            "bwd_packet_length_min": min(self.bwd_len) if n_bwd else 0,
            "bwd_packet_length_mean": stat(self.bwd_len, "mean"),
            "bwd_packet_length_std": _std(self.bwd_len),
            "flow_bytes/s": (total / dur) if dur > 0 else 0.0,
            "flow_packets/s": (n_pkt / dur) if dur > 0 else 0.0,
            "flow_iat_mean": stat(flow_iat, "mean"),
            "flow_iat_std": _std(flow_iat),
            "flow_iat_max": max(flow_iat) if flow_iat else 0.0,
            "flow_iat_min": min(flow_iat) if flow_iat else 0.0,
            "fwd_iat_total": stat(fwd_iat, "sum"),
            "fwd_iat_mean": stat(fwd_iat, "mean"),
            "fwd_iat_std": _std(fwd_iat),
            "fwd_iat_max": max(fwd_iat) if fwd_iat else 0.0,
            "fwd_iat_min": min(fwd_iat) if fwd_iat else 0.0,
            "bwd_iat_total": stat(bwd_iat, "sum"),
            "bwd_iat_mean": stat(bwd_iat, "mean"),
            "bwd_iat_std": _std(bwd_iat),
            "bwd_iat_max": max(bwd_iat) if bwd_iat else 0.0,
            "bwd_iat_min": min(bwd_iat) if bwd_iat else 0.0,
            "fwd_psh_flags": self.fwd_psh,
            "fwd_urg_flags": self.fwd_urg,
            "fwd_header_length": self.fwd_hdr,
            "bwd_header_length": self.bwd_hdr,
            "fwd_packets/s": (n_fwd / dur) if dur > 0 else 0.0,
            "bwd_packets/s": (n_bwd / dur) if dur > 0 else 0.0,
            "min_packet_length": min(all_len) if all_len else 0,
            "max_packet_length": max(all_len) if all_len else 0,
            "packet_length_mean": stat(all_len, "mean"),
            "packet_length_std": _std(all_len),
            "packet_length_variance": float(np.var(all_len)) if len(all_len) > 1 else 0.0,
            "fin_flag_count": self.flags["FIN"],
            "syn_flag_count": self.flags["SYN"],
            "rst_flag_count": self.flags["RST"],
            "psh_flag_count": self.flags["PSH"],
            "ack_flag_count": self.flags["ACK"],
            "urg_flag_count": self.flags["URG"],
            "cwe_flag_count": self.flags["CWE"],
            "ece_flag_count": self.flags["ECE"],
            "down/up_ratio": (self.bwd_total / self.fwd_total) if self.fwd_total else 0.0,
            "average_packet_size": (total / n_pkt) if n_pkt else 0.0,
            "avg_fwd_segment_size": (self.fwd_total / n_fwd) if n_fwd else 0.0,
            "avg_bwd_segment_size": (self.bwd_total / n_bwd) if n_bwd else 0.0,
            "subflow_fwd_packets": n_fwd,
            "subflow_fwd_bytes": self.fwd_total,
            "subflow_bwd_packets": n_bwd,
            "subflow_bwd_bytes": self.bwd_total,
            "init_win_bytes_forward": self.init_win_fwd,
            "init_win_bytes_backward": self.init_win_bwd,
            "act_data_pkt_fwd": self.act_data_fwd,
            "min_seg_size_forward": self.min_seg_fwd or 0,
            "active_mean": stat(act, "mean"),
            "active_std": _std(act),
            "active_max": max(act) if act else 0.0,
            "active_min": min(act) if act else 0.0,
            "idle_mean": stat(idl, "mean"),
            "idle_std": _std(idl),
            "idle_max": max(idl) if idl else 0.0,
            "idle_min": min(idl) if idl else 0.0,
        }
        return {c: f.get(c, 0.0) for c in inf._feature_cols}


class FlowExtractor:
    """按五元组聚合包的流提取器。"""

    def __init__(self, idle_timeout: float = IDLE_TIMEOUT):
        self.flows: dict = {}
        self.idle_timeout = idle_timeout

    def _tuple(self, pkt):
        if IP in pkt:
            src, dst = pkt[IP].src, pkt[IP].dst
            proto = pkt[IP].proto
        elif IPv6 in pkt:
            src, dst = pkt[IPv6].src, pkt[IPv6].dst
            proto = pkt[IPv6].nh
        else:
            return None
        if TCP in pkt:
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        else:
            sport, dport = 0, 0
        return src, dst, sport, dport, proto

    def process(self, pkt, ts=None) -> None:
        if not _SCAPY_OK:
            return
        t = self._tuple(pkt)
        if t is None:
            return
        src, dst, sport, dport, proto = t
        fwd_key = (src, dst, sport, dport, proto)
        rev_key = (dst, src, dport, sport, proto)
        if fwd_key in self.flows:
            flow, is_fwd = self.flows[fwd_key], True
        elif rev_key in self.flows:
            flow, is_fwd = self.flows[rev_key], False
        else:
            flow = Flow(fwd_key, src, dst, sport, dport, proto)
            self.flows[fwd_key] = flow
            is_fwd = True
        flow.add(pkt, ts if ts is not None else time.time(), is_fwd)

    def expire(self, now=None) -> list[tuple[dict, dict]]:
        """返回已超时流的 (特征字典, 元数据),并从字典中移除。"""
        now = now if now is not None else time.time()
        out = []
        for key in list(self.flows):
            flow = self.flows[key]
            if now - flow.last > self.idle_timeout:
                out.append((flow.to_features(), flow.to_meta()))
                del self.flows[key]
        return out

    def size(self) -> int:
        return len(self.flows)


def sniff_loop(interface=None, on_flow=None, on_packet=None, count=None,
               stop_event=None, idle_timeout: float = IDLE_TIMEOUT,
               bpf_filter: str | None = None):
    """阻塞式实时抓包:每来一个包更新流状态,超时流回调 on_flow(features, meta)。"""
    if not _SCAPY_OK:
        print("[scapy] 未安装: pip install scapy,并安装 Npcap(WinPcap)后才能实时抓包")
        return
    from scapy.all import sniff
    extractor = FlowExtractor(idle_timeout=idle_timeout)
    seen = 0

    def _pkt(pkt):
        nonlocal seen
        seen += 1
        extractor.process(pkt)
        if on_packet:
            on_packet(pkt)
        for feat, meta in extractor.expire():
            if on_flow:
                on_flow(feat, meta)

    def _stop(pkt):
        return bool(stop_event is not None and stop_event.is_set())

    print(f"[抓包] 开始监听 {interface or '默认网卡'}(Ctrl+C 停止)")
    while not (stop_event is not None and stop_event.is_set()):
        if count is not None and seen >= count:
            break
        remaining = 0 if count is None else max(count - seen, 0)
        sniff(iface=interface, prn=_pkt, store=False, count=remaining,
              stop_filter=_stop, filter=bpf_filter, timeout=1)
        for feat, meta in extractor.expire():
            if on_flow:
                on_flow(feat, meta)

    # 停止时结算剩余活跃流,避免短演示里没有任何流到期。
    for feat, meta in extractor.expire(now=float("inf")):
        if on_flow:
            on_flow(feat, meta)


def replay_pcap(path: str, on_flow=None, progress=None):
    """读取 pcap 文件,逐包喂入 FlowExtractor,回调 on_flow(features, meta)。

    纯 Python 解析,不需要 Npcap,用于离线流量验证与演示。
    """
    if not _SCAPY_OK:
        print("[scapy] 未安装: pip install scapy 后才能解析 pcap")
        return
    from scapy.all import rdpcap

    extractor = FlowExtractor(idle_timeout=1.0)  # 回放用短超时,尽快结算流
    pkts = rdpcap(path)
    total = len(pkts)
    for i, pkt in enumerate(pkts):
        ts = float(getattr(pkt, "time", 0) or 0)
        extractor.process(pkt, ts=ts)
        for feat, meta in extractor.expire(now=ts):
            if on_flow:
                on_flow(feat, meta)
        if progress and i % 200 == 0:
            progress(i + 1, total)
    for feat, meta in extractor.expire(now=float("inf")):  # 剩余流全部结算
        if on_flow:
            on_flow(feat, meta)
    print(f"[回放] 共处理 {total} 个包")


class PortScanDetector:
    """行为启发式:同一源 IP 在时间窗口内访问的不同目标端口数超阈值 -> 疑似端口扫描。

    单包扫描流无法靠 69 维特征被模型识别,此检测器用跨流行为补足(与模型互补)。
    """

    def __init__(self, max_ports: int = 15, window: float = 30.0):
        self.max_ports = max_ports
        self.window = window
        self._seen: dict[str, dict[int, float]] = {}
        self._dst: dict[str, str] = {}  # src_ip -> 最近目标 IP

    def feed(self, meta: dict, ts=None) -> dict | None:
        """喂一条流的元数据;命中端口扫描返回 {'src_ip':..., 'ports':N, 'dst_ip':...},否则 None。"""
        proto = str(meta.get("proto", ""))
        if proto not in ("TCP", "6"):
            return None
        src = meta.get("src_ip")
        port = int(meta.get("dst_port", 0) or 0)
        dst_ip = meta.get("dst_ip", "")
        if not src or not port:
            return None
        now = ts if ts is not None else time.time()
        d = self._seen.setdefault(src, {})
        d[port] = now
        if dst_ip:
            self._dst[src] = dst_ip
        stale = [p for p, t in d.items() if now - t > self.window]
        for p in stale:
            del d[p]
        if len(d) > self.max_ports:
            self._seen[src] = {}
            return {"src_ip": src, "ports": len(d), "dst_ip": self._dst.get(src, "")}
        return None


class SynFloodDetector:
    """行为启发式:同一源 IP 在时间窗内对同一目标端口发送大量 SYN 且无回包 -> SYN 洪泛。

    SYN 洪泛通常使用轮换源端口,每个包是独立五元组流,单流特征无法识别;
    此检测器按 (src, dst, dport) 聚合 SYN 计数(与模型互补)。
    """

    def __init__(self, max_syn: int = 100, window: float = 10.0):
        self.max_syn = max_syn
        self.window = window
        self._syn: dict[tuple, tuple[float, int]] = {}
        self._last_fire: dict[tuple, float] = {}   # 冷却:避免同一洪泛反复告警

    def feed(self, features: dict, meta: dict, ts=None) -> dict | None:
        """喂一条到期流的特征与元数据;命中返回 {'src_ip','dst_ip','dst_port','syn'},否则 None。"""
        syn = int(features.get("syn_flag_count", 0) or 0)
        n_bwd = int(features.get("total_backward_packets", 0) or 0)
        if syn == 0 or n_bwd > 0:
            return None
        key = (meta.get("src_ip"), meta.get("dst_ip"), meta.get("dst_port"))
        if not key[0] or not key[1]:
            return None
        now = ts if ts is not None else time.time()
        if key in self._last_fire and now - self._last_fire[key] < self.window:
            return None                               # 冷却期内不重复告警
        if key not in self._syn:                      # 首见:初始化窗口起点
            self._syn[key] = (now, 0)
        last, cnt = self._syn[key]
        if now - last > self.window:                  # 窗口过期:重置计数
            cnt = 0
            self._syn[key] = (now, 0)
        cnt += syn
        self._syn[key] = (now, cnt)
        if cnt > self.max_syn:
            self._syn[key] = (now, 0)
            self._last_fire[key] = now
            return {"src_ip": key[0], "dst_ip": key[1], "dst_port": key[2], "syn": cnt}
        return None
