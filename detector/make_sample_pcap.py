# -*- coding: utf-8 -*-
"""
生成演示用样本 pcap
===================
用 Scapy 合成三类流,写入 detector/samples/test_flows.pcap:
  1) BENIGN    正常 TCP 会话(少量包,长间隔)
  2) DDoS      同一五元组的 SYN 洪泛(高频、大量 SYN)
  3) PortScan  同一源对多个目标端口的 SYN 探测(触发行为启发式)

运行: python make_sample_pcap.py
产出: detector/samples/test_flows.pcap(可用 /capture/replay 回放)
"""
from __future__ import annotations

from pathlib import Path

from scapy.all import IP, TCP, wrpcap

OUT = Path(__file__).resolve().parent / "samples" / "test_flows.pcap"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    pkts = []
    t = 1_700_000_000.0  # 起始时间戳

    # 1) 正常 HTTPS 会话:握手 + 少量数据包,间隔约 50ms
    base = ("192.168.1.8", "10.0.0.5", 54321, 443)
    seq = [(0, "S", 500), (0.02, "SA", 500), (0.04, "A", 100),
           (0.06, "PA", 800), (0.1, "A", 100), (0.2, "PA", 1200),
           (0.3, "FA", 200), (0.34, "FA", 200), (0.36, "A", 100)]
    for dt, fl, win in seq:
        p = IP(src=base[0], dst=base[1]) / TCP(sport=base[2], dport=base[3], flags=fl, window=win)
        p.time = t + dt
        pkts.append(p)
    t += 1.0

    # 2) DDoS:Src1 对 10.0.0.5:80 的 SYN 洪泛,0.3 秒 300 个包
    for i in range(300):
        p = IP(src="192.168.1.105", dst="10.0.0.5") / TCP(sport=40000 + i, dport=80, flags="S", window=64240)
        p.time = t + i * 0.001  # 1ms 一个包
        pkts.append(p)
    t += 0.5

    # 3) PortScan:Src2 对 10.0.0.20 的 20 个端口做 SYN 探测
    ports = [22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 993, 995, 1433, 3306, 3389, 5432, 6379, 8080, 9200]
    for j, dport in enumerate(ports):
        p = IP(src="192.168.1.15", dst="10.0.0.20") / TCP(sport=5555, dport=dport, flags="S", window=1024)
        p.time = t + j * 0.01
        pkts.append(p)

    wrpcap(str(OUT), pkts)
    print(f"[生成] {len(pkts)} 个包 -> {OUT}")


if __name__ == "__main__":
    main()
