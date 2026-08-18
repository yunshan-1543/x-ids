# CICIDS2017 特征分析报告

- 数据规模: 2,520,798 行 × 70 列
- 类别数: 15 种(BENIGN + 14 种攻击)

## 类别分布

| 类别 | 样本数 | 占比 |
|---|---|---|
| BENIGN | 2,095,057 | 83.11% |
| DoS Hulk | 172,846 | 6.86% |
| DDoS | 128,014 | 5.08% |
| PortScan | 90,694 | 3.60% |
| DoS GoldenEye | 10,286 | 0.41% |
| FTP-Patator | 5,931 | 0.24% |
| DoS slowloris | 5,385 | 0.21% |
| DoS Slowhttptest | 5,228 | 0.21% |
| SSH-Patator | 3,219 | 0.13% |
| Bot | 1,948 | 0.08% |
| Web Attack � Brute Force | 1,470 | 0.06% |
| Web Attack � XSS | 652 | 0.03% |
| Infiltration | 36 | 0.00% |
| Web Attack � Sql Injection | 21 | 0.00% |
| Heartbleed | 11 | 0.00% |

## 高度相关特征对(|r|>=0.95,共 38 对)

- total_fwd_packets ↔ total_backward_packets (r=1.0)
- total_fwd_packets ↔ total_length_of_bwd_packets (r=1.0)
- total_fwd_packets ↔ subflow_fwd_packets (r=1.0)
- total_fwd_packets ↔ subflow_bwd_packets (r=1.0)
- total_fwd_packets ↔ subflow_bwd_bytes (r=1.0)
- total_backward_packets ↔ total_length_of_bwd_packets (r=1.0)
- total_backward_packets ↔ subflow_fwd_packets (r=1.0)
- total_backward_packets ↔ subflow_bwd_packets (r=1.0)
- total_backward_packets ↔ subflow_bwd_bytes (r=1.0)
- total_length_of_fwd_packets ↔ subflow_fwd_bytes (r=1.0)
- total_length_of_bwd_packets ↔ subflow_fwd_packets (r=1.0)
- total_length_of_bwd_packets ↔ subflow_bwd_packets (r=1.0)
- total_length_of_bwd_packets ↔ subflow_bwd_bytes (r=1.0)
- fwd_packet_length_mean ↔ avg_fwd_segment_size (r=1.0)
- bwd_packet_length_mean ↔ avg_bwd_segment_size (r=1.0)
- fwd_psh_flags ↔ syn_flag_count (r=1.0)
- fwd_urg_flags ↔ cwe_flag_count (r=1.0)
- rst_flag_count ↔ ece_flag_count (r=1.0)
- subflow_fwd_packets ↔ subflow_bwd_packets (r=1.0)
- subflow_fwd_packets ↔ subflow_bwd_bytes (r=1.0)

## Top 20 特征重要性

- packet_length_variance: 0.0935
- packet_length_std: 0.0816
- bwd_packet_length_std: 0.0788
- avg_bwd_segment_size: 0.0649
- bwd_packet_length_max: 0.0509
- packet_length_mean: 0.0401
- bwd_packet_length_mean: 0.0398
- max_packet_length: 0.0369
- average_packet_size: 0.0341
- subflow_bwd_bytes: 0.0313
- destination_port: 0.0257
- total_length_of_bwd_packets: 0.0234
- subflow_fwd_bytes: 0.0215
- total_length_of_fwd_packets: 0.0200
- subflow_fwd_packets: 0.0195
- total_fwd_packets: 0.0191
- min_seg_size_forward: 0.0150
- fwd_packet_length_max: 0.0141
- act_data_pkt_fwd: 0.0141
- fwd_iat_std: 0.0131
