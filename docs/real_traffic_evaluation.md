# 真实流量观测与评估

本流程区分无标签观测和有标签评估。观测可以检查包是否被解析、流是否正常结算、模型是否报错及告警分布；准确率、误报率、召回率等需要独立复核的标签。模型预测为 BENIGN 不等于已确认正常，模型产生攻击告警也不等于已确认遭到攻击。

首轮真实观测结果见 [真实流量实测记录](../reports/real_traffic_observation.md)。

## 已补充的功能

- 独立命令行入口 `detector/evaluate_traffic.py`，支持环境检查、限时本机采集、pcap/pcapng 分析、人工标签评分。
- 每次运行建立新目录，保留会话 ID、输入文件与模型哈希、特征版本、提取器与规则代码哈希、配置和依赖版本；目录已存在时拒绝覆盖。
- 记录所有流，包括正常预测、规则命中和推理失败，避免只观察告警样本。
- 生成不预填标签、不显示预测结论的人工复核 CSV。评分核对会话、流 ID、标签类别、证据说明和复核人。
- 模型单独、规则单独、二者 OR 组合的二分类对比；模型的多分类混淆矩阵、各类 Precision/Recall/F1 和标签覆盖率。
- API 抓包路径与评估脚本共用 `DetectionEngine`，每次采集或回放重新创建规则状态，防止上轮历史累计进入下一轮。

## 运行环境

当前电脑使用 `D:\miniconda3\envs\ids\python.exe`。现有环境已具备依赖，无需为评估另建环境或重新安装依赖。安装与下载操作仍由用户本人执行。

在项目根目录的 PowerShell 中运行：

```powershell
Set-Location E:\project1
& 'D:\miniconda3\envs\ids\python.exe' detector\evaluate_traffic.py doctor
```

`doctor` 输出 Npcap 是否可用、默认网卡和可选网卡。Windows 下 Npcap 安装完成后，需要重新启动旧后端进程和 Python 进程，已有进程未必能识别新驱动。遇到访问权限错误时，以管理员身份打开 PowerShell 后重试。

## 1. 限时观测当前电脑

```powershell
& 'D:\miniconda3\envs\ids\python.exe' detector\evaluate_traffic.py observe --seconds 60 --max-packets 20000 --source-note 'Local workstation; ordinary usage; labels not yet reviewed'
```

默认监听 Scapy 路由选择的网卡，过滤条件为 `ip or ip6`，关闭混杂模式，最长等待 60 秒或收到 2 万包即停止。单次时长最多允许 600 秒。它只被动读取网卡可见流量，不主动扫描、不发送攻击包，也不下载外部资源。

指定网卡时，在命令末尾增加 `--interface` 和 `doctor` 给出的 `network_name`。在观测期间按平时方式使用电脑即可；需要不同业务覆盖时，分别建立浏览、会议、文件传输等会话并写入 `--source-note`。不要因为是自己使用的电脑，就把全部样本自动标为正常。

默认结果位置为 `artifacts/traffic/时间-随机后缀/`。也可加 `--output artifacts\traffic\自定新目录`；每轮必须使用新目录。

| 文件 | 用途 |
| --- | --- |
| `capture.pcap` | 本轮原始数据包，可复现分析；可能包含通信内容，仅保留本地 |
| `environment.json` | 抓包环境和网卡信息 |
| `capture.json` | 采集时间、包数、过滤器、是否达到包数上限 |
| `flows.jsonl` | 所有流的端点、时间、69 维特征、模型概率、规则命中和失败记录 |
| `review.csv` | 人工标签模板；`true_label`、`label_basis`、`reviewer` 留空 |
| `summary.json` | 统计信息、版本哈希、无标签评估边界 |
| `report.md` | 自动生成的观测报告 |
| `failure.json` | 运行失败时记录原因，不能把失败或空采集当成零误报 |

原始抓包、IP、标签等位于 `.gitignore` 已排除的 `artifacts/traffic/` 中。`--output` 指向其他位置时应自行确认该目录不会被提交。可提交的实测总结只保留汇总统计，不复制端点或通信内容。

采集阶段只落盘，结束后离线提取并推理，避免模型计算阻塞抓包回调。报告里的推理毫秒数只测模型调用，不能当作网页实时告警的端到端延迟；驱动丢包数没有测量，不能报告“零丢包”。

## 2. 重放真实抓包

例如重放首轮已保存的本机观测文件：

```powershell
& 'D:\miniconda3\envs\ids\python.exe' detector\evaluate_traffic.py pcap artifacts\traffic\live-observation-20260905\capture.pcap --provenance real --source-note 'Replay of local passive observation'
```

输入支持 Scapy 可读取的 pcap/pcapng。脚本逐包读取，默认最多处理 2 万包；超过上限会在报告标注截断。处理时间戳逆序或非有限值会报错，不会静默重排并声称是同一实验。

`--provenance` 支持 `real`、`synthetic`、`unknown`，默认 `unknown`。外部文件来源由提供者负责说明；工具只能检测项目自带的已知合成样本，无法自动认证任意文件是真实流量。无论文件是否有标签，来源声明都不能省略。

仅测试软件流程时使用：

```powershell
& 'D:\miniconda3\envs\ids\python.exe' detector\evaluate_traffic.py pcap detector\samples\test_flows.pcap --provenance synthetic
```

这条命令的输出不得纳入真实网络效果结论。

## 3. 独立人工复核

填写对应会话的 `review.csv`，每行代表本次特征提取划分出的一条流。五元组加起止时间帮助与原始抓包对应；以会话 ID 与流 ID 作为评分键，不能混用另一轮的标签表。

| 字段 | 填写要求 |
| --- | --- |
| `true_label` | `BENIGN`、`DDoS`、`PortScan`、`Botnet`、`BruteForce`、`WebAttack`、`DoS`、`Infiltration` 之一；不能确认时留空 |
| `label_basis` | 独立证据，例如相同时间段的应用日志、已核实业务连接、授权实验记录；不能只写模型结论 |
| `reviewer` | 复核人或可追溯的复核标识 |

既要复核告警，也要复核预测正常的流。只选择容易判断或只选择告警的样本，会使评价偏向被选择的子集。最好完整复核短会话，或者预先固定抽样方法，并在报告中说明覆盖率和不确定标签。

独立复核需要结合可获得的证据；本工具不会因“没有主动攻击”或“浏览器看起来正常”就自动生成 BENIGN 标签，也不会把 SHAP 解释当成真实标签。

## 4. 计算标签指标

完成首轮标签复核后：

```powershell
& 'D:\miniconda3\envs\ids\python.exe' detector\evaluate_traffic.py score --run artifacts\traffic\live-observation-20260905 --labels artifacts\traffic\live-observation-20260905\review.csv
```

评分输出在另一个新目录中，包含 `metrics.json` 与 `report.md`，不覆盖原始观测。全部标签留空会明确拒绝评分；不允许重复 ID、跨会话标签、未知类别或缺少证据与复核人的已填写标签。已复核流若预测失败，也会拒绝评分，防止把失败样本偷偷从指标中删除。

指标解释：

- 二分类按 `BENIGN` 与所有攻击类别区分，分别计算模型、规则、OR 组合的 TP/FP/TN/FN、Precision、Recall、F1、FPR 和 Accuracy。
- `FPR = FP / (FP + TN)` 的分母是已复核正常流，不是所有观测流；没有正常标签时 FPR 为 `null`。
- 没有攻击标签时不能评价攻击召回率。无定义分母的二分类指标记为 `null`，报告显示 `N/A`，不伪造 0 或 100%。
- 多分类混淆矩阵的行是真实类别，列是模型预测类别。`macro_f1_supported_classes` 只平均本次有真实样本的类别，不能直接与离线完整 8 类 Macro-F1 混为一谈。
- 规则对比以“触发告警的流”为单位；规则还没累计到阈值的早期攻击流会记作未触发。这不是攻击活动级召回率，不能把一次规则命中回填到前面所有流来抬高分数。
- 组合仅表示“模型异常或规则命中”，同一条流只计一次二分类告警，不是学习型融合模型。网页仍可能分别展示模型和规则事件，所以页面事件条数不一定等于评估中的告警流数。

## 特征与时间修正

特征版本为 `scapy-flow-v2-microseconds`：

1. Flow Duration、各方向 IAT、active/idle 时长均输出微秒；包速率和字节速率仍以秒作分母。
2. 起始时间为 0 的流能正常计算持续时间。
3. 新包加入前先结算已空闲超时的流，避免间隔很长、五元组相同的两次连接被错误合并。
4. 过期流按结束时间排序，跨流规则使用 `end_ts`，不再用回放时机器的处理时钟。
5. 服务回放与评估默认使用同样的 15 秒空闲超时，采集/回放之间重置规则状态。

以上修正没有宣称与 CICFlowMeter 完全一致。包长定义、头长、子流、active/idle 的分段阈值等仍有近似；当前也没有按 TCP FIN/RST 立即结束流、最大活跃时长结算或严格的 IP 分片/IPv6 扩展头重组。中途开始采集可能把响应方向当成正向，结束强制结算的流可能不完整。

规则在流结算后才执行，时间使用该流最后一个包的时间。这比处理时钟可复现，但仍不是逐包滑动窗口；流内很早发生的 SYN 会集中归入流结束时间，应把它列入规则评估的局限。

## 回归检查

```powershell
& 'D:\miniconda3\envs\ids\python.exe' -m unittest discover -s tests -v
```

测试中的包只在内存和临时 pcap 文件中构造，不发送到网络。覆盖双向流、微秒单位、流超时分割、规则阈值与时间窗、服务与评估一致性、无标签拒绝评分、错误标签拒绝、推理失败计数等。

## 后续实验

本次真实观测仅覆盖一台电脑的一小段业务，不能证明不同网络和攻击类型的效果。下一阶段应按固定业务场景开展多轮采集、独立标注、报告每轮和汇总指标；在已有授权隔离环境中的标注流量上再测攻击召回，并使用同一份 pcap 与参考特征提取器逐列对齐。不要用本次观测调整模型后，仍把同一份数据作为独立测试集。
