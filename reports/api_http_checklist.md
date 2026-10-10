# HTTP 检查记录（2026-09-11 11:53:44）

结果：20/20 项通过。服务真实运行，数据库为临时文件。

| 范围 | 请求 | 预期 | 实际 | 结果 |
|---|---|---|---|---|
| 服务状态 | GET /health | 200 + status/classes/features | 200 {'status': 'ok', 'classes': 8, 'features': 69, 'events': 0} | 通过 |
| 服务状态 | GET /models | 200 + 69 特征 / 类别列表 | 200 features=69 classes=8 | 通过 |
| 推理接口 | POST /predict 有效69维向量 | 200 + attack/confidence/probabilities | 200 attack=BENIGN conf=0.9996 | 通过 |
| 推理接口 | POST /predict 错误维数(3维列表) | 4xx + 明确错误信息 | 400 error={'error': '特征数量应为 69,实际收到 3。若传 dict,请使用 CICIDS 特征名。'} | 通过 |
| 推理接口 | POST /predict 无效JSON | 4xx + 明确错误信息 | 400 error={'error': 'Expecting value: line 1 column 1 (char 0)'} | 通过 |
| 特征缺失 | POST /predict 缺字段字典 {Flow Duration:1234} | 记录实际行为(当前实现补0而非拒绝) | 200 attack=BENIGN conf=0.9997 | 通过 |
| 特征解释 | POST /explain 有效向量 | 200 + attack/confidence/top_features | 200 top=5 | 通过 |
| 特征解释 | GET /explain/{event_id} 有效事件 28d3f5f9 | 200 + top_features | 200 keys=['attack', 'confidence', 'features', 'top_features'] | 通过 |
| 特征解释 | GET /explain/{event_id} 不存在事件 | 404 + 明确错误 | 404 error=该事件的特征已过期(仅保留最近 200 条) | 通过 |
| 告警查询 | GET /alerts 无条件 | 200 + 事件列表 | 200 count=3 | 通过 |
| 告警查询 | GET /alerts?attack_type=BENIGN | 200 + 返回项 attack_type 全部匹配 | 200 count=3 all_match=True | 通过 |
| 告警查询 | GET /alerts?risk_level=LOW | 200 + 返回项 risk_level 全部匹配 | 200 count=3 all_match=True | 通过 |
| 状态更新 | POST /events/28d3f5f9/status CONFIRMED | 200 + 内存状态同步 | 200 {'id': '28d3f5f9', 'status': 'CONFIRMED'} mem_synced=True | 通过 |
| 状态更新 | POST /events/{id}/status 非法状态 | 400 + 状态名与提示 | 400 error=status 必须为 ['CLOSED', 'CONFIRMED', 'IGNORED', 'NEW'] | 通过 |
| 状态更新 | POST /events/{id}/status 不存在ID | 404 + 明确错误 | 404 error=事件不存在或已过期 | 通过 |
| SSE与前端 | 订阅 GET /events 后触发 /predict | 订阅端收到该新事件的同一 id | predict=200 new_ids=['52dd593b'] sse_events=4 matched=True | 通过 |
| 历史与导出 | GET /events/export | 200 CSV + 表头字段 | 200 header=time,attack,risk_level,src_ip,dst_ip,dst_port,confidence,status | 通过 |
| 各环节结果一致 | SQLite 中事件 28d3f5f9 状态 | 与内存更新一致(CONFIRMED) | db_status=CONFIRMED | 通过 |
| 各环节结果一致 | SQLite 告警行数 | >0 且与 /alerts 可读 | rows=4 | 通过 |
| 历史与导出 | 重启服务后 GET /alerts | 200 + 历史告警可恢复 | 200 count=4 | 通过 |
