# -*- coding: utf-8 -*-
"""
CICIDS2017 实时入侵检测服务 (FastAPI)
=====================================
架构:流量 -> (Scapy 抓包 / pcap 回放 / 模拟) -> CICIDS 特征 -> XGBoost -> 攻击类型+置信度 -> Web 面板

端点:
  GET  /                  Vue + ECharts 监控面板
  POST /predict           模型推理(输入特征字典 -> 攻击类型 + 置信度)
  POST /explain           SHAP 特征归因(解释模型判定依据,需 pip install shap)
  GET  /events            SSE 实时事件流
  GET  /events/history    最近事件(含风险等级/证据/建议/源目标IP)
  GET  /models            模型元信息
  GET  /health            健康检查
  POST /simulate/start|stop    模拟流量事件源(真实数据抽样,无需抓包)
  POST /capture/start|stop      Scapy 实时抓包(需 Npcap + 管理员)
  POST /capture/replay         离线 pcap 回放(纯 Python,无需 Npcap)

运行:
  cd detector
  uvicorn api_server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))

import db
import inference as inf   # 用模块引用,避免 from-import 绑定旧列表导致 classes=0
from inference import load_model, predict
from explain import explain
import evidence
from flow_extractor import PortScanDetector, SynFloodDetector, replay_pcap
from simulate import run_simulator

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---------------- 事件存储(线程安全,内存环形缓冲) ----------------
MAX_EVENTS = 500
_events: list[dict] = []


def _make_event(attack: str, confidence: float, src: str,
                features: dict | None = None, meta: dict | None = None,
                extras: dict | None = None) -> dict:
    features = features or {}
    meta = meta or {}
    ev = {
        "id": uuid.uuid4().hex[:8],
        "time": time.strftime("%H:%M:%S", time.localtime()),
        "ts": time.time(),
        "attack": attack,
        "confidence": round(float(confidence), 4),
        "status": "NEW",   # 告警生命周期: NEW -> CONFIRMED/IGNORED -> CLOSED
        "risk_level": evidence.risk_level(attack),
        "evidence": evidence.evidence(features, attack),
        "recommendation": evidence.recommendations(attack),
        "src": src,
        "src_ip": meta.get("src_ip", ""),
        "dst_ip": meta.get("dst_ip", ""),
        "dst_port": meta.get("dst_port", 0),
        "proto": meta.get("proto", ""),
    }
    if extras:
        ev.update(extras)
    return ev


def _emit_event(ev: dict) -> None:
    _events.append(ev)
    if len(_events) > MAX_EVENTS:
        del _events[: len(_events) - MAX_EVENTS]


_feature_store: dict[str, dict] = {}   # event_id -> 特征(供详情页 SHAP 解释)


def _handle_detection(attack: str, confidence: float, src: str,
                      features=None, meta=None, extras=None) -> dict:
    ev = _make_event(attack, confidence, src, features, meta, extras)
    _emit_event(ev)
    _feature_store[ev["id"]] = features or {}
    if len(_feature_store) > 200:  # 只保留最近 200 条事件的特征
        for k in list(_feature_store)[:-200]:
            _feature_store.pop(k, None)
    db.insert_alert(ev)   # 持久化到 SQLite
    return ev


def _db_row_to_event(r: dict) -> dict:
    """SQLite 行 -> 内存事件结构(与实时事件字段对齐)。"""
    ts = float(r.get("timestamp") or 0)
    attack = r.get("attack_type") or "Unknown"
    return {
        "id": r.get("id"),
        "ts": ts,
        "time": time.strftime("%H:%M:%S", time.localtime(ts)),
        "attack": attack,
        "risk_level": r.get("risk_level") or evidence.risk_level(attack),
        "confidence": r.get("confidence") or 0.0,
        "status": r.get("status") or "NEW",
        "evidence": evidence.evidence({}, attack),        # 历史事件按攻击类型推导
        "recommendation": evidence.recommendations(attack),
        "src": r.get("source") or "",
        "src_ip": r.get("src_ip") or "",
        "dst_ip": r.get("dst_ip") or "",
        "dst_port": r.get("dst_port") or 0,
        "proto": "",
        "shap_result": r.get("shap_result"),              # 缓存的 SHAP 解释(JSON)
    }


# ---------------- 模拟流量 ----------------
_sim_thread: threading.Thread | None = None
_sim_stop: threading.Event | None = None


def _simulate_emit(attack, confidence, src, features, meta) -> None:
    # 模拟器返回 numpy 向量,转成 {特征名: 值} dict,便于证据规则与 SHAP 复用
    if isinstance(features, (list, tuple)) or hasattr(features, "tolist"):
        features = {col: float(v) for col, v in zip(inf._feature_cols, features)}
    _handle_detection(attack, confidence, src, features=features, meta=meta)


# ---------------- 实时抓包 / pcap 回放 ----------------
_capture_thread: threading.Thread | None = None
_capture_stop: threading.Event | None = None
_scan = PortScanDetector(max_ports=15, window=30.0)
_syn_flood = SynFloodDetector(max_syn=100, window=10.0)


def _capture_emit(features: dict, meta: dict) -> None:
    """每个到期流:模型判定 + 行为启发式(端口扫描 / SYN 洪泛)。
    产品语义:正常流量不产生告警,只上报异常事件。"""
    try:
        r = predict(features)
    except Exception:  # noqa: BLE001
        return

    if r["attack"] != "BENIGN":
        _handle_detection(r["attack"], r["confidence"], "capture",
                          features=features, meta=meta)

    hit = _scan.feed(meta)
    if hit:
        _handle_detection(
            "PortScan", 0.92, "capture",
            features={},
            meta={"src_ip": hit["src_ip"], "dst_ip": hit.get("dst_ip", ""),
                  "dst_port": 0, "proto": "TCP"},
            extras={
                "confidence": 0.92,
                "evidence": [f"来源 {hit['src_ip']} 在时间窗内探测了 {hit['ports']} 个目标端口",
                             "行为模式符合端口扫描(模型单流特征无法覆盖)"],
            })

    flood = _syn_flood.feed(features, meta)
    if flood:
        _handle_detection(
            "DoS", 0.93, "capture",
            features=features, meta=meta,
            extras={
                "confidence": 0.93,
                "evidence": [f"来源 {flood['src_ip']} 对 {flood['dst_ip']}:{flood['dst_port']} "
                             f"聚合 {flood['syn']} 个 SYN 且无回包",
                             "行为模式符合 SYN 洪泛(轮换源端口,需跨流聚合)"],
            })


# ---------------- FastAPI 应用 ----------------
app = FastAPI(
    title="Explainable AI-based Real-time Network Intrusion Detection System",
    description="基于机器学习与可解释AI的实时网络入侵检测与安全分析平台",
    version="1.3.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

load_model()  # 启动即加载模型


@app.on_event("startup")
async def _startup() -> None:
    db.init_db()
    # 从数据库加载历史告警到内存环形缓冲,页面加载时可直接展示
    for row in db.query_alerts(limit=MAX_EVENTS):
        _events.append(_db_row_to_event(row))
    print(f"[启动] 模型已加载: {len(inf._class_names)} 类 / {len(inf._feature_cols)} 特征")
    print(f"[启动] 已加载 {len(_events)} 条历史告警 (SQLite: {db.DB_PATH})")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "classes": len(inf._class_names),
            "features": len(inf._feature_cols), "events": len(_events)}


@app.get("/models")
async def models_info() -> dict:
    return {"class_names": inf._class_names, "feature_cols": inf._feature_cols}


@app.post("/predict")
async def predict_endpoint(request: Request) -> JSONResponse:
    """接收任意 CICIDS 特征字典,返回攻击类型 + 置信度,并把结果推入实时事件流。
    注:裸 dict 参数会被 FastAPI 当作查询参数,故用 Request 手动解析 JSON body。"""
    try:
        payload = await request.json()
        result = predict(payload)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})
    _handle_detection(result["attack"], result["confidence"], "api", features=payload)
    return JSONResponse(result)


@app.post("/explain")
async def explain_endpoint(request: Request) -> JSONResponse:
    """SHAP 特征归因:解释模型为何判定为某攻击类型。"""
    try:
        payload = await request.json()
        result = explain(payload)
    except ImportError:
        return JSONResponse(status_code=400,
                            content={"error": "shap 未安装,请执行: pip install shap"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return JSONResponse(result)


@app.get("/explain/{event_id}")
async def explain_by_id(event_id: str) -> JSONResponse:
    """按事件 ID 获取 SHAP 解释(使用该事件产生时的流特征)。"""
    features = _feature_store.get(event_id)
    if features is None:
        return JSONResponse(status_code=404,
                            content={"error": "该事件的特征已过期(仅保留最近 200 条)"})
    try:
        result = explain(features)
    except ImportError:
        return JSONResponse(status_code=400,
                            content={"error": "shap 未安装,请执行: pip install shap"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})
    result["features"] = features  # 附带该事件的原始特征,供详情页"完整特征"区块展示
    db.update_shap(event_id, json.dumps(result, ensure_ascii=False))  # 缓存解释,历史可回看
    return JSONResponse(result)


@app.get("/events")
async def events_stream(request: Request) -> StreamingResponse:
    """SSE 实时事件流。面板通过 EventSource 订阅,0.5s 轮询增量推送。"""

    async def gen():
        last = 0
        while True:
            if await request.is_disconnected():
                break
            if len(_events) > last:
                for ev in _events[last:]:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                last = len(_events)
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/events/history")
async def events_history() -> dict:
    return {"events": _events[-100:]}


@app.get("/alerts")
async def query_alerts(attack_type: str | None = None, risk_level: str | None = None,
                       start: float | None = None, end: float | None = None,
                       limit: int = 300) -> dict:
    """从 SQLite 查询历史告警(支持按攻击类型 / 风险等级 / 时间范围过滤)。"""
    rows = db.query_alerts(attack_type=attack_type, risk_level=risk_level,
                           start=start, end=end, limit=limit)
    return {"events": [_db_row_to_event(r) for r in rows]}


VALID_STATUS = {"NEW", "CONFIRMED", "IGNORED", "CLOSED"}


@app.post("/events/{event_id}/status")
async def set_event_status(event_id: str, payload: dict | None = None) -> JSONResponse:
    """告警状态流转:NEW -> CONFIRMED / IGNORED -> CLOSED。"""
    status = ((payload or {}).get("status") or "").upper()
    if status not in VALID_STATUS:
        return JSONResponse(status_code=400,
                            content={"error": f"status 必须为 {sorted(VALID_STATUS)}"})
    for ev in _events:
        if ev["id"] == event_id:
            ev["status"] = status
            db.update_status(event_id, status)   # 同步持久化
            return JSONResponse({"id": event_id, "status": status})
    return JSONResponse(status_code=404, content={"error": "事件不存在或已过期"})


@app.get("/events/export")
async def export_events() -> StreamingResponse:
    """导出全部告警为 CSV(带 UTF-8 BOM,Excel 打开中文不乱码)。"""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "attack", "risk_level", "src_ip", "dst_ip",
                "dst_port", "confidence", "status"])
    for ev in _events:
        w.writerow([ev["time"], ev["attack"], ev["risk_level"], ev["src_ip"],
                    ev["dst_ip"], ev["dst_port"], ev["confidence"],
                    ev.get("status", "NEW")])
    content = "﻿" + buf.getvalue()
    return StreamingResponse(
        iter([content]), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="alerts.csv"'})


@app.post("/simulate/start")
async def simulate_start(payload: dict | None = None) -> JSONResponse:
    global _sim_thread, _sim_stop
    if _sim_thread and _sim_thread.is_alive():
        return JSONResponse({"status": "already_running"})
    rate = float((payload or {}).get("rate", 1.0))
    _sim_stop = threading.Event()
    _sim_thread = threading.Thread(
        target=run_simulator, args=(rate, _simulate_emit, _sim_stop), daemon=True)
    _sim_thread.start()
    return JSONResponse({"status": "started", "rate": rate})


@app.post("/simulate/stop")
async def simulate_stop() -> JSONResponse:
    global _sim_thread, _sim_stop
    if _sim_thread and _sim_thread.is_alive() and _sim_stop:
        _sim_stop.set()
        _sim_thread.join(timeout=5)
    return JSONResponse({"status": "stopped"})


@app.post("/capture/start")
async def capture_start(payload: dict | None = None) -> JSONResponse:
    """Scapy 实时抓包。需要 scapy + Npcap + 管理员权限。"""
    global _capture_thread, _capture_stop
    if _capture_thread and _capture_thread.is_alive():
        return JSONResponse({"status": "already_running"})
    from flow_extractor import _SCAPY_OK, sniff_loop
    if not _SCAPY_OK:
        return JSONResponse(status_code=400, content={
            "error": "scapy 未安装或不可用,请执行 pip install scapy 并安装 Npcap"})
    iface = (payload or {}).get("iface")
    _capture_stop = threading.Event()
    _capture_thread = threading.Thread(
        target=sniff_loop, kwargs={"interface": iface, "on_flow": _capture_emit,
                                   "stop_event": _capture_stop}, daemon=True)
    _capture_thread.start()
    return JSONResponse({"status": "started", "iface": iface or "默认网卡"})


@app.post("/capture/stop")
async def capture_stop() -> JSONResponse:
    global _capture_thread, _capture_stop
    if _capture_thread and _capture_thread.is_alive() and _capture_stop:
        _capture_stop.set()
        _capture_thread.join(timeout=5)
    return JSONResponse({"status": "stopped"})


@app.post("/capture/replay")
async def capture_replay(payload: dict | None = None) -> JSONResponse:
    """离线 pcap 回放:把 pcap 文件逐包解析成流并检测(纯 Python,无需 Npcap)。
    传 {"sample": true} 使用项目自带样本 test_flows.pcap。"""
    global _capture_thread
    payload = payload or {}
    pcap = payload.get("pcap", "")
    if not pcap and payload.get("sample"):
        pcap = str(Path(__file__).resolve().parent / "samples" / "test_flows.pcap")
    if not pcap:
        return JSONResponse(status_code=400, content={"error": "需要 pcap 文件路径或 sample=true"})
    if not Path(pcap).exists():
        return JSONResponse(status_code=400, content={"error": f"文件不存在: {pcap}"})
    if _capture_thread and _capture_thread.is_alive():
        return JSONResponse({"status": "busy"})
    _capture_thread = threading.Thread(
        target=replay_pcap, args=(pcap,), kwargs={"on_flow": _capture_emit}, daemon=True)
    _capture_thread.start()
    return JSONResponse({"status": "started", "pcap": pcap})
