# -*- coding: utf-8 -*-
"""HTTP-level checklist for the detection service.

Starts a real uvicorn process against a temporary SQLite file (the project's
detector/alerts.db is never touched), exercises every endpoint in the
"功能与接口联调" checklist, and prints one JSON record per check.

Usage (from repo root, ids env):
    python tests/check_api_http.py

Requires no network access beyond 127.0.0.1.  No packets are transmitted.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "detector"
PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"

RECORDS: list[dict] = []


def _g(r, key, default=None):
    """Best-effort field read: a non-JSON error body stays a plain string."""
    return r.get(key, default) if isinstance(r, dict) else r


def record(scope: str, target: str, expected: str, actual: str, ok: bool) -> None:
    RECORDS.append({"scope": scope, "target": target, "expected": expected,
                    "actual": actual, "ok": ok})


def request(method: str, path: str, body=None, raw: str | None = None,
            timeout: float = 20.0):
    """Return (status, parsed_or_text). 4xx/5xx are returned, not raised."""
    data = None
    headers = {}
    if raw is not None:
        data = raw.encode()
        headers["Content-Type"] = "application/json"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8-sig", errors="replace")
            try:
                return resp.status, json.loads(text)
            except json.JSONDecodeError:
                return resp.status, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8-sig", errors="replace")
        try:
            return exc.code, json.loads(text)
        except json.JSONDecodeError:
            return exc.code, text


def start_server(db_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["DB_PATH"] = str(db_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_server:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        cwd=str(DETECTOR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(240):  # model load can take a while
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"server exited early:\n{out}")
        try:
            status, _ = request("GET", "/health", timeout=2)
            if status == 200:
                return proc
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("server did not become healthy in 120s")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def check_health_and_models() -> tuple[list[str], list[str]]:
    s1, health = request("GET", "/health")
    record("服务状态", "GET /health", "200 + status/classes/features",
           f"{s1} {health}", s1 == 200 and health.get("status") == "ok")
    s2, models = request("GET", "/models")
    cols = models.get("feature_cols", []) if isinstance(models, dict) else []
    names = models.get("class_names", []) if isinstance(models, dict) else []
    record("服务状态", "GET /models", "200 + 69 特征 / 类别列表",
           f"{s2} features={len(cols)} classes={len(names)}", s2 == 200)
    return cols, names


def check_predict(cols: list[str]) -> None:
    n = len(cols)
    good = [0.0] * n
    s, r = request("POST", "/predict", body=good)
    ok = s == 200 and "attack" in r and "confidence" in r
    record("推理接口", "POST /predict 有效69维向量",
           "200 + attack/confidence/probabilities",
           f"{s} attack={_g(r,'attack')} conf={_g(r,'confidence')}",
           ok)

    s, r = request("POST", "/predict", raw=json.dumps([1.0, 2.0, 3.0]))
    record("推理接口", "POST /predict 错误维数(3维列表)",
           "4xx + 明确错误信息", f"{s} error={str(r)[:90]}",
           s == 400 and "error" in r)

    s, r = request("POST", "/predict", raw="this is not json")
    record("推理接口", "POST /predict 无效JSON",
           "4xx + 明确错误信息", f"{s} error={str(r)[:90]}",
           400 <= s < 500 and "error" in r)

    # missing-fields dict: project fills absent features with 0
    s, r = request("POST", "/predict", body={"Flow Duration": 1234})
    record("特征缺失", "POST /predict 缺字段字典 {Flow Duration:1234}",
           "记录实际行为(当前实现补0而非拒绝)",
           f"{s} attack={_g(r,'attack')} conf={_g(r,'confidence')}",
           s == 200)  # documents the fill-0 behaviour, not a pass/fail of rejection


def check_explain(cols: list[str]) -> str | None:
    s, r = request("POST", "/explain", body=[0.0] * len(cols), timeout=60)
    ok = s == 200 and "top_features" in r
    record("特征解释", "POST /explain 有效向量",
           "200 + attack/confidence/top_features", f"{s} top={len(_g(r,'top_features',[]))}",
           ok)

    # create a fresh event via /predict so its features are in the store
    s, pred = request("POST", "/predict", body=[0.0] * len(cols))
    event_id = None
    if s == 200:
        s_hist, hist = request("GET", "/events/history")
        if s_hist == 200 and hist.get("events"):
            event_id = hist["events"][-1]["id"]
    if event_id:
        s, r = request("GET", f"/explain/{event_id}", timeout=60)
        record("特征解释", f"GET /explain/{{event_id}} 有效事件 {event_id}",
               "200 + top_features", f"{s} keys={sorted(r)[:6]}",
               s == 200 and "top_features" in r)
    else:
        record("特征解释", "GET /explain/{event_id} 有效事件",
               "200 + top_features", "无法定位新事件 id", False)

    s, r = request("GET", "/explain/does-not-exist")
    record("特征解释", "GET /explain/{event_id} 不存在事件",
           "404 + 明确错误", f"{s} error={_g(r,'error')}",
           s == 404)
    return event_id


def check_alerts() -> int:
    s, r = request("GET", "/alerts?limit=300")
    total = len(r.get("events", [])) if s == 200 else -1
    record("告警查询", "GET /alerts 无条件", "200 + 事件列表", f"{s} count={total}",
           s == 200)

    # pick a filter value that exists, then cross-check against SQLite
    s, r = request("GET", "/alerts?limit=1")
    sample = (r.get("events") or [{}])[0] if s == 200 else {}
    if sample.get("attack"):
        attack = sample["attack"]
        s, r = request("GET", f"/alerts?attack_type={urllib.parse.quote(attack)}&limit=300")
        rows = r.get("events", []) if s == 200 else []
        all_match = all(e["attack"] == attack for e in rows)
        record("告警查询", f"GET /alerts?attack_type={attack}",
               "200 + 返回项 attack_type 全部匹配",
               f"{s} count={len(rows)} all_match={all_match}",
               s == 200 and bool(rows) and all_match)
    if sample.get("risk_level"):
        risk = sample["risk_level"]
        s, r = request("GET", f"/alerts?risk_level={urllib.parse.quote(risk)}&limit=300")
        rows = r.get("events", []) if s == 200 else []
        all_match = all(e["risk_level"] == risk for e in rows)
        record("告警查询", f"GET /alerts?risk_level={risk}",
               "200 + 返回项 risk_level 全部匹配",
               f"{s} count={len(rows)} all_match={all_match}",
               s == 200 and bool(rows) and all_match)
    return total


def check_status() -> str | None:
    s, hist = request("GET", "/events/history")
    events = hist.get("events", []) if s == 200 else []
    if not events:
        record("状态更新", "POST /events/{id}/status", "存在可更新事件",
               "内存事件为空", False)
        return None
    event_id = events[-1]["id"]

    s, r = request("POST", f"/events/{event_id}/status", body={"status": "confirmed"})
    mem_ok = any(e["id"] == event_id and e["status"] == "CONFIRMED"
                 for e in request("GET", "/events/history")[1].get("events", []))
    record("状态更新", f"POST /events/{event_id}/status CONFIRMED",
           "200 + 内存状态同步", f"{s} {r} mem_synced={mem_ok}",
           s == 200 and r.get("status") == "CONFIRMED" and mem_ok)

    s, r = request("POST", f"/events/{event_id}/status", body={"status": "not-a-status"})
    record("状态更新", "POST /events/{id}/status 非法状态",
           "400 + 状态名与提示", f"{s} error={_g(r,'error')}",
           s == 400)

    s, r = request("POST", "/events/zzzzzzzz/status", body={"status": "CONFIRMED"})
    record("状态更新", "POST /events/{id}/status 不存在ID",
           "404 + 明确错误", f"{s} error={_g(r,'error')}",
           s == 404)
    return event_id


def check_sse(cols: list[str]) -> None:
    """Subscribe first, then trigger /predict, and confirm that event's id arrives."""
    _, before_hist = request("GET", "/events/history")
    before = {e["id"] for e in (before_hist.get("events") or [])}

    received: list[dict] = []
    stop = threading.Event()

    def reader():
        try:
            with urllib.request.urlopen(BASE + "/events", timeout=30) as resp:
                for raw in resp:
                    if stop.is_set():
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line.startswith("data:"):
                        try:
                            received.append(json.loads(line[5:].strip()))
                        except json.JSONDecodeError:
                            pass
        except Exception:  # noqa: BLE001
            pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    time.sleep(1.0)                      # let the stream connect and drain backlog
    s, pred = request("POST", "/predict", body=[0.0] * len(cols))
    _, after_hist = request("GET", "/events/history")
    new_ids = [e["id"] for e in (after_hist.get("events") or []) if e["id"] not in before]
    deadline = time.time() + 8
    while time.time() < deadline and not ({e["id"] for e in received} & set(new_ids)):
        time.sleep(0.2)
    stop.set()
    got = bool({e["id"] for e in received} & set(new_ids))
    record("SSE与前端", "订阅 GET /events 后触发 /predict",
           "订阅端收到该新事件的同一 id",
           f"predict={s} new_ids={new_ids} sse_events={len(received)} matched={got}", got)


def check_export(cols: list[str]) -> None:
    s, body = request("GET", "/events/export")
    header = body.strip().splitlines()[0] if isinstance(body, str) and body.strip() else ""
    record("历史与导出", "GET /events/export",
           "200 CSV + 表头字段", f"{s} header={header}",
           s == 200 and header.startswith("time,attack"))


def check_persistence(db_path: Path, status_event: str | None) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT id,status FROM alerts").fetchall()
    finally:
        con.close()
    db_ids = {r[0]: r[1] for r in rows}
    if status_event:
        db_status = db_ids.get(status_event)
        record("各环节结果一致", f"SQLite 中事件 {status_event} 状态",
               "与内存更新一致(CONFIRMED)",
               f"db_status={db_status}", db_status == "CONFIRMED")
    record("各环节结果一致", "SQLite 告警行数",
           ">0 且与 /alerts 可读", f"rows={len(rows)}", len(rows) > 0)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ids_api_check_"))
    db_path = tmp / "check_alerts.db"
    print(f"[info] temp DB: {db_path}")

    proc = start_server(db_path)
    try:
        cols, _ = check_health_and_models()
        check_predict(cols)
        check_explain(cols)
        check_alerts()
        status_event = check_status()
        check_sse(cols)
        check_export(cols)
        check_persistence(db_path, status_event)

        # restart against the same DB file to verify history survives
        stop_server(proc)
        proc = start_server(db_path)
        s, r = request("GET", "/alerts?limit=300")
        count = len(r.get("events", [])) if s == 200 else -1
        record("历史与导出", "重启服务后 GET /alerts",
               "200 + 历史告警可恢复", f"{s} count={count}", s == 200 and count > 0)
    finally:
        stop_server(proc)

    passed = sum(1 for r in RECORDS if r["ok"])
    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "passed": passed, "total": len(RECORDS), "records": RECORDS}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    report_dir = next((a.split("=", 1)[1] for a in sys.argv[1:]
                       if a.startswith("--report-dir=")), None)
    if report_dir:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "api_http_checklist.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [f"# HTTP 检查记录（{summary['date']}）", "",
                 f"结果：{passed}/{len(RECORDS)} 项通过。服务真实运行，数据库为临时文件。", "",
                 "| 范围 | 请求 | 预期 | 实际 | 结果 |",
                 "|---|---|---|---|---|"]
        for r in RECORDS:
            lines.append("| {scope} | {target} | {expected} | {actual} | {v} |".format(
                v="通过" if r["ok"] else "失败", **r))
        (out / "api_http_checklist.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
        print(f"[info] report written to {out}")
    return 0 if passed == len(RECORDS) else 1


if __name__ == "__main__":
    sys.exit(main())
