# -*- coding: utf-8 -*-
"""
SQLite 告警持久化
=================
所有检测事件自动写入 alerts 表;支持历史查询与 SHAP 结果缓存。
线程安全:每个操作使用独立连接(捕获/模拟线程均会写入)。
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# 数据库路径可用环境变量覆盖(供 Docker 持久卷使用),默认在 detector/ 下
DB_PATH = Path(os.environ.get("DB_PATH", str(Path(__file__).resolve().parent / "alerts.db")))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表(如不存在)与索引。"""
    conn = _conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id          TEXT PRIMARY KEY,
                timestamp   REAL NOT NULL,
                attack_type TEXT NOT NULL,
                risk_level  TEXT,
                src_ip      TEXT,
                dst_ip      TEXT,
                dst_port    INTEGER,
                confidence  REAL,
                status      TEXT DEFAULT 'NEW',
                shap_result TEXT,
                source      TEXT
            )
            """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(alerts)").fetchall()}
        if "shap_result" not in cols:
            conn.execute("ALTER TABLE alerts ADD COLUMN shap_result TEXT")
        if "source" not in cols:
            conn.execute("ALTER TABLE alerts ADD COLUMN source TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts(risk_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(attack_type)")
        conn.commit()
    finally:
        conn.close()


def insert_alert(ev: dict) -> None:
    """写入一条告警事件(幂等:按 id 覆盖)。"""
    conn = _conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO alerts "
            "(id, timestamp, attack_type, risk_level, src_ip, dst_ip, dst_port,"
            " confidence, status, shap_result, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ev["id"], ev.get("ts"), ev.get("attack"), ev.get("risk_level"),
             ev.get("src_ip"), ev.get("dst_ip"), ev.get("dst_port"),
             ev.get("confidence"), ev.get("status", "NEW"), None, ev.get("src")),
        )
        conn.commit()
    finally:
        conn.close()


def update_status(event_id: str, status: str) -> None:
    conn = _conn()
    try:
        conn.execute("UPDATE alerts SET status=? WHERE id=?", (status, event_id))
        conn.commit()
    finally:
        conn.close()


def update_shap(event_id: str, shap_json: str) -> None:
    """缓存 SHAP 解释结果,历史告警重开详情无需重算。"""
    conn = _conn()
    try:
        conn.execute("UPDATE alerts SET shap_result=? WHERE id=?", (shap_json, event_id))
        conn.commit()
    finally:
        conn.close()


def delete_alerts_by_source(source: str) -> int:
    """删除指定来源的告警,返回删除条数。"""
    conn = _conn()
    try:
        cur = conn.execute("DELETE FROM alerts WHERE source=?", (source,))
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def query_alerts(attack_type: str | None = None, risk_level: str | None = None,
                 start: float | None = None, end: float | None = None,
                 limit: int = 200) -> list[dict]:
    """按条件查询历史告警,按时间倒序。"""
    sql = "SELECT * FROM alerts WHERE 1=1"
    args: list = []
    if attack_type:
        sql += " AND attack_type=?"
        args.append(attack_type)
    if risk_level:
        sql += " AND risk_level=?"
        args.append(risk_level)
    if start is not None:
        sql += " AND timestamp>=?"
        args.append(float(start))
    if end is not None:
        sql += " AND timestamp<=?"
        args.append(float(end))
    sql += " ORDER BY timestamp DESC LIMIT ?"
    args.append(int(limit))
    conn = _conn()
    try:
        rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
