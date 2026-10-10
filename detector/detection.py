"""Shared model and rule decisions for live detection and offline evaluation."""
from __future__ import annotations

import time

import inference as inf
from flow_extractor import PortScanDetector, SynFloodDetector


class DetectionEngine:
    def __init__(self, predictor=None):
        self.predictor = predictor or inf.predict
        self.scan = PortScanDetector(max_ports=15, window=30.0)
        self.syn_flood = SynFloodDetector(max_syn=100, window=10.0)

    def detect(self, features: dict, meta: dict) -> dict:
        started = time.perf_counter()
        prediction = self.predictor(features)
        model_ms = (time.perf_counter() - started) * 1000
        # Use packet time, not replay speed or the wall clock of the evaluator.
        ts = meta["end_ts"]
        rules = []
        scan = self.scan.feed(meta, ts=ts)
        if scan:
            rules.append({"rule": "port_scan", "attack": "PortScan", **scan})
        flood = self.syn_flood.feed(features, meta, ts=ts)
        if flood:
            rules.append({"rule": "syn_flood", "attack": "DoS", **flood})
        model_alert = prediction["attack"] != "BENIGN"
        return {
            "model": prediction,
            "rules": rules,
            "model_alert": model_alert,
            "rule_alert": bool(rules),
            "combined_alert": model_alert or bool(rules),
            "model_ms": model_ms,
        }
