"""Offline regression tests; packets are constructed in memory, never transmitted."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))

from scapy.all import ARP, Ether, IP, TCP, wrpcap

import inference as inf
from detection import DetectionEngine
from evaluate_traffic import analyze_pcap, binary_metrics, dump_json, score_run
from flow_extractor import FlowExtractor, replay_pcap


def packet(ts, reverse=False, port=443):
    endpoints = ("192.0.2.1", "198.51.100.1", 50000, port)
    if reverse:
        endpoints = (endpoints[1], endpoints[0], endpoints[3], endpoints[2])
    src, dst, sport, dport = endpoints
    result = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="A")
    result.time = ts
    return result


def benign(_features):
    return {"attack": "BENIGN", "confidence": 0.8,
            "probabilities": {"BENIGN": 0.8, "DoS": 0.2}}


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inf.load_model()

    def test_bidirectional_counts_microseconds_and_epoch_zero(self):
        extractor = FlowExtractor()
        for pkt in (packet(0), packet(0.25, reverse=True), packet(0.5), packet(3.5)):
            extractor.process(pkt, float(pkt.time))
        [(features, meta)] = extractor.expire(float("inf"))
        self.assertEqual(features["total_fwd_packets"], 3)
        self.assertEqual(features["total_backward_packets"], 1)
        self.assertEqual(features["flow_duration"], 3500000)
        self.assertAlmostEqual(features["flow_iat_mean"], 3500000 / 3)
        self.assertEqual(features["fwd_iat_total"], 3500000)
        self.assertEqual(features["active_max"], 500000)
        self.assertEqual(features["idle_max"], 3000000)
        self.assertAlmostEqual(features["flow_packets/s"], 4 / 3.5)
        self.assertEqual(meta["src_port"], 50000)
        self.assertEqual(meta["start_ts"], 0)
        self.assertEqual(list(features), inf._feature_cols)

    def test_expiration_is_chronological_not_insertion_order(self):
        extractor = FlowExtractor(15)
        extractor.process(packet(0, port=80), 0)
        extractor.process(packet(1, port=443), 1)
        extractor.process(packet(2, port=80), 2)
        rows = extractor.expire(float("inf"))
        self.assertEqual([meta["end_ts"] for _, meta in rows], [1, 2])

    def test_reused_five_tuple_after_idle_gap_is_not_merged(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.pcap"
            wrpcap(str(path), [packet(0), packet(20)])
            rows = []
            replay_pcap(str(path), on_flow=lambda f, m: rows.append((f, m)))
            self.assertEqual(len(rows), 2)
            self.assertEqual([f["total_fwd_packets"] for f, _ in rows], [1, 1])

    def test_invalid_timeout(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FlowExtractor(timeout)

    def test_replay_rejects_reversed_packet_time(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unordered.pcap"
            wrpcap(str(path), [packet(20), packet(0)])
            with self.assertRaisesRegex(ValueError, "timestamps"):
                replay_pcap(str(path))


class RuleTests(unittest.TestCase):
    def decision(self, engine, ts, port=80, syn=0, bwd=0):
        return engine.detect({"syn_flag_count": syn, "total_backward_packets": bwd},
                             {"src_ip": "192.0.2.1", "dst_ip": "198.51.100.1",
                              "dst_port": port, "proto": "TCP", "end_ts": ts})

    def test_scan_threshold_and_event_clock(self):
        engine = DetectionEngine(benign)
        for i in range(15):
            self.assertFalse(self.decision(engine, i, port=1000 + i)["rule_alert"])
        decision = self.decision(engine, 15, port=1015)
        self.assertEqual(decision["rules"][0]["rule"], "port_scan")
        self.assertFalse(decision["model_alert"])
        self.assertTrue(decision["combined_alert"])
        engine = DetectionEngine(benign)
        # CPU-speed replay of 16 widely spaced flows must not look like a scan.
        for i in range(16):
            self.assertFalse(self.decision(engine, i * 31, port=1000 + i)["rule_alert"])

    def test_syn_threshold_cooldown_and_new_session(self):
        engine = DetectionEngine(benign)
        self.assertFalse(self.decision(engine, 0, syn=100)["rule_alert"])
        self.assertTrue(self.decision(engine, 1, syn=1)["rule_alert"])
        self.assertFalse(self.decision(engine, 2, syn=101)["rule_alert"])
        self.assertTrue(self.decision(engine, 12, syn=101)["rule_alert"])
        self.assertFalse(self.decision(DetectionEngine(benign), 13, syn=1)["rule_alert"])

    def test_syn_with_response_is_not_rule_hit(self):
        self.assertFalse(self.decision(DetectionEngine(benign), 1, syn=101, bwd=1)["rule_alert"])

    def test_api_uses_shared_engine_and_retains_rule_evidence(self):
        import api_server
        engine = DetectionEngine(benign)
        with patch.object(api_server, "_capture_engine", engine), \
             patch.object(api_server, "_capture_stats", {"flows": 0, "alerts": 0, "last_error": ""}), \
             patch.object(api_server, "_handle_detection") as emit:
            for i in range(16):
                api_server._capture_emit({"syn_flag_count": 1, "total_backward_packets": 0},
                                         {"src_ip": "192.0.2.1", "dst_ip": "198.51.100.1",
                                          "dst_port": 1000 + i, "proto": "TCP", "end_ts": i})
            self.assertEqual(api_server._capture_stats["flows"], 16)
            self.assertEqual(api_server._capture_stats["alerts"], 1)
            self.assertEqual(emit.call_args.args[0], "PortScan")
            self.assertEqual(emit.call_args.kwargs["extras"]["detector"], "port_scan")


class EvaluationTests(unittest.TestCase):
    def test_pcap_run_is_unlabeled_and_matches_service_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "input.pcap"
            wrpcap(str(path), [packet(0), packet(0.1, reverse=True),
                              Ether() / ARP(), packet(20)])
            # Give the non-IP packet a timestamp within the stream.
            from scapy.all import rdpcap
            pkts = rdpcap(str(path))
            pkts[2].time = 0.2
            wrpcap(str(path), pkts)
            output = root / "run"
            output.mkdir()
            summary = analyze_pcap(path, output, "synthetic")
            self.assertIsNone(summary["labeled_metrics"])
            self.assertEqual(summary["flows_total"], 2)
            self.assertEqual(summary["non_ip_packets"], 1)
            records = [json.loads(s) for s in (output / "flows.jsonl").read_text().splitlines()]
            service_rows = []
            engine = DetectionEngine()
            replay_pcap(str(path), on_flow=lambda f, m: service_rows.append((f, engine.detect(f, m))))
            for record, (features, decision) in zip(records, service_rows):
                self.assertEqual(record["features"], features)
                self.assertEqual(record["model"], decision["model"])
                self.assertEqual(record["rules"], decision["rules"])
            with (output / "review.csv").open(encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    self.assertEqual(row["true_label"], "")
                    self.assertNotIn("model", row)
            with self.assertRaisesRegex(ValueError, "No independently"):
                score_run(output, output / "review.csv")

    def test_non_ip_capture_is_empty_not_perfect_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "arp.pcap"
            wrpcap(str(path), [Ether() / ARP()])
            result = analyze_pcap(path, root, "synthetic")
            self.assertEqual(result["status"], "empty")
            self.assertIsNone(result["combined_alert_fraction"])

    def test_known_synthetic_fixture_cannot_be_claimed_real(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "synthetic"):
                analyze_pcap(ROOT / "detector/samples/test_flows.pcap", Path(temp), "real")

    def test_failed_predictions_are_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "input.pcap"
            wrpcap(str(path), [packet(0)])
            with patch("detection.inf.predict", side_effect=ValueError("test failure")):
                summary = analyze_pcap(path, root, "synthetic")
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["flow_errors"], 1)
            self.assertIsNone(summary["combined_alert_fraction"])

    def test_packet_limit_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "input.pcap"
            wrpcap(str(path), [packet(0), packet(1)])
            summary = analyze_pcap(path, root, "synthetic", max_packets=1)
            self.assertEqual(summary["packets_read"], 1)
            self.assertTrue(summary["packet_limit_truncated"])

    def test_binary_metrics_do_not_invent_missing_denominators(self):
        values = binary_metrics([False, False], [False, True])
        self.assertEqual(values["false_positive_rate"], 0.5)
        self.assertIsNone(values["recall"])
        self.assertIsNone(binary_metrics([], [])["accuracy"])

    def test_labels_and_ablation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {"flow_id": "a", "status": "ok", "model": {"attack": "BENIGN"},
                 "model_alert": False, "rule_alert": False, "combined_alert": False},
                {"flow_id": "b", "status": "ok", "model": {"attack": "BENIGN"},
                 "model_alert": False, "rule_alert": True, "combined_alert": True},
                {"flow_id": "c", "status": "ok", "model": {"attack": "DoS"},
                 "model_alert": True, "rule_alert": False, "combined_alert": True},
            ]
            dump_json(root / "summary.json", {"run_id": "test", "provenance": "synthetic",
                      "model": {"classes": ["BENIGN", "DoS"]}, "flow_errors": 0})
            (root / "flows.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            labels = [{"run_id": "test", "flow_id": flow_id, "true_label": label,
                       "label_basis": "unit test fixture, not real evidence", "reviewer": "test"}
                      for flow_id, label in (("a", "BENIGN"), ("b", "DoS"), ("c", "BENIGN"))]

            def write_labels():
                with (root / "labels.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(labels[0]))
                    writer.writeheader()
                    writer.writerows(labels)

            write_labels()
            result = score_run(root, root / "labels.csv")
            self.assertEqual(result["binary_ablation"]["model"]["fn"], 1)
            self.assertEqual(result["binary_ablation"]["combined"]["tp"], 1)
            self.assertEqual(result["binary_ablation"]["combined"]["false_positive_rate"], 0.5)
            self.assertEqual(result["model_multiclass"]["confusion_matrix_true_rows_predicted_columns"],
                             [[1, 1], [1, 0]])
            labels[0]["run_id"] = "other-session"
            write_labels()
            with self.assertRaisesRegex(ValueError, "does not belong"):
                score_run(root, root / "labels.csv")
            labels[0]["run_id"] = "test"
            labels[0]["true_label"] = "MADE_UP_CLASS"
            write_labels()
            with self.assertRaisesRegex(ValueError, "Unknown true_label"):
                score_run(root, root / "labels.csv")
            labels[0]["true_label"] = "BENIGN"
            labels[0]["label_basis"] = ""
            write_labels()
            with self.assertRaisesRegex(ValueError, "Evidence and reviewer"):
                score_run(root, root / "labels.csv")
            labels[0]["label_basis"] = "test"
            labels.append(labels[0])
            write_labels()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                score_run(root, root / "labels.csv")


if __name__ == "__main__":
    unittest.main()
