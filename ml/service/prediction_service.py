"""
prediction_service.py
=====================
Python MQTT microservice that:
  1. Subscribes to the IoTEdgeSense telemetry topic (devices/data)
  2. Maintains a per-node feature window
  3. Runs the trained ONNX model via onnxruntime
  4. Publishes MaintenanceAlert JSON to devices/maintenance/{node_tag}

This service runs alongside the C++ firmware — on the same machine
or on a separate application server — and complements the C++ edge
inference engine with a richer inference environment (full Python
scientific stack, model reloading, multi-node aggregation).

Usage:
    python ml/service/prediction_service.py
    python ml/service/prediction_service.py \
        --broker mqtt.example.com --port 8883 --tls \
        --model ml/data/checkpoints/transformer_thermal.onnx \
        --meta  ml/data/checkpoints/transformer_thermal.json

Environment variables (override CLI args):
    MQTT_BROKER, MQTT_PORT, MQTT_USER, MQTT_PASS, MQTT_USE_TLS
    MODEL_PATH, META_PATH, SCALER_PATH
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("paho-mqtt not installed: pip install paho-mqtt")
    sys.exit(1)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: onnxruntime not installed — stub predictions will be used")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PredictionService")


# ---------------------------------------------------------------------------
# Node state — per-node feature accumulator
# ---------------------------------------------------------------------------

class NodeState:
    """Maintains rolling feature history for a single node."""

    BUFFER_SIZE = 1440 * 4   # 4 days at 1-minute resolution

    def __init__(self, node_tag: str, node_id: int):
        self.node_tag  = node_tag
        self.node_id   = node_id
        self.readings: deque = deque(maxlen=self.BUFFER_SIZE)
        self.last_inference_ts: float = 0.0
        self.inference_interval_s: float = 900.0   # every 15 min

    def add_reading(self, reading: dict):
        reading["_ts"] = time.time()
        self.readings.append(reading)

    def is_inference_due(self) -> bool:
        return (time.time() - self.last_inference_ts) >= self.inference_interval_s

    def mark_inferred(self):
        self.last_inference_ts = time.time()

    def has_data(self) -> bool:
        return len(self.readings) >= 5


# ---------------------------------------------------------------------------
# Feature assembler — mirrors Python TransformerFeatureExtractor
# ---------------------------------------------------------------------------

class FeatureAssembler:
    """
    Assembles a 43-element feature vector from accumulated node readings.
    Matches the feature order produced by TransformerFeatureExtractor.
    """

    WINDOWS = [30, 60, 360, 1440]   # minutes → samples (at 1-min resolution)

    def __init__(self, rated_current_a: float = 1000.0,
                       hotspot_limit_c: float  = 140.0,
                       hotspot_cont_c: float   = 98.0):
        self.rated_current_a = rated_current_a
        self.hotspot_limit_c = hotspot_limit_c
        self.hotspot_cont_c  = hotspot_cont_c

    def assemble(self, state: NodeState) -> Optional[np.ndarray]:
        """
        Build a 43-feature vector from the node's reading history.
        Returns None if insufficient data.
        """
        if not state.has_data():
            return None

        readings = list(state.readings)
        latest   = readings[-1]

        # ── Extract key quantities from latest reading ───────────────────
        # Readings carry sensor values[] in the order published by firmware
        # We expect: [voltage, current, frequency, power_factor, temperature]
        # (actual mapping depends on registered sensors)
        values = latest.get("values", [0.0])

        # Best effort extraction from telemetry payload
        current_a = float(latest.get("current_a", values[0] if values else 0.0))
        K         = float(np.clip(current_a / self.rated_current_a, 0.0, 2.0))
        K_sq      = K * K
        theta_to  = float(latest.get("top_oil_temp_c", 65.0))
        theta_h   = float(latest.get("theta_h_c",
                                      theta_to + 23.0 * (K ** 2)))  # approx
        ambient   = float(latest.get("ambient_temp_c", 20.0))
        V_rate    = float(np.exp(15000.0 * (1.0/383.0 - 1.0/(273.0 + theta_h))))
        cum_age   = float(latest.get("cumulative_ageing_h", 0.0))

        # ── Physics features (15) ────────────────────────────────────────
        features = [
            K,                                         # 0  K
            K_sq,                                      # 1  K_sq
            ambient,                                   # 2  theta_ambient_c
            theta_to,                                  # 3  theta_to_c
            theta_h,                                   # 4  theta_h_c
            theta_to - ambient,                        # 5  delta_theta_to_c
            theta_h  - theta_to,                       # 6  delta_theta_h_c
            V_rate,                                    # 7  V
            cum_age,                                   # 8  cumulative_ageing_h
            self.hotspot_cont_c - theta_h,             # 9  thermal_margin_c
            theta_h / self.hotspot_limit_c * 100.0,   # 10 hotspot_to_limit_pct
            0.0,                                       # 11 load_trend (computed below)
            V_rate,                                    # 12 ageing_acceleration
            float(theta_h > self.hotspot_cont_c),     # 13 overtemp_flag
            float(K > 1.0),                            # 14 overload_flag
        ]

        # Load trend
        if len(readings) >= 30:
            past_k = np.clip(
                float(readings[-30].get("current_a", 0.0)) / self.rated_current_a,
                0.0, 2.0
            )
            features[11] = float(np.sign(K - past_k))

        # ── Rolling window features (24: 6 stats × 4 windows) ────────────
        k_hist      = np.array([
            np.clip(r.get("current_a", 0.0) / self.rated_current_a, 0, 2)
            for r in readings
        ])
        theta_h_hist = np.array([
            r.get("theta_h_c", r.get("top_oil_temp_c", 65.0) + 23.0)
            for r in readings
        ])
        v_hist = np.exp(15000.0 * (1.0/383.0 - 1.0/(273.0 + theta_h_hist)))

        for w in self.WINDOWS:
            n     = min(w, len(readings))
            k_w   = k_hist[-n:]
            th_w  = theta_h_hist[-n:]
            v_w   = v_hist[-n:]
            features += [
                float(np.mean(k_w)),               # K_mean
                float(np.max(k_w)),                # K_max
                float(np.std(k_w)) if n > 1 else 0.0,  # K_std
                float(np.mean(th_w)),              # theta_h_mean
                float(np.max(th_w)),               # theta_h_max
                float(np.mean(v_w)),               # V_mean
            ]

        # ── Temporal features (4) ─────────────────────────────────────────
        now = datetime.now()
        features += [
            float(now.hour),
            float(now.weekday()),
            float(now.month),
            float(1 if 7 <= now.hour <= 22 else 0),
        ]

        assert len(features) == 43, f"Feature count {len(features)} != 43"
        return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Prediction service
# ---------------------------------------------------------------------------

class PredictionService:

    TELEMETRY_TOPIC   = "devices/data"
    MAINTENANCE_TOPIC = "devices/maintenance/{node_tag}"
    STATUS_TOPIC      = "devices/ml/status"

    def __init__(self, args):
        self.args      = args
        self.states    = {}               # node_tag → NodeState
        self.assembler = FeatureAssembler()
        self.scaler    = None
        self.ort_sess  = None
        self.running   = False
        self._lock     = threading.Lock()
        self.stats     = defaultdict(int)

        # MQTT client
        self.client = mqtt.Client(client_id="iotedgesense-ml-service",
                                   protocol=mqtt.MQTTv5)
        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message    = self._on_message

        if args.tls:
            self.client.tls_set()
        if args.user:
            self.client.username_pw_set(args.user, args.password)

    # ── Initialisation ────────────────────────────────────────────────────

    def start(self):
        log.info("Starting IoTEdgeSense ML Prediction Service")
        self._load_model()
        self.running = True
        self.client.connect(self.args.broker, self.args.port, keepalive=60)
        self.client.loop_start()
        log.info(f"Connected to {self.args.broker}:{self.args.port}")

    def stop(self):
        log.info("Shutting down prediction service...")
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()

    def _load_model(self):
        model_path  = self.args.model
        scaler_path = self.args.scaler
        meta_path   = self.args.meta

        # Load scaler
        if scaler_path and Path(scaler_path).exists() and JOBLIB_AVAILABLE:
            self.scaler = joblib.load(scaler_path)
            log.info(f"Feature scaler loaded: {scaler_path}")
        else:
            log.warning("No scaler found — features will NOT be normalised")

        # Load meta (feature names, config)
        if meta_path and Path(meta_path).exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            log.info(f"Model metadata loaded: {meta_path}")
        else:
            self.meta = {}

        # Load ONNX session
        if not ONNX_AVAILABLE:
            log.warning("ONNX Runtime not available — stub predictions active")
            return

        if not model_path or not Path(model_path).exists():
            log.warning(f"Model not found at {model_path} — stub mode active")
            return

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.ort_sess = ort.InferenceSession(model_path, sess_options=opts)
            log.info(f"ONNX model loaded: {model_path}")
            log.info(f"Inputs:  {[i.name for i in self.ort_sess.get_inputs()]}")
            log.info(f"Outputs: {[o.name for o in self.ort_sess.get_outputs()]}")
        except Exception as e:
            log.error(f"Failed to load ONNX model: {e}")

    # ── MQTT callbacks ────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            log.info(f"MQTT connected, subscribing to '{self.TELEMETRY_TOPIC}'")
            client.subscribe(self.TELEMETRY_TOPIC, qos=1)
            client.subscribe("devices/maintenance/+/ack", qos=0)
            self._publish_status("online")
        else:
            log.error(f"MQTT connect failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        log.warning(f"MQTT disconnected: rc={rc}")

    def _on_message(self, client, userdata, msg):
        try:
            self._process_message(msg.topic, msg.payload)
        except Exception as e:
            log.error(f"Error processing message from {msg.topic}: {e}")

    # ── Message processing ────────────────────────────────────────────────

    def _process_message(self, topic: str, payload: bytes):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return

        # Handle both single reading and batch array
        if isinstance(data, list):
            readings = data
        else:
            readings = [data]

        for reading in readings:
            node_id  = reading.get("id",   0)
            node_tag = reading.get("node", f"node_{node_id}")
            ts       = reading.get("ts",   0)

            with self._lock:
                if node_tag not in self.states:
                    self.states[node_tag] = NodeState(node_tag, node_id)
                    log.info(f"New node registered: {node_tag}")

                state = self.states[node_tag]
                state.add_reading(reading)
                self.stats["readings_received"] += 1

                # Check if inference is due
                if state.is_inference_due():
                    self._run_inference(state)

    def _run_inference(self, state: NodeState):
        features = self.assembler.assemble(state)
        if features is None:
            return

        # Apply scaler
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))[0]

        # Run ONNX or stub
        if self.ort_sess is not None:
            try:
                outputs = self.ort_sess.run(
                    None, {"features": features.reshape(1, -1)}
                )
                rul_days    = float(outputs[0][0][0])
                ageing      = float(outputs[1][0][0])
                fault_prob  = float(outputs[2][0][0])
            except Exception as e:
                log.error(f"Inference failed for {state.node_tag}: {e}")
                return
        else:
            # Stub prediction from physics features
            K       = float(features[0]) if self.scaler is None else 1.0
            theta_h = float(features[4]) if self.scaler is None else 70.0
            ageing  = min(1.0, max(0.0, (theta_h - 20) / 120.0))
            rul_days= max(0.0, 14600.0 * (1.0 - ageing))
            fault_prob = 0.7 if theta_h > 90.0 else (0.3 if theta_h > 75.0 else 0.1)

        # Determine severity
        fault_imminent = fault_prob > 0.5
        if fault_imminent or rul_days < 30:
            severity = "CRITICAL"
            action   = "IMMEDIATE INSPECTION REQUIRED. Reduce load."
        elif rul_days < 90 or fault_prob > 0.3:
            severity = "WARNING"
            action   = "Schedule inspection within 30 days."
        elif rul_days < 365:
            severity = "MONITOR"
            action   = "Increase monitoring frequency."
        else:
            severity = "NORMAL"
            action   = "No action required."

        # Build and publish alert
        alert = {
            "ts":            int(time.time() * 1000),
            "node":          state.node_tag,
            "nodeId":        state.node_id,
            "severity":      severity,
            "rul_days":      round(rul_days, 1),
            "ageing_state":  round(ageing, 4),
            "fault_prob":    round(fault_prob, 4),
            "fault_imminent": fault_imminent,
            "action":        action,
            "model":         "TransformerThermalModel_v1",
        }
        topic = self.MAINTENANCE_TOPIC.format(node_tag=state.node_tag)
        self.client.publish(topic, json.dumps(alert), qos=1)
        state.mark_inferred()
        self.stats["inferences_run"] += 1

        log.info(
            f"[{state.node_tag}] {severity}  "
            f"RUL={rul_days:.0f}d  "
            f"fault_p={fault_prob:.3f}  "
            f"ageing={ageing:.3f}"
        )

        if severity in ("WARNING", "CRITICAL"):
            log.warning(f"[{state.node_tag}] Action: {action}")

    def _publish_status(self, status: str):
        payload = json.dumps({
            "service": "iotedgesense-ml",
            "status":  status,
            "ts":      int(time.time() * 1000),
            "model_loaded": self.ort_sess is not None,
        })
        self.client.publish(self.STATUS_TOPIC, payload, qos=0, retain=True)

    def run_forever(self):
        self.start()
        try:
            while self.running:
                time.sleep(30)
                with self._lock:
                    log.info(
                        f"Status: nodes={len(self.states)}  "
                        f"readings={self.stats['readings_received']}  "
                        f"inferences={self.stats['inferences_run']}"
                    )
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IoTEdgeSense ML Prediction Service"
    )
    parser.add_argument("--broker",   default=os.getenv("MQTT_BROKER",  "localhost"))
    parser.add_argument("--port",     type=int,
                        default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--user",     default=os.getenv("MQTT_USER",    ""))
    parser.add_argument("--password", default=os.getenv("MQTT_PASS",    ""))
    parser.add_argument("--tls",      action="store_true",
                        default=os.getenv("MQTT_USE_TLS", "").lower() == "true")
    parser.add_argument("--model",
                        default=os.getenv("MODEL_PATH",
                                          "ml/data/checkpoints/transformer_thermal.onnx"))
    parser.add_argument("--meta",
                        default=os.getenv("META_PATH",
                                          "ml/data/checkpoints/transformer_thermal.json"))
    parser.add_argument("--scaler",
                        default=os.getenv("SCALER_PATH",
                                          "ml/data/checkpoints/feature_scaler.pkl"))
    args = parser.parse_args()

    service = PredictionService(args)

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(sig, frame):
        log.info(f"Signal {sig} received — shutting down")
        service.running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    service.run_forever()


if __name__ == "__main__":
    main()

