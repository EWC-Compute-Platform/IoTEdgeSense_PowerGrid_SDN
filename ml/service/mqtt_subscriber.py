"""
mqtt_subscriber.py
==================
Reusable MQTT subscriber base class for the IoTEdgeSense ML service layer.

Provides:
  - Clean connect/disconnect lifecycle with auto-reconnect
  - Topic-to-handler routing (subscribe to many topics, dispatch to
    the right handler by exact match or wildcard prefix)
  - Thread-safe message queue with configurable backpressure
  - Connection health monitoring (last-seen timestamp per topic)
  - Structured logging consistent with the rest of the platform
  - TLS support (CA cert + optional client cert)

PredictionService builds on top of MqttSubscriber by calling
  subscriber.register_handler("devices/data", self._on_telemetry)
and never touching paho directly.

Usage as a standalone subscriber:

    from ml.service.mqtt_subscriber import MqttSubscriber, MqttConfig

    cfg = MqttConfig(broker="mqtt.example.com", port=8883, use_tls=True)
    sub = MqttSubscriber(cfg)

    @sub.on_topic("devices/data")
    def handle_telemetry(topic: str, payload: dict):
        print(f"Received {len(payload)} readings on {topic}")

    sub.start()
    sub.run_until_stopped()    # blocks; Ctrl-C triggers clean shutdown
"""

from __future__ import annotations

import json
import logging
import queue
import signal
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.client import MQTTMessage
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False
    mqtt = None        # type: ignore
    MQTTMessage = Any  # type: ignore


log = logging.getLogger("MqttSubscriber")

# Type alias for message handler callbacks
MessageHandler = Callable[[str, Any], None]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MqttConfig:
    """Connection and behaviour configuration for MqttSubscriber."""

    # Broker connection
    broker:   str = "localhost"
    port:     int = 1883
    client_id: str = "iotedgesense-ml"
    keepalive: int = 60

    # Authentication
    username: str = ""
    password: str = ""

    # TLS
    use_tls:       bool = False
    ca_cert:       str  = ""     # path to CA certificate
    client_cert:   str  = ""     # path to client certificate (mutual TLS)
    client_key:    str  = ""     # path to client private key
    tls_insecure:  bool = False   # set True only for development

    # Reconnection
    reconnect_delay_s:     float = 5.0
    reconnect_max_delay_s: float = 120.0
    reconnect_backoff:     float = 2.0    # exponential backoff multiplier

    # Message queue
    queue_maxsize: int = 10000   # 0 = unlimited
    queue_timeout: float = 0.1   # seconds to wait when queue is full

    # Topics published by this client (for heartbeat / status)
    status_topic:  str = "devices/ml/status"
    publish_qos:   int = 1


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------

class ConnectionState:
    DISCONNECTED  = "DISCONNECTED"
    CONNECTING    = "CONNECTING"
    CONNECTED     = "CONNECTED"
    RECONNECTING  = "RECONNECTING"


# ---------------------------------------------------------------------------
# MqttSubscriber
# ---------------------------------------------------------------------------

class MqttSubscriber:
    """
    Thread-safe MQTT subscriber with topic routing and auto-reconnect.

    The subscriber runs a background thread that processes the inbound
    message queue and dispatches to registered handlers. Handlers are
    called from that background thread — they must not block for extended
    periods. For heavy processing (ML inference), offload to a thread pool.
    """

    def __init__(self, cfg: MqttConfig):
        if not PAHO_AVAILABLE:
            raise ImportError(
                "paho-mqtt is required: pip install paho-mqtt"
            )

        self.cfg   = cfg
        self._state = ConnectionState.DISCONNECTED
        self._stop  = threading.Event()
        self._lock  = threading.Lock()

        # Topic routing: exact matches and prefix matches
        self._handlers:       Dict[str, List[MessageHandler]] = {}
        self._prefix_handlers: Dict[str, List[MessageHandler]] = {}

        # Subscriptions to register on (re)connect
        self._subscriptions: Dict[str, int] = {}  # topic → qos

        # Inbound message queue (paho thread → dispatch thread)
        self._queue: queue.Queue = queue.Queue(maxsize=cfg.queue_maxsize)

        # Per-topic health tracking
        self._last_seen: Dict[str, float] = {}
        self._msg_counts: Dict[str, int]  = {}

        # Reconnect state
        self._reconnect_delay = cfg.reconnect_delay_s
        self._connect_ts: float = 0.0

        # Build paho client
        self._client = mqtt.Client(
            client_id=cfg.client_id,
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message
        self._client.on_subscribe  = self._on_subscribe

        if cfg.username:
            self._client.username_pw_set(cfg.username, cfg.password)

        if cfg.use_tls:
            self._configure_tls()

        # Dispatch thread (processes queue → calls handlers)
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="mqtt-dispatch",
            daemon=True,
        )

    # ── TLS ────────────────────────────────────────────────────────────────

    def _configure_tls(self):
        ctx = ssl.create_default_context()
        if self.cfg.ca_cert:
            ctx.load_verify_locations(cafile=self.cfg.ca_cert)
        if self.cfg.client_cert and self.cfg.client_key:
            ctx.load_cert_chain(
                certfile=self.cfg.client_cert,
                keyfile=self.cfg.client_key,
            )
        if self.cfg.tls_insecure:
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
        self._client.tls_set_context(ctx)

    # ── Handler registration ───────────────────────────────────────────────

    def register_handler(self,
                          topic: str,
                          handler: MessageHandler,
                          qos: int = 1,
                          prefix_match: bool = False):
        """
        Register a callback for a specific MQTT topic.

        @param topic         MQTT topic string (exact or prefix)
        @param handler       Callable(topic: str, payload: Any)
                             Payload is auto-decoded: dict if JSON, else raw bytes.
        @param qos           QoS level for the subscription
        @param prefix_match  If True, handler fires for any topic starting with `topic`
        """
        with self._lock:
            if prefix_match:
                self._prefix_handlers.setdefault(topic, []).append(handler)
            else:
                self._handlers.setdefault(topic, []).append(handler)
            self._subscriptions[topic] = qos

        log.debug(f"Registered handler for topic '{topic}' "
                   f"(prefix={prefix_match}, qos={qos})")

    def on_topic(self, topic: str, qos: int = 1, prefix: bool = False):
        """
        Decorator form of register_handler:

            @subscriber.on_topic("devices/data")
            def handle(topic, payload):
                ...
        """
        def decorator(fn: MessageHandler) -> MessageHandler:
            self.register_handler(topic, fn, qos=qos, prefix_match=prefix)
            return fn
        return decorator

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        """Connect to broker and start background threads."""
        log.info(f"Connecting to {self.cfg.broker}:{self.cfg.port} "
                  f"(tls={self.cfg.use_tls})")
        self._state = ConnectionState.CONNECTING
        self._connect_ts = time.time()

        self._client.connect_async(
            host=self.cfg.broker,
            port=self.cfg.port,
            keepalive=self.cfg.keepalive,
        )
        self._client.loop_start()
        self._dispatch_thread.start()

    def stop(self, timeout: float = 5.0):
        """Graceful shutdown: drain queue, disconnect, stop threads."""
        log.info("Stopping MQTT subscriber...")
        self._stop.set()
        self._publish_status("offline")
        time.sleep(0.2)
        self._client.loop_stop()
        self._client.disconnect()
        self._dispatch_thread.join(timeout=timeout)
        log.info("MQTT subscriber stopped")

    def run_until_stopped(self):
        """
        Block the calling thread until stop() is called or SIGINT/SIGTERM.
        Suitable for using MqttSubscriber as a standalone process.
        """
        def _sig(s, f):
            log.info(f"Signal {s} — stopping")
            self.stop()

        signal.signal(signal.SIGINT,  _sig)
        signal.signal(signal.SIGTERM, _sig)

        while not self._stop.is_set():
            self._stop.wait(timeout=30.0)
            self._log_health()

    # ── Paho callbacks ─────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties=None):
        if reason_code == 0:
            self._state = ConnectionState.CONNECTED
            self._reconnect_delay = self.cfg.reconnect_delay_s  # reset backoff
            log.info(f"Connected to {self.cfg.broker}:{self.cfg.port}")
            # Re-subscribe to all registered topics
            with self._lock:
                subs = list(self._subscriptions.items())
            for topic, qos in subs:
                client.subscribe(topic, qos=qos)
                log.debug(f"Subscribed: {topic} (qos={qos})")
            self._publish_status("online")
        else:
            self._state = ConnectionState.RECONNECTING
            log.error(f"Connection failed: reason_code={reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        self._state = ConnectionState.RECONNECTING
        if reason_code != 0:
            log.warning(f"Unexpected disconnect: rc={reason_code}. "
                         f"Reconnecting in {self._reconnect_delay:.0f}s...")
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * self.cfg.reconnect_backoff,
                self.cfg.reconnect_max_delay_s,
            )

    def _on_message(self, client, userdata, msg: MQTTMessage):
        """Called by paho network thread — enqueue immediately, decode later."""
        try:
            self._queue.put_nowait((msg.topic, msg.payload))
        except queue.Full:
            log.warning(f"Message queue full — dropping message on {msg.topic}")

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties=None):
        log.debug(f"Subscription confirmed mid={mid} "
                   f"reason_codes={reason_codes}")

    # ── Dispatch loop (background thread) ──────────────────────────────────

    def _dispatch_loop(self):
        log.debug("Dispatch loop started")
        while not self._stop.is_set():
            try:
                topic, raw_payload = self._queue.get(
                    timeout=self.cfg.queue_timeout
                )
            except queue.Empty:
                continue

            # Decode payload
            try:
                payload = json.loads(raw_payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = raw_payload  # pass raw bytes to handler

            # Update health tracking
            self._last_seen[topic] = time.time()
            self._msg_counts[topic] = self._msg_counts.get(topic, 0) + 1

            # Exact-match handlers
            with self._lock:
                exact    = list(self._handlers.get(topic, []))
                prefixes = list(self._prefix_handlers.items())

            for handler in exact:
                try:
                    handler(topic, payload)
                except Exception as e:
                    log.error(f"Handler error on topic '{topic}': {e}",
                               exc_info=True)

            # Prefix-match handlers
            for prefix, handlers in prefixes:
                if topic.startswith(prefix):
                    for handler in handlers:
                        try:
                            handler(topic, payload)
                        except Exception as e:
                            log.error(
                                f"Prefix handler error "
                                f"(prefix='{prefix}', topic='{topic}'): {e}",
                                exc_info=True,
                            )

        log.debug("Dispatch loop stopped")

    # ── Publishing ─────────────────────────────────────────────────────────

    def publish(self,
                 topic: str,
                 payload: Any,
                 qos: int = None,
                 retain: bool = False) -> bool:
        """
        Publish a message. Payload is JSON-serialised if it is a dict/list.
        Returns True if the publish call was accepted by paho.
        """
        if isinstance(payload, (dict, list)):
            raw = json.dumps(payload)
        elif isinstance(payload, str):
            raw = payload
        else:
            raw = str(payload)

        if qos is None:
            qos = self.cfg.publish_qos

        result = self._client.publish(topic, raw, qos=qos, retain=retain)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            log.error(f"Publish failed on '{topic}': rc={result.rc}")
            return False
        return True

    def _publish_status(self, status: str):
        if self._state not in (ConnectionState.CONNECTED,
                                ConnectionState.DISCONNECTED):
            return
        self.publish(
            self.cfg.status_topic,
            {
                "service":      self.cfg.client_id,
                "status":       status,
                "ts":           int(time.time() * 1000),
                "broker":       self.cfg.broker,
                "subscriptions": list(self._subscriptions.keys()),
            },
            qos=0,
            retain=True,
        )

    # ── Health and introspection ────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    @property
    def connection_state(self) -> str:
        return self._state

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def topic_last_seen(self, topic: str) -> Optional[float]:
        """Unix timestamp of last message received on topic, or None."""
        return self._last_seen.get(topic)

    def topic_message_count(self, topic: str) -> int:
        return self._msg_counts.get(topic, 0)

    def health_summary(self) -> dict:
        """Returns a dict suitable for logging or a status endpoint."""
        now = time.time()
        topic_health = {}
        with self._lock:
            for topic in self._subscriptions:
                last = self._last_seen.get(topic)
                topic_health[topic] = {
                    "messages":    self._msg_counts.get(topic, 0),
                    "last_seen_s": round(now - last, 1) if last else None,
                }
        return {
            "state":         self._state,
            "broker":        f"{self.cfg.broker}:{self.cfg.port}",
            "queue_depth":   self._queue.qsize(),
            "uptime_s":      round(now - self._connect_ts, 1),
            "topics":        topic_health,
        }

    def _log_health(self):
        summary = self.health_summary()
        log.info(
            f"Health: state={summary['state']}  "
            f"queue={summary['queue_depth']}  "
            f"uptime={summary['uptime_s']}s  "
            + "  ".join(
                f"{t}: {v['messages']} msgs"
                for t, v in summary["topics"].items()
            )
        )


