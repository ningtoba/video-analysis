"""
Pipeline Health Monitor — automated quality tracking, drift detection & alerts.

Provides a comprehensive health monitoring system for the video analysis
pipeline that tracks:

- **Pipeline run metrics** — duration, success/failure rates, per-stage timing
- **Quality trends** — OCR accuracy, detection confidence, transcription WER
- **Anomaly detection** — statistical deviation from rolling baseline for every
  tracked metric; configurable sensitivity with automatic alert generation
- **Drift tracking** — per-metric drift detection using z-score or IQR methods,
  comparing the latest N runs against a rolling baseline window
- **Health scoring** — composite pipeline health score (0.0-1.0) aggregating
  all tracked metrics with configurable thresholds
- **Alert management** — auto-generated, deduplicated alerts with severity
  levels (info/warning/error/critical), suppression windows, and expiry
- **Reporting** — structured health reports for dashboard rendering, API
  endpoints, and LLM context injection

All data is persisted to SQLite (co-located with the knowledge graph data dir)
so health history survives restarts.

Usage:
    from video_analysis.pipeline_health import PipelineHealthMonitor

    monitor = PipelineHealthMonitor(config)

    # After each pipeline run:
    monitor.record_run(
        video_id="abc123",
        duration_s=45.2,
        success=True,
        stage_timings={"scene_detection": 5.2, "ocr": 12.1, ...},
    )

    # Check health:
    report = monitor.get_health_report()
    alerts = monitor.get_active_alerts()
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from video_analysis.config import Config

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_WINDOW_SIZE = 50  # number of recent runs for rolling baseline
DEFAULT_Z_SCORE_THRESHOLD = 3.0  # z-score > this = anomaly
DEFAULT_MIN_DATA_POINTS = 10  # minimum datapoints before drift detection activates
DEFAULT_ALERT_COOLDOWN_SECONDS = 3600  # suppress duplicate alerts for 1 hour
DEFAULT_ALERT_EXPIRY_SECONDS = 86400  # auto-expire alerts after 24h
HEALTH_TOP_ENTITIES = 5  # top entities per video for health reports

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}

# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class PipelineRun:
    """A single pipeline run record.

    Attributes:
        run_id: Auto-incrementing run ID.
        video_id: ID of the video being processed.
        timestamp: Unix timestamp of the run.
        duration_s: Total pipeline duration in seconds.
        success: Whether the pipeline completed without error.
        stage_timings: Dict of stage name → duration in seconds.
        stage_successes: Dict of stage name → success bool.
        ocr_confidence: Average OCR confidence (0.0-1.0).
        detection_confidence: Average detection confidence (0.0-1.0).
        transcript_confidence: Average transcription confidence (0.0-1.0).
        metadata: Arbitrary JSON metadata.
    """

    run_id: int = 0
    video_id: str = ""
    timestamp: float = 0.0
    duration_s: float = 0.0
    success: bool = True
    stage_timings: Dict[str, float] = field(default_factory=dict)
    stage_successes: Dict[str, bool] = field(default_factory=dict)
    ocr_confidence: Optional[float] = None
    detection_confidence: Optional[float] = None
    transcript_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSnapshot:
    """A snapshot of a single metric at a point in time."""

    metric_name: str = ""
    value: float = 0.0
    timestamp: float = 0.0
    is_anomaly: bool = False
    z_score: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 0.0


@dataclass
class HealthAlert:
    """An auto-generated health alert.

    Attributes:
        alert_id: Auto-incrementing alert ID.
        severity: Alert severity ('info', 'warning', 'error', 'critical').
        title: Short alert title.
        message: Human-readable description.
        metric_name: The metric that triggered the alert.
        value: The metric value that triggered the alert.
        threshold: The threshold/baseline that was breached.
        created_at: Unix timestamp when the alert was created.
        expires_at: Unix timestamp when the alert auto-expires.
        acknowledged: Whether the alert has been acknowledged.
    """

    alert_id: int = 0
    severity: str = "warning"
    title: str = ""
    message: str = ""
    metric_name: str = ""
    value: float = 0.0
    threshold: float = 0.0
    created_at: float = 0.0
    expires_at: float = 0.0
    acknowledged: bool = False


@dataclass
class HealthReport:
    """Composite health report for the pipeline."""

    health_score: float = 1.0
    total_runs: int = 0
    recent_runs: int = 0
    success_rate: float = 1.0
    avg_duration_s: float = 0.0
    anomaly_count: int = 0
    active_alerts: int = 0
    alerts: List[HealthAlert] = field(default_factory=list)
    metrics: List[MetricSnapshot] = field(default_factory=list)
    degraded_metrics: List[str] = field(default_factory=list)
    run_history: List[PipelineRun] = field(default_factory=list)


# ── Pipeline Health Monitor ──────────────────────────────────────────────


class PipelineHealthMonitor:
    """Persistent pipeline health monitor backed by SQLite.

    Thread-safe — uses a per-instance reentrant lock so it can be called
    from multiple workers.

    Schema:
        - pipeline_runs: run_id, video_id, timestamp, duration_s, success,
                          stage_timings (JSON), stage_successes (JSON),
                          ocr_confidence, detection_confidence,
                          transcript_confidence, metadata (JSON)
        - health_alerts: alert_id, severity, title, message, metric_name,
                          value, threshold, created_at, expires_at, acknowledged
    """

    def __init__(
        self,
        config=None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        z_score_threshold: float = DEFAULT_Z_SCORE_THRESHOLD,
        min_data_points: int = DEFAULT_MIN_DATA_POINTS,
        alert_cooldown_s: float = DEFAULT_ALERT_COOLDOWN_SECONDS,
        alert_expiry_s: float = DEFAULT_ALERT_EXPIRY_SECONDS,
    ):
        self._config = config or Config()
        self._db_path: Path = self._config.data_dir / "pipeline_health.db"
        self._window_size = window_size
        self._z_score_threshold = z_score_threshold
        self._min_data_points = min_data_points
        self._alert_cooldown_s = alert_cooldown_s
        self._alert_expiry_s = alert_expiry_s
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── Database init ───────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create or open the SQLite database and ensure schema exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            self._conn.execute("""CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    duration_s REAL NOT NULL DEFAULT 0.0,
                    success INTEGER NOT NULL DEFAULT 1,
                    stage_timings TEXT DEFAULT '{}',
                    stage_successes TEXT DEFAULT '{}',
                    ocr_confidence REAL,
                    detection_confidence REAL,
                    transcript_confidence REAL,
                    metadata TEXT DEFAULT '{}'
                )""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS health_alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    severity TEXT NOT NULL DEFAULT 'warning',
                    title TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    metric_name TEXT NOT NULL DEFAULT '',
                    value REAL NOT NULL DEFAULT 0.0,
                    threshold REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0
                )""")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_timestamp "
                "ON pipeline_runs(timestamp DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_video "
                "ON pipeline_runs(video_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_severity "
                "ON health_alerts(severity)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_active "
                "ON health_alerts(expires_at, acknowledged)"
            )
            self._conn.commit()

    # ── Run recording ──────────────────────────────────────────────────

    def record_run(
        self,
        video_id: str,
        duration_s: float,
        success: bool = True,
        stage_timings: Optional[Dict[str, float]] = None,
        stage_successes: Optional[Dict[str, bool]] = None,
        ocr_confidence: Optional[float] = None,
        detection_confidence: Optional[float] = None,
        transcript_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a pipeline run and check for anomalies.

        Returns:
            The run ID.
        """
        now = time.time()
        timings = stage_timings or {}
        successes = stage_successes or {}
        meta = metadata or {}

        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO pipeline_runs
                   (video_id, timestamp, duration_s, success,
                    stage_timings, stage_successes,
                    ocr_confidence, detection_confidence,
                    transcript_confidence, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video_id,
                    now,
                    duration_s,
                    int(success),
                    json.dumps(timings),
                    json.dumps(successes),
                    ocr_confidence,
                    detection_confidence,
                    transcript_confidence,
                    json.dumps(meta),
                ),
            )
            run_id = cur.lastrowid or 0
            self._conn.commit()

        # Check for anomalies after recording (outside lock for speed)
        self._check_all_metrics()

        return run_id

    # ── Anomaly detection ───────────────────────────────────────────────

    def _get_metric_values(self, metric_name: str) -> List[float]:
        """Get recent values for a metric (newest first)."""
        with self._lock:
            if metric_name == "duration_s":
                rows = self._conn.execute(
                    "SELECT duration_s FROM pipeline_runs "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (self._window_size,),
                ).fetchall()
                return [r["duration_s"] for r in rows]
            elif metric_name == "ocr_confidence":
                rows = self._conn.execute(
                    "SELECT ocr_confidence FROM pipeline_runs "
                    "WHERE ocr_confidence IS NOT NULL "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (self._window_size,),
                ).fetchall()
                return [r["ocr_confidence"] for r in rows]
            elif metric_name == "detection_confidence":
                rows = self._conn.execute(
                    "SELECT detection_confidence FROM pipeline_runs "
                    "WHERE detection_confidence IS NOT NULL "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (self._window_size,),
                ).fetchall()
                return [r["detection_confidence"] for r in rows]
            elif metric_name == "transcript_confidence":
                rows = self._conn.execute(
                    "SELECT transcript_confidence FROM pipeline_runs "
                    "WHERE transcript_confidence IS NOT NULL "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (self._window_size,),
                ).fetchall()
                return [r["transcript_confidence"] for r in rows]
            elif metric_name.startswith("stage_"):
                stage = metric_name[6:]
                rows = self._conn.execute(
                    "SELECT stage_timings FROM pipeline_runs "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (self._window_size,),
                ).fetchall()
                values = []
                for r in rows:
                    timings = json.loads(r["stage_timings"] or "{}")
                    if stage in timings:
                        values.append(timings[stage])
                return values
            return []

    def _compute_baseline(self, values: List[float]) -> Tuple[float, float]:
        """Compute mean and std from a list of values.

        Returns (mean, std).  Returns (0, 0) for empty lists.
        """
        if len(values) < 2:
            return (0.0, 0.0)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        return (mean, std)

    def _check_metric_anomaly(
        self, metric_name: str, latest_value: float
    ) -> Optional[MetricSnapshot]:
        """Check if the latest value of a metric is anomalous.

        Returns a MetricSnapshot if sufficient data exists, else None.
        """
        values = self._get_metric_values(metric_name)
        if len(values) < self._min_data_points:
            return None

        baseline_mean, baseline_std = self._compute_baseline(values)
        if baseline_std == 0:
            return None

        z_score = abs((latest_value - baseline_mean) / baseline_std)
        is_anomaly = z_score > self._z_score_threshold

        return MetricSnapshot(
            metric_name=metric_name,
            value=latest_value,
            timestamp=time.time(),
            is_anomaly=is_anomaly,
            z_score=z_score,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
        )

    def _check_all_metrics(self) -> List[HealthAlert]:
        """Check all tracked metrics for anomalies.

        Returns any newly generated alerts.
        """
        with self._lock:
            # Get latest run values
            latest = self._conn.execute(
                "SELECT * FROM pipeline_runs ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if not latest:
                return []

            new_alerts: List[HealthAlert] = []
            metrics_to_check = [
                ("duration_s", latest["duration_s"]),
            ]

            if latest["ocr_confidence"] is not None:
                metrics_to_check.append(("ocr_confidence", latest["ocr_confidence"]))
            if latest["detection_confidence"] is not None:
                metrics_to_check.append(
                    ("detection_confidence", latest["detection_confidence"])
                )
            if latest["transcript_confidence"] is not None:
                metrics_to_check.append(
                    ("transcript_confidence", latest["transcript_confidence"])
                )

            # Stage timings
            timings = json.loads(latest["stage_timings"] or "{}")
            for stage, dur in timings.items():
                metrics_to_check.append((f"stage_{stage}", dur))

            for metric_name, value in metrics_to_check:
                snapshot = self._check_metric_anomaly(metric_name, value)
                if snapshot and snapshot.is_anomaly:
                    alert = self._create_alert(metric_name, snapshot)
                    if alert:
                        new_alerts.append(alert)

            self._expire_old_alerts()
            return new_alerts

    def _create_alert(
        self, metric_name: str, snapshot: MetricSnapshot
    ) -> Optional[HealthAlert]:
        """Create an alert if no recent duplicate exists for this metric."""
        now = time.time()
        cooldown_start = now - self._alert_cooldown_s

        # Check for recent duplicate
        existing = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM health_alerts "
            "WHERE metric_name = ? AND created_at > ? "
            "AND acknowledged = 0 AND expires_at > ?",
            (metric_name, cooldown_start, now),
        ).fetchone()
        if existing and existing["cnt"] > 0:
            return None  # Duplicate suppressed

        # Determine severity based on z-score
        z = snapshot.z_score
        if z > 5.0:
            severity = "critical"
        elif z > 4.0:
            severity = "error"
        elif z > self._z_score_threshold:
            severity = "warning"
        else:
            severity = "info"

        value_diff = snapshot.value - snapshot.baseline_mean
        direction = "increased" if value_diff > 0 else "decreased"
        pct_change = (
            abs(value_diff) / snapshot.baseline_mean * 100
            if snapshot.baseline_mean > 0
            else 0
        )

        title = f"Anomaly detected: {metric_name}"
        message = (
            f"Metric '{metric_name}' {direction} by {pct_change:.1f}% "
            f"(value={snapshot.value:.3f}, baseline={snapshot.baseline_mean:.3f}, "
            f"z-score={snapshot.z_score:.2f})"
        )

        created_at = now
        expires_at = now + self._alert_expiry_s

        cur = self._conn.execute(
            """INSERT INTO health_alerts
               (severity, title, message, metric_name, value,
                threshold, created_at, expires_at, acknowledged)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                severity,
                title,
                message,
                metric_name,
                snapshot.value,
                snapshot.baseline_mean
                + self._z_score_threshold * snapshot.baseline_std,
                created_at,
                expires_at,
            ),
        )
        self._conn.commit()
        alert_id = cur.lastrowid or 0

        logger.warning(
            "Health alert [%s] %s: %s",
            severity,
            metric_name,
            message,
        )

        # Fire webhook for critical alerts (v0.59.0)
        if severity in ("critical", "error"):
            try:
                from video_analysis.webhook import get_webhook_dispatcher

                wh = get_webhook_dispatcher()
                if wh.enabled:
                    event = (
                        "health.critical" if severity == "critical" else "health.alert"
                    )
                    wh.fire(
                        event,
                        {
                            "alert_id": alert_id,
                            "severity": severity,
                            "title": title,
                            "message": message,
                            "metric_name": metric_name,
                            "value": snapshot.value,
                            "z_score": snapshot.z_score,
                        },
                    )
            except Exception:
                pass

        return HealthAlert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            value=snapshot.value,
            threshold=snapshot.baseline_mean
            + self._z_score_threshold * snapshot.baseline_std,
            created_at=created_at,
            expires_at=expires_at,
        )

    # ── Alert management ───────────────────────────────────────────────

    def _expire_old_alerts(self) -> int:
        """Expire alerts whose expiry time has passed. Returns count expired."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM health_alerts WHERE expires_at <= ?",
                (time.time(),),
            )
            self._conn.commit()
            return cur.rowcount

    def get_active_alerts(
        self, min_severity: str = "info", limit: int = 50
    ) -> List[HealthAlert]:
        """Get all active (non-expired, non-acknowledged) alerts."""
        min_order = SEVERITY_ORDER.get(min_severity, 0)
        severities = [s for s, o in SEVERITY_ORDER.items() if o >= min_order]
        with self._lock:
            self._expire_old_alerts()
            placeholders = ",".join("?" for _ in severities)
            rows = self._conn.execute(
                f"SELECT * FROM health_alerts "
                f"WHERE acknowledged = 0 AND severity IN ({placeholders}) "
                f"ORDER BY created_at DESC LIMIT ?",
                (*severities, limit),
            ).fetchall()
            return [self._row_to_alert(r) for r in rows]

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged. Returns True if found."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE health_alerts SET acknowledged = 1 " "WHERE alert_id = ?",
                (alert_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def acknowledge_all_alerts(self) -> int:
        """Acknowledge all active alerts. Returns count acknowledged."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE health_alerts SET acknowledged = 1 " "WHERE acknowledged = 0"
            )
            self._conn.commit()
            return cur.rowcount

    # ── Health scoring ─────────────────────────────────────────────────

    def compute_health_score(self) -> float:
        """Compute composite pipeline health score (0.0-1.0).

        Factors:
        - Success rate of recent runs (weight=0.4)
        - Active alert severity (weight=0.3)
        - Duration stability (weight=0.15)
        - Confidence metrics (weight=0.15)
        """
        with self._lock:
            self._expire_old_alerts()

            # Success rate (recent 50 runs)
            recent = self._conn.execute(
                "SELECT success, duration_s, ocr_confidence, "
                "detection_confidence, transcript_confidence "
                "FROM pipeline_runs ORDER BY timestamp DESC LIMIT ?",
                (self._window_size,),
            ).fetchall()

            if not recent:
                return 1.0  # No data yet — assume healthy

            n = len(recent)
            success_rate = sum(r["success"] for r in recent) / n
            duration_values = [r["duration_s"] for r in recent]
            ocr_values = [
                r["ocr_confidence"] for r in recent if r["ocr_confidence"] is not None
            ]
            det_values = [
                r["detection_confidence"]
                for r in recent
                if r["detection_confidence"] is not None
            ]
            trans_values = [
                r["transcript_confidence"]
                for r in recent
                if r["transcript_confidence"] is not None
            ]

            # Duration stability: lower CV (coefficient of variation) = better
            if len(duration_values) > 1:
                dur_mean = sum(duration_values) / len(duration_values)
                dur_var = sum((v - dur_mean) ** 2 for v in duration_values) / (
                    len(duration_values) - 1
                )
                dur_cv = math.sqrt(dur_var) / dur_mean if dur_mean > 0 else 1.0
                duration_stability = max(0.0, 1.0 - min(dur_cv, 1.0))
            else:
                duration_stability = 1.0

            # Confidence health: average of available confidence metrics
            conf_vals = ocr_values + det_values + trans_values
            avg_confidence = sum(conf_vals) / len(conf_vals) if conf_vals else 1.0

            # Alert severity penalty
            alerts = self._conn.execute(
                "SELECT severity FROM health_alerts " "WHERE acknowledged = 0"
            ).fetchall()
            alert_penalty = 0.0
            for a in alerts:
                sev_order = SEVERITY_ORDER.get(a["severity"], 0)
                if sev_order >= 4:  # critical
                    alert_penalty += 0.25
                elif sev_order >= 3:  # error
                    alert_penalty += 0.15
                elif sev_order >= 2:  # warning
                    alert_penalty += 0.08
                else:
                    alert_penalty += 0.03
            alert_health = max(0.0, 1.0 - min(alert_penalty, 1.0))

            score = (
                0.4 * success_rate
                + 0.3 * alert_health
                + 0.15 * duration_stability
                + 0.15 * avg_confidence
            )
            return round(max(0.0, min(1.0, score)), 4)

    # ── Reports ─────────────────────────────────────────────────────────

    def get_health_report(self, recent_count: int = 20) -> HealthReport:
        """Generate a composite health report.

        Args:
            recent_count: Number of recent runs to include in the report.

        Returns:
            A HealthReport dataclass with all current health data.
        """
        score = self.compute_health_score()

        with self._lock:
            self._expire_old_alerts()

            total = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM pipeline_runs"
            ).fetchone()["cnt"]

            recent = self._conn.execute(
                "SELECT * FROM pipeline_runs " "ORDER BY timestamp DESC LIMIT ?",
                (recent_count,),
            ).fetchall()

            runs = [self._row_to_run(r) for r in recent]
            recent_successes = sum(1 for r in runs if r.success)
            success_rate = recent_successes / len(runs) if runs else 1.0
            avg_dur = sum(r.duration_s for r in runs) / len(runs) if runs else 0.0

            alerts = self.get_active_alerts()

            # Collect metric snapshots
            metric_snapshots: List[MetricSnapshot] = []
            degraded_metrics: List[str] = []

            for metric in [
                "duration_s",
                "ocr_confidence",
                "detection_confidence",
                "transcript_confidence",
            ]:
                values = self._get_metric_values(metric)
                if len(values) >= self._min_data_points:
                    mean, std = self._compute_baseline(values)
                    latest = values[0] if values else 0.0
                    z = abs((latest - mean) / std) if std > 0 else 0.0
                    is_anom = z > self._z_score_threshold
                    metric_snapshots.append(
                        MetricSnapshot(
                            metric_name=metric,
                            value=latest,
                            timestamp=time.time(),
                            is_anomaly=is_anom,
                            z_score=z,
                            baseline_mean=mean,
                            baseline_std=std,
                        )
                    )
                    if is_anom:
                        degraded_metrics.append(metric)

            return HealthReport(
                health_score=score,
                total_runs=total,
                recent_runs=len(runs),
                success_rate=success_rate,
                avg_duration_s=avg_dur,
                anomaly_count=sum(1 for m in metric_snapshots if m.is_anomaly),
                active_alerts=len(alerts),
                alerts=alerts,
                metrics=metric_snapshots,
                degraded_metrics=degraded_metrics,
                run_history=runs,
            )

    def get_health_summary(self) -> Dict[str, Any]:
        """Return a concise health summary dict for API/UI consumption."""
        report = self.get_health_report(recent_count=20)
        return {
            "health_score": report.health_score,
            "total_runs": report.total_runs,
            "recent_runs": report.recent_runs,
            "success_rate": report.success_rate,
            "avg_duration_s": round(report.avg_duration_s, 1),
            "active_alerts": report.active_alerts,
            "anomalies": report.anomaly_count,
            "degraded_metrics": report.degraded_metrics,
            "status": (
                "healthy"
                if report.health_score >= 0.8
                else "degraded" if report.health_score >= 0.5 else "unhealthy"
            ),
        }

    def get_health_context(self) -> str:
        """Generate a compact LLM-friendly health summary for context injection."""
        report = self.get_health_report(recent_count=20)
        status = (
            "healthy"
            if report.health_score >= 0.8
            else "degraded" if report.health_score >= 0.5 else "unhealthy"
        )
        lines = [
            "## Pipeline Health Summary",
            f"",
            f"- **Health score**: {report.health_score:.2f}/1.0",
            f"- **Status**: {status}",
            f"- **Total runs**: {report.total_runs}",
            f"- **Recent success rate**: {report.success_rate:.1%}",
            f"- **Average duration**: {report.avg_duration_s:.1f}s",
            f"- **Active alerts**: {report.active_alerts}",
            f"- **Degraded metrics**: {report.degraded_metrics or 'none'}",
        ]
        if report.alerts:
            lines.append("")
            lines.append("### Active alerts")
            for a in report.alerts[:10]:
                lines.append(f"- [{a.severity}] {a.title}: {a.message}")
        return "\n".join(lines)

    # ── Maintenance ─────────────────────────────────────────────────────

    def vacuum(self) -> None:
        """Reclaim storage space."""
        with self._lock:
            self._conn.execute("PRAGMA optimize")
            self._conn.execute("VACUUM")
            self._conn.commit()

    def clear_runs(self, older_than_days: int = 0) -> int:
        """Clear old pipeline runs.  If older_than_days is 0, clears all."""
        with self._lock:
            if older_than_days > 0:
                cutoff = time.time() - older_than_days * 86400
                cur = self._conn.execute(
                    "DELETE FROM pipeline_runs WHERE timestamp < ?",
                    (cutoff,),
                )
            else:
                cur = self._conn.execute("DELETE FROM pipeline_runs")
            self._conn.commit()
            return cur.rowcount

    def clear_alerts(self) -> int:
        """Clear all alerts."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM health_alerts")
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.execute("PRAGMA optimize")
                self._conn.close()
                self._conn = None

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> PipelineRun:
        return PipelineRun(
            run_id=row["run_id"],
            video_id=row["video_id"],
            timestamp=row["timestamp"],
            duration_s=row["duration_s"],
            success=bool(row["success"]),
            stage_timings=json.loads(row["stage_timings"] or "{}"),
            stage_successes=json.loads(row["stage_successes"] or "{}"),
            ocr_confidence=row["ocr_confidence"],
            detection_confidence=row["detection_confidence"],
            transcript_confidence=row["transcript_confidence"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> HealthAlert:
        return HealthAlert(
            alert_id=row["alert_id"],
            severity=row["severity"],
            title=row["title"],
            message=row["message"],
            metric_name=row["metric_name"],
            value=row["value"],
            threshold=row["threshold"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            acknowledged=bool(row["acknowledged"]),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
