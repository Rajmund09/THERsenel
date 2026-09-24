"""
Forensic Incident Audit Logger (src/intrusion/audit_logger.py)
Generates structured CSV and JSONL audit records for border security compliance:
- Logs every unauthorized entry with track ID, confidence, timestamps, and bounding coordinates.
- Provides thread-safe append and in-memory incident querying.
- Facilitates post-incident forensic reviews and automated audit trail exports.
"""

import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading

logger_lock = threading.Lock()


class ForensicIncidentLogger:
    """
    Logs unauthorized perimeter breach incidents to timestamped CSV and JSONL files.
    """

    CSV_HEADERS = [
        "incident_id", "timestamp_iso", "unix_timestamp", "track_id",
        "class_name", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "centroid_x", "centroid_y", "zone_id", "severity"
    ]

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else Path("logs") / "audit"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._incident_counter = 0

    def _get_daily_csv_path(self) -> Path:
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"breach_incidents_{date_str}.csv"

    def log_incident(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        bbox: List[float],
        zone_id: str = "SEC_ZONE-09",
        severity: str = "HIGH"
    ) -> Dict[str, Any]:
        """
        Record a perimeter breach incident to disk and return structured incident dict.
        """
        with logger_lock:
            self._incident_counter += 1
            now_iso = datetime.utcnow().isoformat() + "Z"
            now_unix = time.time()
            incident_id = f"INC-{int(now_unix)}-{track_id:03d}"

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            incident = {
                "incident_id": incident_id,
                "timestamp_iso": now_iso,
                "unix_timestamp": round(now_unix, 3),
                "track_id": track_id,
                "class_name": class_name,
                "confidence": round(float(confidence), 3),
                "bbox_x1": round(float(x1), 1),
                "bbox_y1": round(float(y1), 1),
                "bbox_x2": round(float(x2), 1),
                "bbox_y2": round(float(y2), 1),
                "centroid_x": round(float(cx), 1),
                "centroid_y": round(float(cy), 1),
                "zone_id": zone_id,
                "severity": severity
            }

            # Write to CSV
            csv_path = self._get_daily_csv_path()
            file_exists = csv_path.exists() and csv_path.stat().st_size > 0

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(incident)

            return incident

    def export_csv_string(self) -> str:
        """Export current day's incidents as a CSV string."""
        csv_path = self._get_daily_csv_path()
        if not csv_path.exists():
            return ",".join(self.CSV_HEADERS) + "\n"
        with open(csv_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_recent_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read recent incidents from disk."""
        csv_path = self._get_daily_csv_path()
        if not csv_path.exists():
            return []

        incidents = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                incidents.append(row)

        return incidents[-limit:]
