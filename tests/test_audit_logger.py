import tempfile
import pytest
from pathlib import Path
from src.intrusion.audit_logger import ForensicIncidentLogger

def test_log_incident_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ForensicIncidentLogger(log_dir=Path(tmpdir))
        inc = logger.log_incident(
            track_id=1,
            class_name="person",
            confidence=0.92,
            bbox=[100.0, 150.0, 200.0, 300.0],
            zone_id="ZONE_TEST"
        )
        assert inc["track_id"] == 1
        assert inc["class_name"] == "person"
        assert inc["centroid_x"] == 150.0
        assert inc["centroid_y"] == 225.0

        rec = logger.get_recent_incidents()
        assert len(rec) == 1
        assert rec[0]["track_id"] == "1"

        csv_str = logger.export_csv_string()
        assert "incident_id,timestamp_iso" in csv_str
        assert "ZONE_TEST" in csv_str
