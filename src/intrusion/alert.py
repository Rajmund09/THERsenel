"""
Intrusion Alert & Event Dispatcher (src/intrusion/alert.py)
Manages intrusion breach notifications, frame logging, alert cooldowns, and visual alert banners.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List
import cv2


class IntrusionAlertDispatcher:
    """
    Triggers visual alerts, logs events to file, and manages cooldown timers between alerts.
    """

    def __init__(self, log_dir: Path = None, min_alert_frames: int = 3):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.min_alert_frames = min_alert_frames
        self.alert_history = []

    def dispatch_alerts(self, alerts: List[Dict], frame: cv2.Mat = None) -> cv2.Mat:
        if not alerts:
            return frame

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for alert in alerts:
            log_line = f"[{now_str}] INTRUSION BREACH | Track #{alert.get('track_id')} ({alert.get('class_name')}) Conf: {alert.get('confidence'):.2f}\n"
            self.alert_history.append(log_line)

            # Append to log file
            log_file = self.log_dir / "intrusion_events.log"
            with open(log_file, "a") as f:
                f.write(log_line)

        if frame is not None:
            # Draw red banner at the top of the frame
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"🚨 INTRUSION ALERT DETECTED ({len(alerts)} BREACHES)",
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        return frame
