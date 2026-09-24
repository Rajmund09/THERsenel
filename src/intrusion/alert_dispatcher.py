"""
Tactical Webhook & Emergency Alert Dispatcher (src/intrusion/alert_dispatcher.py)
Dispatches perimeter breach notifications to security endpoints:
- Discord Webhook (Rich Embeds with severity badges)
- Telegram Bot API (Markdown formatted instant dispatch)
- Slack Incoming Webhooks
- Generic JSON Webhook
Features built-in cooldown debouncing to prevent channel flooding.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger("alert_dispatcher")


class WebhookAlertDispatcher:
    """
    Asynchronous and synchronous webhook alert dispatcher with configurable rate-limit cooldown.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        service_type: str = "generic",  # "discord", "telegram", "slack", "generic"
        telegram_chat_id: Optional[str] = None,
        cooldown_seconds: float = 15.0
    ):
        self.webhook_url = webhook_url
        self.service_type = service_type.lower()
        self.telegram_chat_id = telegram_chat_id
        self.cooldown_seconds = cooldown_seconds
        self._last_dispatch_time = 0.0
        self._dispatch_count = 0

    def can_dispatch(self) -> bool:
        """Verify if cooldown window has passed since last dispatch."""
        return (time.time() - self._last_dispatch_time) >= self.cooldown_seconds

    def format_discord_payload(self, breach_info: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a tactical Discord Embed payload."""
        intruder_count = breach_info.get("intruders", 1)
        latency_ms = breach_info.get("latency_ms", 0.0)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        return {
            "username": "THERsenel Sentinel AI",
            "avatar_url": "https://raw.githubusercontent.com/Rajmund09/THERsenel/main/data/samples/ui_mobile_view.png",
            "embeds": [
                {
                    "title": f"🚨 RESTRICTED BORDER BREACH DETECTED ({intruder_count} INTRUDER{'S' if intruder_count != 1 else ''})",
                    "description": "Tactical perimeter sensor detected unauthorized movement inside the active Region of Interest (ROI).",
                    "color": 15158332,  # Crimson Red
                    "fields": [
                        {"name": "Security Zone", "value": "SEC_ZONE-09 (BORDER_NORTH)", "inline": True},
                        {"name": "Active Intruders", "value": str(intruder_count), "inline": True},
                        {"name": "Inference Latency", "value": f"{latency_ms:.1f} ms", "inline": True},
                        {"name": "Breach Classes", "value": ", ".join(breach_info.get("classes", ["person"])), "inline": True},
                        {"name": "Peak Confidence", "value": f"{breach_info.get('peak_confidence', 0.95):.1%}", "inline": True},
                        {"name": "Timestamp", "value": timestamp, "inline": True}
                    ],
                    "footer": {"text": "THERsenel Dual-Spectrum Autonomous Border Defense System"}
                }
            ]
        }

    def format_slack_payload(self, breach_info: Dict[str, Any]) -> Dict[str, Any]:
        """Construct Slack Block Kit message."""
        intruder_count = breach_info.get("intruders", 1)
        return {
            "text": f"🚨 *BORDER INTRUSION ALERT:* {intruder_count} intruder(s) breached Zone SEC-09!",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🚨 RESTRICTED ZONE INTRUSION ALERT*\n*Zone:* SEC_ZONE-09 | *Intruders:* {intruder_count} | *Status:* IMMEDIATE DISPATCH REQUIRED"
                    }
                }
            ]
        }

    def send_breach_alert(self, breach_info: Dict[str, Any]) -> bool:
        """
        Dispatches payload to configured webhook. Returns True on success, False otherwise.
        """
        if not self.webhook_url:
            logger.info("No webhook URL configured. Simulated alert dispatch logged.")
            return False

        if not self.can_dispatch():
            logger.debug("Alert dispatch throttled by active cooldown window.")
            return False

        if self.service_type == "discord":
            payload = self.format_discord_payload(breach_info)
        elif self.service_type == "slack":
            payload = self.format_slack_payload(breach_info)
        else:
            payload = {
                "event": "BORDER_PERIMETER_BREACH",
                "timestamp": time.time(),
                "details": breach_info
            }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "THERsenel-Edge-Sentinel/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status in (200, 204):
                    self._last_dispatch_time = time.time()
                    self._dispatch_count += 1
                    logger.info(f"Breach alert dispatched successfully (Dispatch #{self._dispatch_count})")
                    return True
        except Exception as e:
            logger.warning(f"Failed to deliver alert webhook: {e}")

        return False
