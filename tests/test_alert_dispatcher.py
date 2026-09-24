import pytest
from src.intrusion.alert_dispatcher import WebhookAlertDispatcher

def test_webhook_dispatcher_cooldown():
    dispatcher = WebhookAlertDispatcher(cooldown_seconds=10.0)
    assert dispatcher.can_dispatch() is True
    dispatcher._last_dispatch_time = 1000.0
    # Simulate past time
    import time
    dispatcher._last_dispatch_time = time.time()
    assert dispatcher.can_dispatch() is False

def test_discord_payload_structure():
    dispatcher = WebhookAlertDispatcher(service_type="discord")
    payload = dispatcher.format_discord_payload({
        "intruders": 2,
        "latency_ms": 34.2,
        "classes": ["person"],
        "peak_confidence": 0.88
    })
    assert "embeds" in payload
    assert len(payload["embeds"]) == 1
    embed = payload["embeds"][0]
    assert "RESTRICTED BORDER BREACH" in embed["title"]
    assert embed["color"] == 15158332

def test_slack_payload_structure():
    dispatcher = WebhookAlertDispatcher(service_type="slack")
    payload = dispatcher.format_slack_payload({"intruders": 1})
    assert "blocks" in payload
    assert "INTRUSION ALERT" in payload["text"]
