import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2

from src.inference.api import app, ip_request_history, ip_lock

client = TestClient(app)


def create_test_image(width=100, height=100, color=(128, 128, 128)):
    """Generate a valid in-memory encoded image."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


class TestSecurityHeaders:
    """Verify Defense-in-Depth HTTP Security Headers."""

    def test_security_headers_present(self):
        response = client.get("/health")
        assert response.status_code == 200
        headers = response.headers

        assert headers.get("x-content-type-options") == "nosniff"
        assert headers.get("x-frame-options") == "SAMEORIGIN"
        assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
        assert headers.get("x-xss-protection") == "1; mode=block"
        assert "camera=()" in headers.get("permissions-policy", "")

    def test_cache_control_on_root(self):
        response = client.get("/")
        assert response.status_code == 200
        headers = response.headers
        assert "no-store" in headers.get("cache-control", "")
        assert headers.get("pragma") == "no-cache"


class TestHealthEndpoint:
    """Verify Service Liveness & Health Probe."""

    def test_health_check_payload(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ("healthy", "degraded")
        assert "uptime_seconds" in data
        assert "model_loaded" in data
        assert data["max_upload_mb"] == 15
        assert data["active_concurrency_limit"] == 4
        assert data["version"] == "1.0.0"


class TestInputValidationAndRejection:
    """Verify input sanitization, file caps, and error handling."""

    def test_reject_unauthorized_extension_and_mime(self):
        fake_executable = b"MZ\x90\x00\x03\x00\x00\x00"
        response = client.post(
            "/predict",
            files={"file": ("exploit.exe", io.BytesIO(fake_executable), "application/x-msdownload")},
            data={"conf_threshold": "0.5"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file format" in data.get("detail", "")

    def test_reject_empty_file(self):
        empty_bytes = b""
        response = client.post(
            "/predict",
            files={"file": ("empty.png", io.BytesIO(empty_bytes), "image/png")},
            data={"conf_threshold": "0.5"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data.get("detail", "").lower()

    def test_reject_corrupted_image_binary(self):
        junk_data = b"NOT_A_REAL_IMAGE_CORRUPTED_BYTES" * 32
        response = client.post(
            "/predict",
            files={"file": ("corrupt.png", io.BytesIO(junk_data), "image/png")},
            data={"conf_threshold": "0.5"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "Corrupted, unreadable" in data.get("detail", "")

    def test_reject_oversized_payload_content_length(self):
        # Pretend Content-Length header is 20MB
        oversized_headers = {"content-length": str(20 * 1024 * 1024)}
        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        response = client.post(
            "/predict",
            files={"file": ("big.png", io.BytesIO(fake_bytes), "image/png")},
            data={"conf_threshold": "0.5"},
            headers=oversized_headers
        )
        assert response.status_code == 413
        data = response.json()
        assert "exceeds maximum allowed limit" in data.get("detail", "")

    def test_reject_invalid_confidence_threshold_underflow(self):
        valid_img = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", io.BytesIO(valid_img), "image/png")},
            data={"conf_threshold": "-0.10"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data or "detail" in data

    def test_reject_invalid_confidence_threshold_overflow(self):
        valid_img = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", io.BytesIO(valid_img), "image/png")},
            data={"conf_threshold": "1.50"}
        )
        assert response.status_code == 422

    def test_reject_non_numeric_confidence_threshold(self):
        valid_img = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", io.BytesIO(valid_img), "image/png")},
            data={"conf_threshold": "MALICIOUS_INJECTION' OR '1'='1"}
        )
        assert response.status_code == 422


class TestRateLimiter:
    """Verify In-Memory Sliding-Window Rate Limiter."""

    def test_rate_limiter_triggers_429(self):
        # Reset IP history for test isolation
        ip_request_history.clear()

        valid_img = create_test_image()
        test_client_ip = "192.168.1.99"
        
        # We simulate 60 allowed requests, and the 61st must trigger 429
        # Inject 60 recent timestamps into ip_request_history for test_client_ip
        import time
        now = time.time()
        ip_request_history[test_client_ip] = [now - 1.0] * 60

        # Send request using testclient with custom client host
        response = client.post(
            "/predict",
            files={"file": ("test.png", io.BytesIO(valid_img), "image/png")},
            data={"conf_threshold": "0.5"},
            headers={"x-forwarded-for": test_client_ip}
        )
        # Note: TestClient default client.host is 'testclient'
        # Let's test by filling 'testclient' history:
        ip_request_history["testclient"] = [now - 1.0] * 60

        response = client.post(
            "/predict",
            files={"file": ("test.png", io.BytesIO(valid_img), "image/png")},
            data={"conf_threshold": "0.5"}
        )
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json().get("detail", "")
        assert response.headers.get("retry-after") == "60"

        # Cleanup
        ip_request_history.clear()


class TestValidInferenceExecution:
    """Verify that a legitimate request executes cleanly and returns proper telemetry headers."""

    def test_valid_image_inference_success(self):
        ip_request_history.clear()
        valid_img = create_test_image(width=320, height=240)

        response = client.post(
            "/predict",
            files={"file": ("border_feed.png", io.BytesIO(valid_img), "image/png")},
            data={"conf_threshold": "0.45"}
        )

        assert response.status_code == 200
        assert response.headers.get("content-type") == "image/jpeg"
        assert "x-inference-latency-ms" in response.headers
        assert "x-inference-fps" in response.headers
        assert "x-alerts-count" in response.headers
        assert "x-detections-count" in response.headers
        assert len(response.content) > 0
