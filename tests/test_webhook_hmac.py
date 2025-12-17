import os
import json
import time
import hmac
import hashlib

# Ensure the receiver module picks up the secret at import time
os.environ['WEBHOOK_SECRET'] = 'test-secret'

from fastapi.testclient import TestClient
from scripts.webhook_receiver_example import app

client = TestClient(app)


def make_payload():
    return {"job_id": "abc", "status": "finished", "result": {"value": 1}}


def sign_payload(payload_bytes: bytes, ts: str, secret: str) -> str:
    return hmac.new(secret.encode(), ts.encode() + b'.' + payload_bytes, hashlib.sha256).hexdigest()


def test_valid_signature():
    payload_bytes = json.dumps(make_payload(), separators=(',', ':'), sort_keys=True).encode()
    ts = str(int(time.time()))
    sig = sign_payload(payload_bytes, ts, os.environ['WEBHOOK_SECRET'])
    headers = {'X-Signature': f'sha256={sig}', 'X-Signature-Timestamp': ts}

    resp = client.post('/webhook', content=payload_bytes, headers=headers)
    assert resp.status_code == 200


def test_invalid_signature():
    payload_bytes = json.dumps(make_payload(), separators=(',', ':'), sort_keys=True).encode()
    ts = str(int(time.time()))
    # wrong signature
    headers = {'X-Signature': 'sha256=' + '0' * 64, 'X-Signature-Timestamp': ts}

    resp = client.post('/webhook', content=payload_bytes, headers=headers)
    assert resp.status_code == 401


def test_old_timestamp():
    payload_bytes = json.dumps(make_payload(), separators=(',', ':'), sort_keys=True).encode()
    # timestamp far in the past
    ts = str(int(time.time()) - 10000)
    sig = sign_payload(payload_bytes, ts, os.environ['WEBHOOK_SECRET'])
    headers = {'X-Signature': f'sha256={sig}', 'X-Signature-Timestamp': ts}

    resp = client.post('/webhook', content=payload_bytes, headers=headers)
    assert resp.status_code == 400
