from fastapi import FastAPI, Request, HTTPException
import os
import hmac
import hashlib
import time
from hmac import compare_digest

app = FastAPI()

WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '')

@app.post('/webhook')
async def webhook_receiver(request: Request):
    body = await request.body()
    sig_hdr = request.headers.get('X-Signature', '')
    ts = request.headers.get('X-Signature-Timestamp', '')

    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail='Receiver not configured with WEBHOOK_SECRET')

    # basic replay protection: allow 5 minutes
    try:
        if abs(time.time() - float(ts)) > 300:
            raise HTTPException(status_code=400, detail='Timestamp outside allowed window')
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid timestamp')

    if not sig_hdr.startswith('sha256='):
        raise HTTPException(status_code=400, detail='Invalid signature format')

    received = sig_hdr.split('=', 1)[1]
    expected = hmac.new(WEBHOOK_SECRET.encode(), ts.encode() + b'.' + body, hashlib.sha256).hexdigest()

    if not compare_digest(expected, received):
        raise HTTPException(status_code=401, detail='Invalid signature')

    # at this point the webhook is verified
    return {'status': 'ok'}
