import requests
import json
import base64

img_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
b64 = "data:image/png;base64," + img_data.decode("utf-8")
payload = {
    "image": b64,
    "audio_features": [0.5] * 40
}
res = requests.post("http://127.0.0.1:8888/api/detect-lie", json=payload)
print(res.status_code, res.text)