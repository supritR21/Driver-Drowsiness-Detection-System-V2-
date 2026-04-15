import base64
import requests

# Load any image (face image preferred)
with open("test.jpg", "rb") as f:
    img_bytes = f.read()

# Convert to base64
img_base64 = base64.b64encode(img_bytes).decode("utf-8")

# Send request
response = requests.post(
    "http://127.0.0.1:8000/api/v1/inference/frame",
    json={
        "session_id": "demo-session",
        "frame_base64": img_base64
    }
)

print(response.json())