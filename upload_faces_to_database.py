import base64
import requests

def upload_face_payload(name: str, image_path: str):
    url = "http://localhost:8080/api/uploadFacePayload"

    with open(image_path, 'rb') as file:
        b64_img = base64.b64encode(file.read()).decode('utf-8')

    payload = {
        "name": name,
        "photoBase64": b64_img
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print(f"✅ Uploaded face for '{name}'")
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")

# usage
upload_face_payload("Lucas", "lucas.jpg")
