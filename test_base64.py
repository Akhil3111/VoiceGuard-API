import requests
import base64
import json
import io
import wave

# 1. Configuration
API_URL = "http://localhost:8000/api/v1/analyze"
API_KEY = "hackathon-secret-key-123"

def create_dummy_audio():
    """Generates a small valid WAV file in memory if download fails."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1) # Mono
        wav_file.setsampwidth(2) # 2 bytes
        wav_file.setframerate(16000)
        # Write 1 second of silence (enough to pass the >0.5s check)
        wav_file.writeframes(b'\x00' * 32000)
    return buffer.getvalue()

# 2. Get a sample audio file
print("⬇️  Downloading sample audio...")
sample_url = "https://upload.wikimedia.org/wikipedia/commons/4/40/En-wikipedia.ogg"

# FIX: Add a User-Agent header to prevent 403 Forbidden errors
headers_download = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

try:
    response = requests.get(sample_url, headers=headers_download, timeout=5)
    if response.status_code == 200:
        audio_content = response.content
        print("   ✅ Download successful.")
    else:
        print(f"   ⚠️ Download failed (Status {response.status_code}). Using dummy audio instead.")
        audio_content = create_dummy_audio()
except Exception as e:
    print(f"   ⚠️ Network error: {e}. Using dummy audio instead.")
    audio_content = create_dummy_audio()

# 3. Convert to Base64
print("🔄 Converting to Base64...")
b64_string = base64.b64encode(audio_content).decode('utf-8')

# 4. Prepare Payload
payload = {
    "language": "English",
    "audio_format": "ogg" if response.status_code == 200 else "wav", 
    "audio_base64": b64_string
}

headers_api = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

# 5. Send Request
print(f"🚀 Sending request to {API_URL}...")
try:
    api_response = requests.post(API_URL, json=payload, headers=headers_api)
    
    print(f"\nStatus Code: {api_response.status_code}")
    print("Response Body:")
    print(json.dumps(api_response.json(), indent=2))
    
    if api_response.status_code == 200:
        print("\n✅ SUCCESS: API accepted Base64 input!")
    else:
        print("\n❌ FAILED: API rejected the request.")

except requests.exceptions.ConnectionError:
    print("\n❌ FAILED: Could not connect to localhost. Is your uvicorn server running?")