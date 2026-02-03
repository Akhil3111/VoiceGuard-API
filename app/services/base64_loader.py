import base64
import tempfile
import os

class Base64AudioError(Exception):
    pass

def decode_base64_audio(audio_base64: str, suffix=".mp3") -> str:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            return f.name
    except Exception:
        raise Base64AudioError("Invalid Base64 audio input")
