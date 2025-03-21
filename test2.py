from pyannote.audio.pipelines import SpeakerDiarization
from dotenv import load_dotenv
import os

load_dotenv()
print("line 6")
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("line 8")
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
print("line 10")
diarization = pipeline("history.wav")
print("line 12")
print(diarization)
diarization_result = diarization_pipeline(file_path)
print("line 15")
