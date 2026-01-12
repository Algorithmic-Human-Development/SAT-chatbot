from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)


def feed_audio_to_ASR_modal(audio_path):
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
    )
    audio_file = open(audio_path, "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    return transcription.text
