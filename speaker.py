import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print(tts.speakers)


# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")

tts.tts_to_file(
    text="Привет, Мир!",
    speaker_wav="./speaker.wav",
    language="ru",
    file_path="output.mp3",
)
