#!/usr/bin/python3
"""
Transcribe YouTube videos into text

Python script that uses the OpenAI Whisper model to transcribe
YouTube links to text. The uses the pytube library.

Requirements:
   pip install git+https://github.com/openai/whisper.git -q
   pip install pytube

Author: Jason A. Cox
10 Sept 2023
https://github.com/jasonacox/ProtosAI

"""
print("import whisper")
import whisper
model = whisper.load_model('large')
# Available models: tiny.en, tiny, base.en, base, small.en, small, 
#     s              medium.en, medium, large-v1, large

print("import pytube")
from pytube import YouTube
youtube_video_url = "https://www.youtube.com/watch?v=M0PFDJ11Txg"
youtube_video = YouTube(youtube_video_url)

print(f"Title: {youtube_video.title}")

# grab first audio stream
print("grab audios stream and save it")
streams = youtube_video.streams.filter(only_audio=True)
stream = streams.first()
stream.download(filename='test.mp4')

# do the transcription
print("transcribe...")
output = model.transcribe('test.mp4')

print(output['text'])
