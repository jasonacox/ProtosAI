# Whisper

OpenAI Whisper is a speech-to-text model that uses machine learning to transcribe audio into text. It's trained on a large amount of English audio and text, as well as supervised data from the web, making it robust and able to generalize to a variety of languages and tasks. Whisper can also translate speech to English. 

## Setup

To use the Whisper model you need to install ffmpeg and the Whisper library.

```bash
# MacOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Python Library
pip3 install git+https://github.com/openai/whisper.git
```

# Transcribe Audio with Timestamps

To transcribe an audio.mp3 file, you would do the following:


```bash
python3 transcribe.py audio.mp3
```

This creates three file: JSON and RAW file with timestamps and a plain text only TXT file. You can also use the whisper library directly.

```python
import whisper

# Load the whisper weights
model = whisper.load_model('large')

# Transcribe with Timestamps
output = model.transcribe('audio.mp3', erbose=True, word_timestamps=True)

# Print raw text
print(output['text'])

# Print individual timestamp segments
for s in output['segments']:
    start = f"{int(s['start']//3600)}:{(int(s['start']%3600)//60):02d}:{int(s['start']%60):02d}"
    end = f"{int(s['end']//3600)}:{(int(s['end']%3600)//60):02d}:{int(s['end']%60):02d}"
    print(f"{start} - {end}: {s['text']}\n")
```