#!/usr/bin/python3
"""
Transcribe audio files to text with timestamps

Python script that uses the OpenAI Whisper model to transcribe
audio files to text.

CLI Usage:
    python3 transcribe.py audio.mp3

Requirements:
   pip install git+https://github.com/openai/whisper.git

Author: Jason A. Cox
13 July 2024
https://github.com/jasonacox/ProtosAI

"""
import sys
import os
import json
import whisper

MODEL = 'small'

if __name__ == '__main__':
    # Header
    print(f"{whisper.__name__.capitalize()} v{whisper.__version__} ({sys.argv[0]})",
          "- Transcribe audio to text with timestamps")
    print()
    if len(sys.argv) < 2:
        # Print usage and exit
        print(f"Usage: {sys.argv[0]} [audio_file] [model]")
        print()
        print("Arguments:")
        print("  audio_file: Path to the audio file to transcribe (wav, mp3, etc)")
        print(f"  model:      Whisper model to use (default: {MODEL})")
        models = whisper.available_models()
        line = "              Models: "
        for m in models:
            if len(line + m) > 80:
                print(line)
                line = " " * 22
            line += m + ", "
        print(line[:-2])
        print()
        print("Outputs:\n",
              "  [audio_file].raw:  Raw transcription text\n",
              "  [audio_file].json: JSON formatted transcription\n",
              "  [audio_file].txt:  Transcript with timestamps\n")
        sys.exit(1)

    if len(sys.argv) > 2:
        MODEL = sys.argv[2]

    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    # Load the whisper model
    print(f" - Loading the {MODEL} Whisper model...")
    model = whisper.load_model('large')

    # Transcribe the audio file
    print(f" - Transcribing {audio_file}...")
    data = model.transcribe(audio_file, verbose=True, word_timestamps=True)
    print(f"Transcription complete.")
    print()
    print("Writing transcription to files...")

    # Write raw text to file
    text_file = audio_file + ".raw"
    with open(text_file, 'w') as f:
        f.write(data['text'])
    print(f" - Raw transcription saved to {text_file}")

    # Convert to JSON and write to file
    json_file = audio_file + ".json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f" - Transcription saved to {json_file}")

    # Create a transcript file with timestamps
    transcript_file = audio_file + ".txt"
    with open(transcript_file, 'w') as f:
        for s in data['segments']:
            start = f"{int(s['start']//3600)}:{(int(s['start']%3600)//60):02d}:{int(s['start']%60):02d}"
            end = f"{int(s['end']//3600)}:{(int(s['end']%3600)//60):02d}:{int(s['end']%60):02d}"
            f.write(f"{start} - {end}: {s['text']}\n")
    print(f" - Transcript with timestamps saved to {transcript_file}")
