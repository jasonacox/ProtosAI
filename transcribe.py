"""
Transcribe Audio File into Text

Author: Jason A. Cox
24 May 2023
https://github.com/jasonacox/ProtosAI

This uses a HuggingFace transformer https://huggingface.co/

"""
import sys
from transformers import pipeline

def transcribe(filename):
    print("\nLoading model...")
    pipe = pipeline(model="facebook/wav2vec2-base-960h")
    print(f"\nTranscribing {filename}...")
    transcript = pipe(filename, chunk_length_s=10)
    print(transcript['text'])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <filename.wav>")
        sys.exit(1)

    file_path = sys.argv[1]
    transcribe(file_path)
