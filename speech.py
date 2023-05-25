"""
Convert Text to Speech

Author: Jason A. Cox
24 May 2023
https://github.com/jasonacox/ProtosAI

This uses a HuggingFace transformer https://huggingface.co/

"""
# imports
import soundfile as sf
from transformers import SpeechT5HifiGan, SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch
import pyaudio
import wave
import time
import sys

# Input
say1 = "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."
say2 = "I'm sorry, Dave, I'm afraid I can't do that."
say3 = "I fight for the users."

filename = "output.wav"

def play_audio(filename):
    chunk = 1024
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    time.sleep(1)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Title
print("Text to Speech")

# Load models
print("\nLoading models...")
print(" * microsoft/speecht5_tts")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
print(" * speecht5_hifigan")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

def speak(say):
    print("\nConverting text to speech...")
    inputs = processor(text=say, return_tensors="pt")
    print(" * Loading embeddings: Matthijs/cmu-arctic-xvectors")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    """
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

    with torch.no_grad():
        speech = vocoder(spectrogram)
    """

    print(" * Generating audio")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    print(f"\nPlaying {filename}...")
    sf.write(filename, speech.numpy(), samplerate=16000)

    print(f" * {say}")
    play_audio(filename)

# main
if __name__ == '__main__':
    if len(sys.argv) < 2:
        speak(say1)
        sys.exit(1)
    
    phrase = sys.argv[1]
    speak(phrase)
