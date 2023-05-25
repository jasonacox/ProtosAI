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

# Input
say = "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."
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
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    filename = 'audio.wav'  # Change this to your desired filename
   

# Load models
print("\nLoading models...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

print("\nConverting text to speech...")
inputs = processor(text=say, return_tensors="pt")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
"""
spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

with torch.no_grad():
    speech = vocoder(spectrogram)
"""
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

print(f"\nWriting to {filename}...")
sf.write(filename, speech.numpy(), samplerate=16000)

print(f"\nSpeaking: {say}")
play_audio(filename)
