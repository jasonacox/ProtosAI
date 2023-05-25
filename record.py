"""
Record Audio File 

Author: Jason A. Cox
24 May 2023
https://github.com/jasonacox/ProtosAI

Specify 

"""

import pyaudio
import wave
import sys

def record_audio(filename, duration):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print("\nRecording started. Press Ctrl+C to stop recording...")

    frames = []

    try:
        for i in range(0, int(fs / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <filename.wav> <seconds>")
        sys.exit(1)
    duration = 5 # max number of seconds to record
    if sys.argv > 2:
        duration = int(sys.argv[2])
    file_path = sys.argv[1]
    print(f"Record Audio File: {file_path}\n")
    record_audio(file_path, duration)
