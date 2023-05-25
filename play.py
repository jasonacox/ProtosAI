#!/usr/bin/python3
"""
Play an Audio File

Author: Jason A. Cox
24 May 2023
https://github.com/jasonacox/ProtosAI

"""
# imports
import pyaudio
import wave
import sys
import time

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

# main
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Please provide the path to a audio file to play.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"\nPlaying: {file_path}")
    play_audio(file_path)
