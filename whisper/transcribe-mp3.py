#!/usr/bin/python3
"""
Transcribe MP3 audio files to text

Python script that uses the OpenAI Whisper model to transcribe
audio files to text.

Requirements:
   pip install git+https://github.com/openai/whisper.git -q

Author: Jason A. Cox
10 Sept 2023
https://github.com/jasonacox/ProtosAI

"""# Requires
#   pip install git+https://github.com/openai/whisper.git -q

import os
import whisper

def list_files_with_keyword(directory, keyword, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if keyword in file and file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

# Specify the directory, keyword, and extension
directory = '.'  # Replace with the actual directory path
keyword = 'audio'
extension = '.mp3'

# List the files matching the criteria
matching_files = list_files_with_keyword(directory, keyword, extension)

# Print the matching file paths
mp3=[]
total =0 
uniq = 0
valid = 0
for file_path in matching_files:
    # Extract the desired part of the filename
    soundfile = file_path.split(" - ")[0] + ".mp3"
    total += 1
    if soundfile not in mp3:
        uniq += 1
        if os.path.exists(soundfile):
            mp3.append(soundfile)
            valid += 1

mp3.sort()
print(f"Found mp3 files in {directory}: {total=} {uniq=} {valid=}")

# Load the whisper model
model = whisper.load_model('large')

# Transcribe each mp3
for message in mp3:
    fn = os.path.basename(message)
    cn = os.path.splitext(fn)[0]
    print(f"Transcribing {message} {cn=}")

    # Transcribe the audio
    output = model.transcribe(message)
    transcription_text = output['text']

    # Write the transcription text to a file
    output_filename = f"{cn}.txt"
    with open(output_filename, 'w') as file:
        file.write(transcription_text)

    print(f"Transcription {cn} saved to {output_filename}")