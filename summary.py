#!/usr/bin/python3
"""
Summarizer for Text File

Author: Jason A. Cox
23 May 2023
https://github.com/jasonacox/ProtosAI

This uses the HuggingFace transformer https://huggingface.co/

"""
# import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import transformers
transformers.logging.set_verbosity_error()
import sys

# title
print("Summarizer")

# load models
print("\nLoading transformer...")
model_id = "sshleifer/distilbart-cnn-12-6"
print(f" * {model_id}")
summarizer = transformers.pipeline("summarization", model=model_id)

# read text file
def read_text_file(file_path):
    try:
        print(f"\nReading {file_path}...")
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied to open file '{file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An error occurred while reading file '{file_path}':")
        print(e)
        sys.exit(1)

# print output
def print_summary(text):
    num_lines = len(text.split('\n'))
    num_words = len(text.split())
    num_characters = len(text)
    
    print(f"Number of lines: {num_lines}")
    print(f"Number of words: {num_words}")
    print(f"Number of characters: {num_characters}")

    print("\nSummarizing...")
    summary = summarizer(text)
    text = summary[0]['summary_text']
    print(f"Text: {text}")
    num_lines = len(text.split('\n'))
    num_words = len(text.split())
    num_characters = len(text)
    print(f"Number of lines: {num_lines}")
    print(f"Number of words: {num_words}")
    print(f"Number of characters: {num_characters}")


# main
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Please provide the path to a text file as a command-line argument.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    text = read_text_file(file_path)
    print_summary(text)


