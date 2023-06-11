#!/usr/bin/python3
"""
Clean Raw Text 

Simple script to remove special characters from raw text file to help
with training LLMs.

Author: Jason A. Cox
11 June 2023
https://github.com/jasonacox/ProtosAI

"""
import sys
import string

def clean_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()

    # Fix bad quotes    
    translation_table = str.maketrans("’‘‛“”«»„", "'''\"\"\"\"\"")
    text = text.translate(translation_table)
    
    # Remove non-ASCII characters
    cleaned_text = ''.join(char for char in text if char in string.printable)

    # Additional preprocessing if needed
    # cleaned_text = cleaned_text.replace('...', '...')  # Example replacement

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

    print("Text file cleaned and saved as", output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        clean_text(input_file, output_file)

