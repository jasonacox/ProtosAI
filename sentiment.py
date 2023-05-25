#!/usr/bin/python3
"""
Provide Sentiment Based on Input

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

# load models
print("Loading transformer...")
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f" * {model_id}")
classifier = transformers.pipeline("sentiment-analysis", model=model_id)

# start loop
while True:
    user_input = input("\nEnter some text (or empty to end): ")
    if len(user_input) < 1:
        break
    sentiment_score = classifier(user_input)
    print("Sentiment score:", sentiment_score)



