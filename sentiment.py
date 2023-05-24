"""
Provide Sentiment Based on Input

Author: Jason A. Cox
23 May 2023
https://github.com/jasonacox/ProtosAI

This uses the HuggingFace transformer https://huggingface.co/

"""

# import
from transformers import pipeline

# load models
print("Loading transformer...")
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = pipeline("sentiment-analysis", model=model_id)

# star loop
user_input = "start"
while len(user_input) > 0:
    user_input = input("\nEnter some text (or empty to end): ")
    sentiment_score = classifier(user_input)
    print("Sentiment score:", sentiment_score)



