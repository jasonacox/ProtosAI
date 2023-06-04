# ProtosAI

A Study in Artificial Intelligence

This project consist of a collection of scripts that explore capabilities provided by neural networks pre-trained transformers. These scripts are based on the Hugging Face (https://huggingface.co/) models.  

## Setup

Setup required for these scripts:

```bash
# Requirements
pip install transformers datasets
pip install torch
```

Note that during the fist run, the library will download the required model to process the inputs.

## Sentiment Analysis

The `sentiment.py` script prompts the user for a line of text and uses a model to determine the sentiment of the text (positive, neutral or negative).

```text
Enter some text (or empty to end): I love you.
Sentiment score: [{'label': 'positive', 'score': 0.9286843538284302}]

Enter some text (or empty to end): I am sad.
Sentiment score: [{'label': 'negative', 'score': 0.7978498935699463}]

Enter some text (or empty to end): I hate dirty pots.
Sentiment score: [{'label': 'negative', 'score': 0.9309694170951843}]

Enter some text (or empty to end): Don't move!
Sentiment score: [{'label': 'neutral', 'score': 0.6040788292884827}]
```

## Summarization

The `summary.py` script takes a text file input and uses the summarization model to produce a single paragraph summary.

```text
$ python3 summary.py pottery.txt                                     
Loading transformer...

Reading pottery.txt...
Number of lines: 14
Number of words: 566
Number of characters: 3416

Summarizing...
Text:  The key to becoming a great artist, writer, musician, etc., is to keep creating!
Keep drawing, keep writing, keep playing! Quality emerges from the quantity of practice
and continuous learning that makes them more perfect . The prize of perfection comes by
delivering and learning, says Jason Cox .
Number of lines: 1
Number of words: 49
Number of characters: 299
```

## Transcribe

The `transcribe.py` script takes an audio file (mp3 or wav file) and uses a speech model to produce a basic text transcription. A additional tool `record.py` will use your laptops microphone to record your dictation into `audio.wav` that can be used by `transcribe.py`.

```bash
# Requirements
brew install ffmpeg           
```

```text
$ python3 transcribe.py test.wav

Loading model...

Transcribing test.wav...
HELLO THIS IS A TEST
```

## Text to Speech

The `speech.py` script converts a text string into an audio file.  The script requires additional libraries:

```bash
# Requirements
brew install portaudio  # MacOS requires portaudio
pip install espnet torchaudio sentencepiece pyaudio
```

```text
$ python3 speech.py

Loading models...

Converting text to speech...

Writing to audio.wav...

Speaking: Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away.
```

https://github.com/jasonacox/ProtosAI/assets/836718/56759007-90b6-4d94-83f2-3cc3cb78ccbe


## Handwriting to Text

The `handwriting.py` script converts an image of a handwritten single line of text to a string of text.

```bash
# Requirements
pip install image
```

![test.png](test.png)

```text
$ python3 handwriting.py test.png
Converting image to text: test.png

Loading transformer...
 * microsoft/trocr-base-handwritten

Analyzing handwriting from test.png...

Resulting text:
This is a test-Can you read this?
```

## OpenAI Test

The openai.py script prompts the OpenAI gpt-3.5 model and prints the response.

```bash
# Requirements
pip install openai

# Test
$ python3 gpt.py
What do you want to ask? Can you say something to inspire engineers?

Answer: {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Of course! Here's a quote to inspire engineers:\n\n\"Engineering is not only about creating solutions, it's about creating a better world. Every time you solve a problem, you make the world a little bit better.\" - Unknown\n\nAs an engineer, you have the power to make a positive impact on society through your work. Whether you're designing new technologies, improving existing systems, or solving complex problems, your contributions are essential to advancing our world. So keep pushing the boundaries of what's possible, and never forget the impact that your work can have on the world around you.",
        "role": "assistant"
      }
    }
  ],
  "created": 1685856679,
  "id": "chatcmpl-7Nach0z2sJQ5FzZOVl6jZWPU4O6zV",
  "model": "gpt-3.5-turbo-0301",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 117,
    "prompt_tokens": 26,
    "total_tokens": 143
  }
}
```

## GPT-2 Text Generation

The `gpt-2.py` script uses the gpt2-xl model to generate test based on a prompt.

```bash
$ python3 gpt-2.py   
```

```json
[{'generated_text': "Hello, I'm a language model, but what I do you need to know isn't that hard. But if you want to understand us, you"}, {'generated_text': "Hello, I'm a language model, this is my first commit and I'd like to get some feedback to see if I understand this commit.\n"}, {'generated_text': "Hello, I'm a language model, and I'll guide you on your journey!\n\nLet's get to it.\n\nBefore we start"}, {'generated_text': 'Hello, I\'m a language model, not a developer." If everything you\'re learning about code is through books, you\'ll never get to know about'}, {'generated_text': 'Hello, I\'m a language model, please tell me what you think!" – I started out on this track, and now I am doing a lot'}]
```
