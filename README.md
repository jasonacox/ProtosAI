# ProtosAI

A Study in Artificial Intelligence

This project consist of a collection of scripts that explore capabilities provided by neural networks pre-trained transformers. These scripts are based on the Hugging Face (https://huggingface.co/) models.  

## Setup

Setup required for these scripts:

```bash
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

```text
$ python3 transcribe.py test.wav

Loading model...

Transcribing test.wav...
HELLO THIS IS A TEST
```

## Text to Speech

The `speech.py` script converts a text string into an audio file.  A sample output is [output.wav](output.wav). The script requires additional libraries:

```bash
pip install espnet torchaudio sentencepiece
```

```text
$ python3 speech.py

Loading models...

Converting text to speech...

Writing to audio.wav...

Speaking: Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away.
```
