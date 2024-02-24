# Gemma

Google has released an open source LLM model, Gemma. It is built for responsible AI development from the same research and technology used to create Gemini models.

https://github.com/google/gemma.cpp

## Gemma.cpp

The open source project [gemma.cpp](https://github.com/google/gemma.cpp) will run the Gemma LLM on consumer grade hardware, include Apple Silicon. 

## Build and Run

### Step 1 - Git the Project

```bash
# Git the project
git clone https://github.com/google/gemma.cpp.git
cd gemma.cpp
```

### Step 2 - Download the models. I recommend testing both the 2B (first) and 7B instruct models.

* https://www.kaggle.com/models/google/gemma/frameworks/gemmaCpp/variations/2b-it-sfp
* https://www.kaggle.com/models/google/gemma/frameworks/gemmaCpp/variations/7b-it-sfp

```bash
# Extract model
mkdir 2b-it
cd 2b-it
tar -xvf ~/Downloads/archive.tar.gz
cd ..
```

### Step 3 - Build the gemma.cpp project

```bash
# Install cmake if you don't have it already
brew install cmake

# Run CMake
cmake -B build

# Build
cd build
make -j 8 gemma
cd ..

# Run the 2B model
./build/gemma --tokenizer 2b-it/tokenizer.spm --compressed_weights 2b-it/2b-it-sfp.sbs --model 2b-it    

# Run the 7B model
./build/gemma --tokenizer 7b-it/tokenizer.spm --compressed_weights 7b-it/7b-it-sfp.sbs --model 7b-it  
```

### Example Output

```yml
  __ _  ___ _ __ ___  _ __ ___   __ _   ___ _ __  _ __
 / _` |/ _ \ '_ ` _ \| '_ ` _ \ / _` | / __| '_ \| '_ \
| (_| |  __/ | | | | | | | | | | (_| || (__| |_) | |_) |
 \__, |\___|_| |_| |_|_| |_| |_|\__,_(_)___| .__/| .__/
  __/ |                                    | |   | |
 |___/                                     |_|   |_|

tokenizer                     : 2b-it/tokenizer.spm
compressed_weights            : 2b-it/2b-it-sfp.sbs
model                         : 2b-it
weights                       : [no path specified]
max_tokens                    : 3072
max_generated_tokens          : 2048

*Usage*
  Enter an instruction and press enter (%Q quits).

*Examples*
  - Write an email to grandma thanking her for the cookies.
  - What are some historical attractions to visit around Massachusetts?
  - Compute the nth fibonacci number in javascript.
  - Write a standup comedy bit about GPU programming.

> What is an LLM?

[ Reading prompt ] ..............


An LLM (Large Language Model) is a type of artificial intelligence (AI) that has been trained on a massive dataset of text and code. It is a powerful language model that can perform a wide range of tasks, including:

* Natural language processing (NLP)
* Language translation
* Text generation
* Question answering
* Sentiment analysis
* Summarization
* And more

LLMs are different from traditional AI models in that they are trained on a massive dataset of text and code, rather than on a set of images or other data. This allows them to learn a much broader range of information and to perform a wider range of tasks.

LLMs are still under development, but they have the potential to revolutionize the way we interact with computers. They could be used to create more natural and human-like chatbots, to develop new forms of language learning, and to automate a wide range of tasks.

> 
```