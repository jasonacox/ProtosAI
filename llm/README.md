# LLM Examples

The goal of this section is to explore the different LLM models, specifically related to building, training, tuning and using these models.

## LLaMA Models

The [llama.cpp project's](https://github.com/ggerganov/llama.cpp) goal is to run LLaMA models using integer quantization to allow the use of these LLMs on small scale computers like a MacBook.

In the folder [llama.cpp](https://github.com/jasonacox/ProtosAI/tree/master/llm/llama.cpp) we explore the use of the 7B Llama-2 model using llama.cpp (C/C++) and llama-cpp-python (python) projects.

* Building an locally hosted LLM OpenAI API compatible server
* Building a python based CLI chatbot that keeps conversational context
* Build for MacOS (with or without Metal acceleration) and Linux with a Nvidia GPU (CUDA)
* Train the model on user supplied documents

## Bigram Model

This experiment uses the introductory training model based on the lecture and work by Andrej Karpathy and his nanoGPT project (https://github.com/karpathy/nanoGPT). I adjusted the training model to use the GPT tokenization method (tiktoken) for word embedding. It uses a Bigram Language model which simply looks at previous word to determine the next.

I took the [raw text](https://github.com/jasonacox/ProtosAI/files/11715802/input.txt) (468K) from my blog ([jasonacox.com](https://www.jasonacox.com/)) and used that as the training set. I created a simple [clean.py](clean.py) script to remove any special characters from the text. Since the training would run on my M1 iMac, I edited [bigram.py](bigram.py) and added `device = 'mps'` to set it to use the MPS (Apple Silicon Metal Performance Shaders) for the PyTorch acceleration.  

### Code

```bash
# grab text
wget https://github.com/jasonacox/ProtosAI/files/11715163/jason.txt 

# clean the input
python clean.py jason.txt input.txt

# run model
python bigram.py
```

The training ran for 50,000 iterations and took about 8 hours. It produced an output of random musing. While there was quite a bit of nonsensical output, I was amazed at how well this small run did at learning words, basic sentence structure and even picked up on my style. Here are some samples from the output I found interesting, comical and sometimes, spot on:

### Example Output

* It’s a lot of time… But I think we also need science.
* What are your big ideas?  
* Set our management to the adjacent ground (GND) pin.
* I have a task to Disneyland out that this day.
* I love the fun and fanciful moments as kids get to dream into their favorite characters, embrace the identity of their heroes, wrap themselves up starfish back.
* Bring on the “power” of his accidentally detail.
* Your character provided faith, all kindness and don’t care.
* Grab a difference too.
* After several days of emailing, texting and calling, I received a text message.
* Curl has the ability to provide timing data for DNS lookup, it will easily show or avoided.
* Imperfect things with a positive ingredient can become a positive difference, just get that time.
* I also believe we should exploit the fusion power that shows up each day in our company’s data.
* Have you found a vulnerability? Are you concerned about some missing measures or designs that should be modernized or addressed? If so, don’t wait, raise those issues. Speak up and act. You can make a difference.
* "I know what you are thinking." the irony
* We are the ones who make a brighter day.
* The journey ahead is ahead.
* What are you penning today? What adventures are you crafting by your doing? Get up, get moving… keep writing.

### Summary

The raw input was small (468K) and a bit messy.  It had some random code and maker project details that should be cleaned up. But overall, I'm impressed with the results. Next step is to see if I can figure out fine-tuning and how to handle prompting (prompt encoding).

## nanoGPT Model

I have included a fork of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) here to help with the next example. In this example, similar to the Bigram test above, I will use the [raw text](https://github.com/jasonacox/ProtosAI/files/11715802/input.txt) (468K) from my blog (jasonacox.com) and used that as the training set.

The first step was to prepare the input.

```bash
# tokenize the raw text for the model
cd data/jasonacox
wget https://github.com/jasonacox/ProtosAI/files/11715802/input.txt
python prepare.py
cd ..

# run the training (make sure to time it)
time python3 train.py \
    --dataset=jasonacox \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=64 \
    --compile=False \
    --eval_iters=1 \
    --block_size=64 \
    --batch_size=8 \
    --device=mps # for Apple Silicon or change to cpu or cuda
```

## References

* Video: Let's build GPT: from scratch, in code, spelled out by Andrej Karpathy - https://youtu.be/kCc8FmEb1nY
* nanoGPT repo: https://github.com/karpathy/nanoGPT
* Video: Building makemore by Andrej Karpathy - https://youtu.be/PaCmpygFfXo
* Running nanoGPT on a MacBook M2 to generate terrible Shakespeare
https://til.simonwillison.net/llms/nanogpt-shakespeare-m2
