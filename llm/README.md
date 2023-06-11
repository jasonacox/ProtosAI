# LLM Examples

## Training

This experiment uses the introductory training model based on the lecture and work by Andrej Karpathy and his nanoGPT project (https://github.com/karpathy/nanoGPT). I adjusted the training model to use the GPT tokenization method (tiktoken) for the word embedding.

I took the raw text from my blog (jasonacox.com) and used that as the training set. I used an M1 iMac and set the device to `mps` (Apple Silicon Metal Performance Shaders) for the PyTorch settings.  Using 50,000 iterations, it ran for several hours and produced an output of random musing. While there was quite a bit of nonsensical output, I was amazed at how well this small run did at learning basic sentance structure and even picked up on my style. Here are some samples from the output I found entertaining, comical and spot on:

* It’s a lot of time… But I think we also need science. I’ve want to do what matters.
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
* Have you found a vulnerability?  Are you concerned about some missing measures or designs that should be modernized or addressed?  If so, don’t wait, raise those issues.  Speak up and act.  You can make a difference.
* “I know what you are thinking.” the irony
* We are the ones who make a brighter day, so let’s start giving.
* The journey ahead is ahead.
* What are you penning today? What adventures are you crafting by your doing? Get up, get moving… keep writing.
* The kids used the entire space to explore get the power of joy.  Be on the lookout beautiful.

The raw input was small (473k) and a bit messy.  It had some random code and maker project details that should be cleaned up. But overall, I'm impressed with the results. Next step is to see if I can figure out finetuning and how to handle prompting (prompt encoding).