# Source: https://www.kaggle.com/code/tuckerarrants/text-generation-with-huggingface-gpt2/notebook
#for reproducibility, meaning that the same output can be yielded repeatedly
#changes frequently
SEED = 34

#maximum number of words in output
MAX_LEN = 50

#starting sequence used as basis
input_sequence = "There's a reason that my favorite planet is Neptune."

#importing transformers
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

#IF YOU WANT TO USE LARGE SIZED MODEL
#get large GPT2 tokenizer and GPT2 model
#splits the larger text into smaller, more digestible tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

#IF YOU WANT TO USE MEDIUM SIZED MODEL
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id)

#IF YOU WANT TO USE SMALL SIZED MODEL
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#view model parameters
GPT2.summary()

#get deep learning basics
import tensorflow as tf
tf.random.set_seed(SEED)

#first tried the greedy approach that picks highest probability - very naive
#then explored the beam approach that looks at alternate paths
#next tried sampling to randomly pick next word using conditional probability
#later tried the top-k sampling technique of picking the top k most likely words
#subsequently tried p-sampling, picking a small set of words with high probability
#next tried top k and p sampling and reduced low probability words

#encode context the generation is conditioned on
input_ids = tokenizer.encode(input_sequence, return_tensors='tf')

#combine both sampling techniques
sample_output = GPT2.generate(
    input_ids,
    do_sample=True,
    max_length=2*MAX_LEN,
    #to test how long we can generate coherent content
    #temperature = 0.8,
    top_k=50,
    top_p=0.85,
    num_return_sequences=5

)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    print('')