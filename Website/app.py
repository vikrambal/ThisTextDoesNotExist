#import statements
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np

# packages needed for RNN LSTM model

import pickle
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import os

#from gensim.models.fasttext import FastText

from gensim.models import Word2Vec

fast_Text_model = Word2Vec.load("ft_model_yelp")

#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#import seaborn as sns
#import matplotlib.pyplot as plt

#setting up Flask
app = Flask(__name__) #this has 2 underscores on each side
app.secret_key = 'himynameistreihaveabasketballgametmrwwhereimapointguardigotshoegameandi'

#Opening page here
@app.route('/')
def index():
   return render_template('randomgen.html')

@app.route('/qna')
def topic():
    return render_template('qna.html')

@app.route('/embeddings', methods=['POST', 'GET'])
def analyzeWord():
    positives = ''
    negatives = ''
    similarity = ''
    if request.method == 'POST' and 'targetWord' in request.form:
        targetWord = request.form.get("targetWord")
        positives = fast_Text_model.wv.most_similar(targetWord, topn=10)
        negatives = fast_Text_model.wv.most_similar(negative=[targetWord], topn=10)
        similarity = fast_Text_model.wv.similarity(targetWord, 'spirit')
    return render_template('embeddings.html', positiveWords=positives, negativeWords=negatives, similarityScore=similarity)

@app.route('/custom', methods=['POST', 'GET'])
def customGenerate():
    #length of sequence from text
    sequence_length = 100
    # dataset file path
    FILE_PATH = "data/wonderland.txt"
    # FILE_PATH = "data/python_code.py"
    BASENAME = os.path.basename(FILE_PATH)

    #create seed that will start generation
    seed = request.form.get('seed')

    # load vocab dictionaries to this file
    char2int = pickle.load(open(f"{BASENAME}-char2int.pickle", "rb"))
    int2char = pickle.load(open(f"{BASENAME}-int2char.pickle", "rb"))
    vocab_size = len(char2int)

    # building the model again
    model = Sequential([
        LSTM(256, input_shape=(sequence_length, vocab_size), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(vocab_size, activation="softmax"),
    ])
    try:
        if (len(seed) >= 10):
            try:
                # load the optimal weights
                model.load_weights(f"results/{BASENAME}-{sequence_length}.h5")
                s = seed
                n_chars = 50 # can make this an option of the user in the future.
                # generate 400 characters in future but 10 for tests
                generated = ""
                for i in tqdm.tqdm(range(n_chars), "Generating text"):
                    # make the input sequence
                    X = np.zeros((1, sequence_length, vocab_size))
                    for t, char in enumerate(seed):
                        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
                    # predict the next character
                    predicted = model.predict(X, verbose=0)[0]
                    # converting the vector to an integer
                    next_index = np.argmax(predicted)
                    # converting the integer to a character
                    next_char = int2char[next_index]
                    # add the character to results
                    generated += next_char
                    # shift seed and the predicted character
                    seed = seed[1:] + next_char
            except Exception as e:
                generated = "Error: invalid seed. Avoid punctuation and capitalisation for now."
        else:
            generated = "Error: the entered seed was too short (must be at least 10 chars)."
    except Exception as e:
        generated = e

    return render_template('custom.html', generated=generated)
