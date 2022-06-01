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

from gensim.models.fasttext import FastText

from gensim.models import Word2Vec

# Load saved gensim fastText model
fast_Text_model = Word2Vec.load("ft_model_yelp")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import seaborn as sns
#import matplotlib.pyplot as plt


#setting up Flask
app = Flask(__name__) #this has 2 underscores on each side
app.secret_key = 'himynameistreihaveabasketballgametmrwwhereimapointguardigotshoegameandi'

#to access csv file because it gives an error if I don't do this https://stackoverflow.com/questions/20035101/why-does-my-javascript-code-receive-a-no-access-control-allow-origin-header-i
from flask import Flask
from flask_cors import CORS
CORS(app)

#Opening page here
@app.route('/')
def index():
   return render_template('home.html')

@app.route('/generation')
def generation():
    return render_template('generation.html')

@app.route('/randomgen', methods=['POST', 'GET'])
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
        genlength = int(request.form.get('genlength'))
        if genlength >= 50 and genlength <= 1000:
            if len(seed) >= 10:
                try:
                    # load the optimal weights
                    model.load_weights(f"results/{BASENAME}-{sequence_length}.h5")
                    s = seed
                    n_chars = int(genlength)
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
        else:
            generated = "Error: invalid submission."
    except Exception as e:
        generated = e

    return render_template('randomgen.html', generated=generated)

@app.route('/storygen')
def storygen():
    return render_template('storygen.html')

@app.route('/qnagen')
def qnagen():
    return render_template('qnagen.html')

@app.route('/word-embeddings', methods=['POST', 'GET'])
def wordEmbeddings():
    alertmsg = ''
    searchedWord = ''
    positives = ''
    negatives = ''
    #similarity = ''
    if request.method == 'POST' and 'targetWord' in request.form and 'topnSimGraph' in request.form:
        targetWord = request.form.get("targetWord")
        searchedWord = targetWord
        try:
            topnSimGraph = int(request.form.get("topnSimGraph"))
            model_to_csv(for_word=targetWord,w2v_model=fast_Text_model,sim_words_n=topnSimGraph)
        except Exception as e:
            alertmsg = 'Invalid submission'
        
        try:
            topnSim = int(request.form.get("nSimilar"))
            positives = fast_Text_model.wv.most_similar(targetWord, topn=topnSim)
        except Exception as e:
            positives = str(e)
        
        try:
            topnOpp = int(request.form.get("nOpposite"))
            negatives = fast_Text_model.wv.most_similar(negative=[targetWord], topn=topnOpp)
        except Exception as e:
            negatives = str(e)
        
        # similarity = fast_Text_model.wv.similarity(targetWord, 'spirit')

    return render_template('word-embeddings.html', alertmsg=alertmsg, targetWord=searchedWord, positiveWords=positives, negativeWords=negatives) #similarityScore=similarity


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/source')
def sourcepage():
    return render_template('source.html')

# converting model query of "for_word" to csv for D3.js
# original matplot version: https://thinkinfi.com/fasttext-word-embeddings-python-implementation/
def model_to_csv(for_word, w2v_model, sim_words_n):

    # trained fastText model dimension
    dim_size = w2v_model.wv.vectors.shape[1]

    arrays = np.empty((0, dim_size), dtype='f')
    word_labels = [for_word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, w2v_model.wv.__getitem__([for_word]), axis=0)

    # gets list of most similar words
    sim_words = w2v_model.wv.most_similar(for_word, topn=sim_words_n)

    # adds the vector for each of the closest words to the array
    for wrd_score in sim_words:
        wrd_vector = w2v_model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # ---------------------- Apply PCA and tsne to reduce dimension --------------

    # fit 2d PCA model to the similar word vectors
    model_pca = PCA(n_components=10).fit_transform(arrays)

    # Finds 2d coordinates t-SNE
    np.set_printoptions(suppress=True)
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(model_pca)

    # Sets everything up to plot
    df_plot = pd.DataFrame({'xCoord': [x for x in Y[:, 0]],
                            'yCoord': [y for y in Y[:, 1]],
                            'words_name': word_labels,
                            'words_color': color_list})

    # Convert DataFrame to CSV
    df_plot.to_csv('static/ft_model_query.csv', mode='w+', encoding='utf-8', index=False)
