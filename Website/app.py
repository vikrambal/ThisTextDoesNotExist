#import statements
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np

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
    if request.method == 'POST' and 'inputWord' in request.form:
        targetWord = request.form.get("targetWord")
        positives = fast_Text_model.wv.most_similar(targetWord, topn=10)
        negatives = fast_Text_model.wv.most_similar(negative=[targetWord], topn=10)
        similarity = fast_Text_model.wv.similarity(targetWord, 'spirit')
    return render_template('embeddings.html', positiveWords=positives, negativeWords=negatives, similarityScore=similarity)
