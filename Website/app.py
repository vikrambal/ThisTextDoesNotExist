#import statements
#from flask import Flask, render_template, request, redirect, url_for, session
#import pandas as pd
#import numpy as np

#from gensim.models.fasttext import FastText

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
   return render_template('ttdne.html')

@app.route('/topic')
def topic():
    return render_template('topic.html')

@app.route('/embedding')
def analyzeWord(targetWord):
    #word = fast_Text_model.wv[targetWord]
    positives = fast_Text_model.wv.most_similar(targetWord, topn=10)
    #similarity = fast_Text_model.wv.similarity('beer', 'spirit')
    #negatives = fast_Text_model.wv.most_similar(negative=[targetWord], topn=10)
    return render_template('embedding.html', positives)
