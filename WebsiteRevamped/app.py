#import statements
from flask import Flask, render_template, request, redirect, url_for, session
#setting up Flask
app = Flask(__name__) #this has 2 underscores on each side
app.secret_key = 'himynameistreihaveabasketballgametmrwwhereimapointguardigotshoegameandi'

#Opening page here
@app.route('/')
def index():
   return render_template('ttdne.html')

#Opening page here
@app.route('/generation')
def generation():
   return render_template('generation.html')

#Opening page here
@app.route('/about')
def about():
   return render_template('about.html')

 #Opening page here
@app.route('/embedding')
def embedding():
    return render_template('embedding.html')

#Opening page here
@app.route('/resource')
def resource():
    return render_template('source.html')
