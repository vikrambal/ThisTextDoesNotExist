import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# For stopword removal
import nltk
en_stop = set(nltk.corpus.stopwords.words('english'))


from gensim.models.fasttext import FastText

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

"""
# Lemmatization
from nltk import WordNetLemmatizer
stemmer = WordNetLemmatizer()

# Read yelp review tip dataset
yelp_df = pd.read_json("source/yelp_academic_dataset_tip.json", lines=True)

print('List of all columns')
print(list(yelp_df))

# Checking for missing values in our dataframe
# No, there is no missing value
yelp_df.isnull().sum()

# Subset data for gensim fastText model
all_sent = list(yelp_df['text'])
some_sent = all_sent[0:100000]
some_sent[0:10]


# Text cleaning function for gensim fastText in python
def process_text(document):
    # Remove extra white space from text
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Remove all the special characters from text
    document = re.sub(r'\W', ' ', str(document))

    # Remove all single characters from text
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Converting to Lowercase
    document = document.lower()

    # Word tokenization
    tokens = document.split()
    # using NLTK
    lemma_txt = [stemmer.lemmatize(word) for word in tokens]
    # Remove stop words
    lemma_no_stop_txt = [word for word in lemma_txt if word not in en_stop]
    # Drop words
    tokens = [word for word in tokens if len(word) > 3]

    clean_txt = ' '.join(lemma_no_stop_txt)

    return clean_txt


clean_corpus = [process_text(sentence) for sentence in tqdm(some_sent) if sentence.strip() != '']

word_tokenizer = nltk.WordPunctTokenizer()
word_tokens = [word_tokenizer.tokenize(sent) for sent in tqdm(clean_corpus)]
print(word_tokens)

# Defining values for parameters
embedding_size = 300
window_size = 5
min_word = 5
down_sampling = 1e-2

# Create fastText model
# Migrating: https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
fast_Text_model = FastText(word_tokens,
                      vector_size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      workers=4,
                      sg=1,
                      epochs=100)
# Save fastText gensim model
fast_Text_model.save("model/ft_model_yelp")
"""




from gensim.models import Word2Vec
# Load saved gensim fastText model
fast_Text_model = Word2Vec.load("model/ft_model_yelp")

# Check word embedding for a particular word
print(fast_Text_model.wv['chicken'])
print(fast_Text_model.wv.most_similar('chicken', topn=10))
print(fast_Text_model.wv.similarity('beer', 'spirit'))
print(fast_Text_model.wv.most_similar(negative=["chicken"], topn=10))


# tsne plot for below word
# for_word = 'food'
def tsne_plot(for_word, w2v_model):
    # trained fastText model dimension
    dim_size = w2v_model.wv.vectors.shape[1]

    arrays = np.empty((0, dim_size), dtype='f')
    word_labels = [for_word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, w2v_model.wv.__getitem__([for_word]), axis=0)

    # gets list of most similar words
    sim_words = w2v_model.wv.most_similar(for_word, topn=10)

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
    df_plot = pd.DataFrame({'x': [x for x in Y[:, 0]],
                            'y': [y for y in Y[:, 1]],
                            'words_name': word_labels,
                            'words_color': color_list})

    # ------------------------- tsne plot <a href="https://thinkinfi.com/learn-python/" data-internallinksmanager029f6b8e52c="13" title="Best way to learn Python" target="_blank" rel="noopener">Python</a> -----------------------------------

    # plot dots with color and position
    plot_dot = sns.regplot(data=df_plot,
                           x="x",
                           y="y",
                           fit_reg=False,
                           marker="o",
                           scatter_kws={'s': 40,
                                        'facecolors': df_plot['words_color']
                                        }
                           )

    # Adds annotations with color one by one with a loop
    for line in range(0, df_plot.shape[0]):
        plot_dot.text(df_plot["x"][line],
                      df_plot['y'][line],
                      '  ' + df_plot["words_name"][line].title(),
                      horizontalalignment='left',
                      verticalalignment='bottom', size='medium',
                      color=df_plot['words_color'][line],
                      weight='normal'
                      ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for word "{}'.format(for_word.title()) + '"')


# tsne plot for top 10 similar word to 'chicken'
tsne_plot(for_word='chicken', w2v_model=fast_Text_model)
plt.show()

