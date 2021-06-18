# %% [markdown]
#
# Notebook du traitement des données pour en faire des data £utilisables 
# pour les traitements futures.  

# %%
# Import the requiered libraries
from binascii import Error
from collections import defaultdict
import os
from pathlib import Path
import json
import sys
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from matplotlib.pyplot import ylabel
import missingno as msno
from tensorflow.python.util.nest import _yield_value
from tqdm.notebook import tqdm
import logging  
logging.basicConfig(
                    format= "%(levelname)s - %(asctime)s: %(message)s", 
                    datefmt= '%H:%M:%S', 
                    level= logging.INFO
)
import multiprocessing

import pandas as pd
import numpy as np
import re
import gensim
from gensim.parsing.porter import PorterStemmer
import spacy
nlp = spacy.load("en_core_web_sm")


%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.preprocessing import LabelBinarizer

import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dropout 
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Bidirectional 
from keras.layers import Input
from keras.layers import Activation
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras
#from keras.layers.embeddings import Embedding
import pickle

# %%
# Define constants
FILE_TWEETS = 'sentiment140\\training.1600000.processed.noemoticon.csv'
FILE_DF_TRAIN = 'df_train_5000'
FILE_DF_VALIDATE = 'df_validate_1000'
FILE_DF_TEST = 'df_test_1000'
FILE_DF_TWEETS = 'df_tweets'


FILE_DICTIONARY = 'dict_simple'
FILE_WORD2VEC_CBOW = "word2vec_CBOW"
FILE_WORD2VEC_SKIPGRAM = "word2vec_Skip_Gram"


# Word2Vec kind
W2V_KIND = {'CBOW' : 0, 'SKIP_GRAM' : 1} 


# Dictionary size
TOP_WORDS = 5_000
# Max length of the embedding vectors. 
EMBEDDING_LENGTH = 100

# Create the simple dictionary even if it already exists
CREATE_SIMPLE_DICTIONARY = True

# Smiley - http://www.emoticonr.com/emoticons/
SMILEYS = {
            r"O:)"  : ':angel:',
            r"O:-)" : ':angel:',
            r"X("   : ':angry:',
            r">:-(" : ':angry:',
            r">:("  : ':angry:',
            r"X("   : ':angry:',
            r":)"   : ':smile:', 
            r":-)"  : ':smile:',
            r":(("  : ':crying:', 
            r":("   : ':frown:', 
            r"=(("  : ':broken heart',
            r"=("   : ':frown:', 
            r":-("  : ':frown:',
            r":["   : ':frown:', 
            r";)"   : ':winking:', 
            r";^)"  : ':winking:', 
            r";-)"  : ':winking:',
            r",)"   : ':winking:'  , 
            r",-)"  : ':winking:',
            r";p"   : ':sticking tongue out:', 
            r";-p"  : ':sticking tongue out:',
            r":p"   : '::sticking tongue out:', 
            r":-p"  : '::sticking tongue out::',
            r":|"   : ':straight face:', 
            r":-|"  : ':straight face:',
            r"O.o"  : ':confuse:', 
            r" :/"  : ':confuse:',  # Need space due to http://
            r":-/"  : ':confuse:',
            r":'("  : ':crying:', 
            r":'-(" : ':crying:',
            r"<3"   : ':love:',
            r"</3"  : ':broken heart:',
            r":D"   : ':grin:',
            r":-D"  : ':grin:',
            r":*"   : ':kiss:',
            r":-*"  : ':kiss:',
            r":-X"  : ':kiss:',
            r":x"   : ':love:',
}

WANTED_WORKERS = os.cpu_count() - 1
 
ADDED_STOPWORDS = set(['i', 'to'])
STOP_WORDS = gensim.parsing.preprocessing.STOPWORDS.union(ADDED_STOPWORDS)
# We want the negative so we keep no and not
REMOVED_STOPWORDS = set(['no', 'not'])
STOP_WORDS = STOP_WORDS.difference(REMOVED_STOPWORDS)

WORD2VEC_WINDOW = 3

# %%
# Read the data
path = os.getcwd() + "\\data\\"

df_tweets = pd.read_csv(
                        Path('./data/')/f"{FILE_TWEETS}",
#                        path + FILE_TWEETS,
                        names= [
                                'sentiment_score',
                                'id',
                                'date',
                                'query',
                                'user',
                                'tweet'
                        ],
                        encoding= 'latin',
)

df_tweets.drop(
                columns= ['id', 'date', 'query', 'user'],
                inplace= True
)

df_tweets['pos'] = df_tweets['sentiment_score'].apply(
                                                lambda x: x if x == 0 else 1
)
df_tweets['neg'] = df_tweets['pos'].apply(lambda x: 0 if x == 1 else 1)


# %% [markdown]
# Nous créons les différents type de transformations et les sauvegardons dans
# des features

# %%
# Define the functions to work the tweets
def smileys_to_words(tweet):
    for k,v in SMILEYS.items():
        tweet = tweet.replace(k, v)
    return tweet

def clean_tweet(tweet):
    # Remove the @....
    tmp_tweet = re.sub(
                        r'((@)[^ ]*)',
                        '',
                        tweet
    )

    # Delete the http:// and https://
    tmp_tweet = re.sub(
                        r'((http(s|)://)[^ ]*)',
                        '',
                        tmp_tweet
    )

    # Go for lower case
    tmp_tweet = tmp_tweet.lower()

    # delete letters that are repeated more than twice
    tmp_tweet = re.sub(
                        r'(.)\1{2,}', 
                        r'\1', 
                        tmp_tweet
    )

    # changes "n't" to " not"
    tmp_tweet = re.sub(
                        r"(n't)",
                        " not",
                        tmp_tweet
    )

    # remove the "'re" "'s" "'d" as they are to be stop words
    tmp_tweet = re.sub(
                        r"('(re|d|s|m|ll))",
                        "",
                        tmp_tweet
    )

    # keep only the alphabet + # plus :.....: that were smileys
    # and remove the stopwords
    tokens = [
                token
                for token in re.findall(
                                            pattern= r"[a-z]+|#[a-z]+|:[a-z]+:",
                                            string= tmp_tweet,
                                            flags= re.MULTILINE
                )
                if token not in STOP_WORDS
    ]
    return " ".join(tokens)

# Fist pipeline
def pipeline_1(tweet, progress_bar= None):
    if progress_bar is not None:
        progress_bar.update()
    tmp_tweet = smileys_to_words(tweet= tweet)
    tmp_tweet = clean_tweet(tmp_tweet)
    return tmp_tweet

# Stemming
def stem_tweet(tweet, stemmer= None, progress_bar= None):
    if progress_bar is not None:
        progress_bar.update()
    if stemmer is None:
        stemmer = PorterStemmer()
    return stemmer.stem_sentence(tweet)

# Lemmatization
def lemma_tweet(tweet, progress_bar= None):
    if progress_bar is not None:
        progress_bar.update()
    return " ".join([token.lemma_ for token in nlp(tweet)])



# Word to figure : create the dictionary
def word2figure_dict(df, file_name_dict):
    if (os.path.exists(Path('./data/')/f"{file_name_dict}")):
        with open(Path('./data/')/f"{file_name_dict}", 'r') as json_file:
            data = json.load(json_file)
            tokenizer = tokenizer_from_json(data)
    else:
        # We have to create the tokenizer. 
        tokenizer = Tokenizer(
                                lower= True,
                                split= ' '
        )
        tokenizer.fit_on_texts(df.values)
        with open(Path('./data/')/f"{file_name_dict}", 'w') as json_file:
            json_file.write(
                            json.dumps(
                                        tokenizer.to_json(),
                                        ensure_ascii= False
                            )
            )
    return tokenizer

# Word to figure : tweet
def word2figure_tweet(tweet, tokenizer, progress_bar= None):
    if progress_bar is not None:
        progress_bar.update()
    return tokenizer.texts_to_sequences(tweet)


# Word2Vec - Create the dict
def word2vec_dict(df, file_name_dict, kind= W2V_KIND['CBOW']):
    if os.path.exists(Path('./data/')/f"{file_name_dict}"):
        w2v_model = Word2Vec.load(os.getcwd() + f"\data\{file_name_dict}")
    else:
        # We have to create the Word2Vec. 
        w2v_model = Word2Vec(
                                sentences= [
                                            tweet.split(' ') 
                                            for tweet in df
                                ],
                                min_count= 1,
                                vector_size= TOP_WORDS,
                                workers= WANTED_WORKERS,
                                window= WORD2VEC_WINDOW,
                                sg= kind                       # CBOW
        )
        w2v_model.save(os.getcwd() + f"\data\{FILE_WORD2VEC_CBOW}.model")
    return w2v_model

# Word2Vec - create the vectors
def tweet_to_w2v(sentence, w2v_model, w2v_model_vocab, progress_bar= None):
    if progress_bar is not None:
        progress_bar.update()
    if type(w2v_model) == gensim.models.keyedvectors.KeyedVectors:
        return [
                np.mean(w2v_model[token]) 
                if token in w2v_model_vocab else 0
                for token in sentence.split(' ')
        ]
    else:
        return [
                np.mean(w2v_model.wv[token]) 
                if token in w2v_model_vocab else 0
                for token in sentence.split(' ')
        ]
    
# %%
# clean the tweets
progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'treatments',
                    leave= False
)
df_tweets['clean'] = [
                        pipeline_1(
                                    tweet= tweet, 
                                    progress_bar= progress_bar
                        )
                        for tweet in df_tweets['tweet']
]
progress_bar.close()

# %%
# Word to figure of clean tweets
tokenizer = word2figure_dict(
                                df= df_tweets['clean'],
                                file_name_dict= "Word_2_Figure_Clean.json"
)
progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'w2f',
                    leave= False
)
df_tweets['w2f_clean'] = [
                            word2figure_tweet(
                                                tweet= tweet,
                                                tokenizer= tokenizer, 
                                                progress_bar= progress_bar
                            )
                            for tweet in df_tweets['clean']
]
progress_bar.close()


# %%
# Stemming
progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'stem',
                    leave= False
)
stemmer = PorterStemmer()
df_tweets['stem'] = [
                        stem_tweet(
                                    tweet= tweet,
                                    stemmer= stemmer, 
                                    progress_bar= progress_bar
                        )
                        for tweet in df_tweets['clean']
]
progress_bar.close()

tokenizer = word2figure_dict(
                                df= df_tweets['stem'],
                                file_name_dict= "Word_2_Figure_Stem.json"
)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'w2f(stem)',
                    leave= False
)
df_tweets['w2f_stem'] = [
                            word2figure_tweet(
                                                tweet= tweet,
                                                tokenizer= tokenizer, 
                                                progress_bar= progress_bar
                            )
                            for tweet in df_tweets['stem']
]
progress_bar.close()

# %%
# Lemmatization
progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'lemma',
                    leave= False
)
df_tweets['lemma'] = [
                        lemma_tweet(
                                    tweet= tweet,
                                    progress_bar= progress_bar
                        )
                        for tweet in df_tweets['clean']
]
progress_bar.close()

tokenizer = word2figure_dict(
                                df= df_tweets['lemma'],
                                file_name_dict= "Word_2_Figure_Lemma.json"
)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'w2f(lemma)',
                    leave= False
)
df_tweets['w2f_lemma'] = [
                            word2figure_tweet(
                                                tweet= tweet,
                                                tokenizer= tokenizer, 
                                                progress_bar= progress_bar
                            )
                            for tweet in df_tweets['lemma']
]
progress_bar.close()


# %%
# Word 2 Vec on clean tweet
w2v_model = word2vec_dict(
                            df= df_tweets['clean'],
                            file_name_dict= "w2v_CBOW_model_clean.model",
                            kind= W2V_KIND['CBOW']
)
# Create a set so the work is faster
w2v_model_vocab = set(w2v_model.wv.index_to_key)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V CBOW Clean',
                    leave= False
)
df_tweets['w2v_CBOW_clean'] = [
                                tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                                )
                                for sentence in df_tweets['clean']
]
progress_bar.close()

# %%
# Word 2 Vec for stem
w2v_model = word2vec_dict(
                            df= df_tweets['stem'],
                            file_name_dict= "w2v_CBOW_model_stem.model",
                            kind= W2V_KIND['CBOW']
)
# Create a set so the work is faster
w2v_model_vocab = set(w2v_model.wv.index_to_key)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V CBOW Stem',
                    leave= False
)
df_tweets['w2v_CBOW_stem'] = [
                                tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                                )
                                for sentence in df_tweets['stem']
]
progress_bar.close()

# %%
# Word 2 Vec for lemma
w2v_model = word2vec_dict(
                            df= df_tweets['lemma'],
                            file_name_dict= "w2v_CBOW_model_lemma.model",
                            kind= W2V_KIND['CBOW']
)
# Create a set so the work is faster
w2v_model_vocab = set(w2v_model.wv.index_to_key)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V CBOW Lemma',
                    leave= False
)
df_tweets['w2v_CBOW_lemma'] = [
                                tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                                )
                                for sentence in df_tweets['lemma']
]
progress_bar.close()

# %%
# Word 2 Vec Skip Gram on clean tweet
w2v_model = word2vec_dict(
                            df= df_tweets['clean'],
                            file_name_dict= "w2v_SG_model_clean.model",
                            kind= W2V_KIND['SKIP_GRAM']
)
# Create a set so the work is faster
w2v_model_vocab = set(w2v_model.wv.index_to_key)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V SG Clean',
                    leave= False
)
df_tweets['w2v_SG_clean'] = [
                                tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                                )
                                for sentence in df_tweets['clean']
]
progress_bar.close()

# %%
# Word 2 Vec for stem
w2v_model = word2vec_dict(
                            df= df_tweets['stem'],
                            file_name_dict= "w2v_SG_model_stem.model",
                            kind= W2V_KIND['CBOW']
)
# Create a set so the work is faster
w2v_model_vocab = set(w2v_model.wv.index_to_key)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V SG Stem',
                    leave= False
)
df_tweets['w2v_SG_stem'] = [
                            tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                            )
                            for sentence in df_tweets['stem']
]
progress_bar.close()

# %%
# Word 2 Vec for lemma
w2v_model = word2vec_dict(
                            df= df_tweets['lemma'],
                            file_name_dict= "w2v_SG_model_lemma.model",
                            kind= W2V_KIND['SKIP_GRAM']
)
# Create a set so the work is faster
w2v_model_vocab = set(w2v_model.wv.index_to_key)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V SG Lemma',
                    leave= False
)
df_tweets['w2v_SG_lemma'] = [
                                tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                                )
                                for sentence in df_tweets['lemma']
]
progress_bar.close()

# %%
# Word 2 Vec with Google
wordmodelfile=Path('./data/')/"GoogleNews-vectors-negative300.bin.gz"
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                                                            wordmodelfile,
                                                            binary=True
)
# Create a set so the work is faster
w2v_model_vocab = set(w2v_model.index_to_key)

progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V Google Clean',
                    leave= False
)
df_tweets['w2v_Google_clean'] = [
                                tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                                )
                                for sentence in df_tweets['clean']
]
progress_bar.close()

# %%
# Word 2 Vec for stem
progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V Google Stem',
                    leave= False
)
df_tweets['w2v_Google_stem'] = [
                                tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                                )
                                for sentence in df_tweets['stem']
]
progress_bar.close()

# %%
# Word 2 Vec for lemma
progress_bar = tqdm(
                    total= len(df_tweets),
                    desc= 'W2V Google Lemma',
                    leave= False
)
df_tweets['w2v_Google_lemma'] = [
                                    tweet_to_w2v(
                                            sentence= sentence,
                                            w2v_model= w2v_model,
                                            w2v_model_vocab= w2v_model_vocab, 
                                            progress_bar= progress_bar
                                    )
                                    for sentence in df_tweets['lemma']
]
progress_bar.close()


# %%
# Create the Train, Validate and Test to do the comparaisons
df_train = pd.read_pickle(os.getcwd() + f"\\data\\{FILE_DF_TRAIN}.pkl")
index_train = df_train.index
df_train_augmented = df_tweets.loc[index_train]
df_train_augmented.to_pickle(os.getcwd() + f"\\data\\{FILE_DF_TRAIN}-Augmented.pkl")

df_validate = pd.read_pickle(os.getcwd() + f"\\data\\{FILE_DF_VALIDATE}.pkl")
index_validate = df_validate.index
df_validate_augmented = df_tweets.loc[index_validate]
df_validate_augmented.to_pickle(os.getcwd() + f"\\data\\{FILE_DF_VALIDATE}-Augmented.pkl")

df_test = pd.read_pickle(os.getcwd() + f"\\data\\{FILE_DF_TEST}.pkl")
index_test = df_test.index
df_test_augmented = df_tweets.loc[index_test]
df_test_augmented.to_pickle(os.getcwd() + f"\\data\\{FILE_DF_TEST}-Augmented.pkl")


# %%
df_tweets.to_csv(os.getcwd() + f"\\data\\{FILE_DF_TWEETS}.csv")

# %%
# Save everything
with open(os.getcwd() + f"\\data\\{FILE_DF_TWEETS}.pkl", 'wb') as file_pickle:
    pickle.dump(
                df_tweets,
                file_pickle,
                protocol= pickle.HIGHEST_PROTOCOL
    )
