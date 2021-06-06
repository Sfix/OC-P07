# %% [markdown]
#
# Notebook du traitement des données par Keras
# https://docs.microsoft.com/fr-fr/learn/modules/analyze-review-sentiment-with-keras/2-build-and-train-a-neural-network
#  

# %%
# Import the requiered libraries
from collections import defaultdict
import os
from pathlib import Path
import json
import sys
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from keras.layers import embeddings
from keras_preprocessing.text import tokenizer_from_json
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
import nltk.stem
import string
import pickle

from sklearn.feature_extraction import text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from traitlets.traitlets import validate

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding

# %%
# Define constants
FILE_TWEETS = 'sentiment140\\training.1600000.processed.noemoticon.csv'
FILE_DF_TRAIN = 'df_train_5000'
FILE_DF_VALIDATE = 'df_validate_1000'
FILE_DF_TEST = 'df_test_1000'

FILE_DICTIONARY = 'dict_simple'

# Compare 
COMPARE_SOLUTIONS = True

# When prototype COEF = 0.1. Otherwhize = 1
COEF = 0.1

# Dictionary size
TOP_WORDS = 10_000
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

 
ADDED_STOPWORDS = set(['i', 'to'])
STOP_WORDS = gensim.parsing.preprocessing.STOPWORDS.union(ADDED_STOPWORDS)

# %%
# Read the data
df_test = pd.read_pickle(
                            Path('./data/')/f"{FILE_DF_TEST}.pkl"
)
df_train = pd.read_pickle(
                            Path('./data/')/f"{FILE_DF_TRAIN}.pkl"
)
df_validate = pd.read_pickle(
                                Path('./data/')/f"{FILE_DF_VALIDATE}.pkl"
)

# %% [markdown]
# Proceed the DataFrames

# %%
# Clean the tweets
def tweets_cleaner(df):
    """Work on the feature "tweet" to clean them (stopwords, lower ...)
    and return the result in a feature "clean"

    Args:
        df (DataFrame): Tweets must be in "tweet"

    Returns:
        DataFrme: the dataframe with a new feature: "clean"
    """
    def smileys_to_words(tweet):
        for k,v in SMILEYS.items():
            tweet = tweet.replace(k, v)
        return tweet

    df['smiley_translated'] = [
                                smileys_to_words(tweet)
                                for tweet in df['tweet']
    ]    

    # Delete the @....
    df['cleaner'] = [
                        re.sub(
                                r'((@)[^ ]*)',
                                '',
                                tweet
                        )
                        for tweet in df['smiley_translated']
    ]

    # Delete the http:// and https://
    df['cleaner'] = [
                        re.sub(
                                r'((http(s|)://)[^ ]*)',
                                '',
                                tweet
                        )
                        for tweet in df['cleaner']
    ]

    # Go for lower case
    df['cleaner'] = [
                        tweet.lower()
                        for tweet in df['cleaner']
    ]

    # delete letters that are repeated more than twice
    df['cleaner'] = [
                        re.sub(
                                r'([a-z][a-z])\1+', 
                                r'\1', 
                                tweet
                        )
                        for tweet in df['cleaner']
    ]

    # changes "n't" to " not"
    df['cleaner'] = [
                        re.sub(
                                r"(n't)",
                                " not",
                                tweet
                        )
                        for tweet in df['cleaner']
    ]

    # remove the "'re" "'s" "'d" as they are to be stop words
    df['cleaner'] = [
                        re.sub(
                                r"('(re|d|s|m|ll))",
                                "",
                                tweet
                        )
                        for tweet in df['cleaner']
    ]

    # keep only the alphabet + # plus :.....: that were smileys
    def keep_match(tweet):
        regex = r"[a-z]+|#[a-z]+|:[a-z]+:"
        matches = re.findall(
                                pattern= regex,
                                string= tweet,
                                flags= re.MULTILINE
        )
        return " ".join(matches)

    df['cleaner'] = [
                    keep_match(tweet)
                    for tweet in df['cleaner']
    ]

    # remove the stopwords
    df['clean'] = [
                    " ".join(
                                [
                                    word
                                    for word in sentence.split(' ')
                                    if word not in STOP_WORDS
                                ]
                    )
                    for sentence in df['cleaner']
    ]

    # Remove the feature 'cleaner'
    df.drop(columns= ['cleaner'], inplace= True)
    return df

progress_bar = tqdm(
                    total= 3,
                    desc= "clean",
                    leave= False
)
for df in [df_train, df_validate, df_test]:
    progress_bar.update()
    df = tweets_cleaner(df)
progress_bar.close()


# %%
# Create X and y with the simpliest solution
def create_X_y_simple(df, invert_score= True, new_dictionary= CREATE_SIMPLE_DICTIONARY):
    """With the simpliest solution, changing words for their frequency,
    this function create X and y for the DataFrame.
    If new_dictionary at False and there is a FILE_DICTIONAY, then it 
    is loaded. Any word that is not in it will be put as 2.

    Args:
        df (DataFrame): DataFrame with a feature 'score' to generate y
        and 'clean' to generate X.
        invert_score (boolean, optional) : define if the score must be
        inverted (True) or not (False). Defaults to True
        new_dictionary (boolean, optional): define if the dictionary
        is loaded if exists (False) or creted and saved (True). 
        Defaults to CREATE_SIMPLE_DICTIONARY.

    Returns:
        Turple: (X, y)
    """
    # Check if we go for existing disctionnary or not 
    if (
        not(new_dictionary) 
        and 
        os.path.exists(Path('./data/')/f"{FILE_DICTIONARY}.json")
    ):
        with open(Path('./data/')/f"{FILE_DICTIONARY}.json", 'r') as json_file:
            data = json.load(json_file)
            tokenizer = tokenizer_from_json(data)
    else:
        # We have to create the tokenizer. 
        tokenizer = Tokenizer(
                                lower= True,
                                split= ' '
        )
        tokenizer.fit_on_texts(df['clean'].values)
        with open(Path('./data/')/f"{FILE_DICTIONARY}.json", 'w') as json_file:
            json_file.write(
                            json.dumps(
                                        tokenizer.to_json(),
                                        ensure_ascii= False
                            )
            )

    # We have a dictionary, we translate the clean sentences into 
    # series of numbers
    df['es'] = [
                                np.array([
                                    tokenizer.word_index[token]
                                    if token in tokenizer.word_index.keys() else 2
                                    for token in sentence.split(' ')
                                ],
                                dtype= np.int32
                                )
                                for sentence in df['clean'].values
    ]
    X = df['es'].values

    df['embedding_simple'] = tokenizer.texts_to_sequences(df['clean'].values)
    X = sequence.pad_sequences(
                                df['embedding_simple'],
                                maxlen= EMBEDDING_LENGTH,
    )
    if invert_score:
        y = np.array(
                    [1 if score == 0 else 0 for score in df['score']]
        )
    else:
        y = np.array(df['score'].values)
    
    return (X, y)


# Compute the requiered X, y
(X_train, y_train) = create_X_y_simple(
                                        df= df_train,
                                        invert_score= True,
                                        new_dictionary= True
)
(X_validate, y_validate) = create_X_y_simple(
                                                df= df_validate,
                                                invert_score= True,
                                                new_dictionary= False
)
(X_test, y_test) = create_X_y_simple(
                                        df= df_test,
                                        invert_score= True,
                                        new_dictionary= False
)
max_length_sentence = EMBEDDING_LENGTH



# %%
# Create the NN, fit it and display some résults
def create_simple_NN_fit_and_display(train, validate, test, epochs, batch_size):
    """Fit the model and display the results

    Args:
        train (turple): (X_train, y_train)
        validate (turple): (X_validate, y_validate)
        test (turple): (X_test, y_test)
        epochs (int): number of epoch
        batch_size (int): size of the batch

    Returns:
        score (float): score on the test
    """
    # Create the neural network
    embedding_vector_length = 32
    model = Sequential()
    model.add(
                Embedding(
                            EMBEDDING_LENGTH,
                            embedding_vector_length,
                            input_length= len(train[0])
                )
    )
    model.add(Flatten())
    # 16 is totally arbitrary. Other values can be 
    # used.
    model.add(Dense(16, activation= 'relu'))
    model.add(Dense(16, activation= 'relu'))
    model.add(Dense(1,  activation= 'sigmoid'))
    model.compile(
                    loss= 'binary_crossentropy',
                    optimizer= 'adam',
                    metrics= ['accuracy']
    )
    print(model.summary())

    hist = model.fit(
                     train[0], train[1],
                    validation_data= (validate[0], validate[1]),
                    epochs= epochs,
                    batch_size= batch_size
    )
    # Plot the results
    acc = hist.history['accuracy']
    val = hist.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label= 'Training accuracy')
    plt.plot(epochs, val, ':', label= 'Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.plot()
    plt.show()

    scores = model.evaluate(
                            test[0], test[1],
                            verbose= 0
    )
    print(f"Accuracy est de {100 * scores[1]:.2f} %")
    return scores

# %%
# First test
score = create_simple_NN_fit_and_display(
                                train= (X_train, y_train),
                                validate= (X_validate, y_validate),
                                test= (X_test, y_test),
                                epochs= 15,
                                batch_size= 128
)
score = create_simple_NN_fit_and_display(
                                train= (X_train, y_train),
                                validate= (X_validate, y_validate),
                                test= (X_test, y_test),
                                epochs= 3,
                                batch_size= 128
)
score = create_simple_NN_fit_and_display(
                                train= (X_train, y_train),
                                validate= (X_validate, y_validate),
                                test= (X_test, y_test),
                                epochs= 5,
                                batch_size= 32
)
score = create_simple_NN_fit_and_display(
                                train= (X_train, y_train),
                                validate= (X_validate, y_validate),
                                test= (X_test, y_test),
                                epochs= 2,
                                batch_size= 32
)


# %%
# Create a LSTM, fit it and display some résults
def create_LSTM_fit_and_display(
                                train, validate, 
                                test, 
                                embed_dim, lstm_out, batch_size,
                                epochs, verbose
):
    """Create a LSTM, fit the model and display the results

    Args:
        train (turple): (X_train, y_train)
        validate (turple): (X_validate, y_validate)
        test (turple): (X_test, y_test)
        embed_dim (int): number of embed_dim
        lstm_out (int): number for lstm_out
        batch_size (int): size of the batch
        epochs (int): number of epochs
        verbose (int): verbose value

    Returns:
        score (float): score on the test
    """
    # Create the neural network

    model = Sequential()
    model.add(
                Embedding(
                            len(train[0]), 
                            embed_dim,
                            input_length = train.shape[1], 
                            dropout = 0.2
                )
    )
    model.add(
                LSTM(
                        lstm_out, 
                        dropout_U= 0.2, 
                        dropout_W= 0.2
                )
    )
    model.add(
                Dense(
                        2,
                        activation='softmax'
                )
    )
    model.compile(
                    loss= 'categorical_crossentropy', 
                    optimizer='adam',
                    metrics = ['accuracy']
    )
    print(model.summary())

    hist = model.fit(
                        train[0], train[1],
                        validation_data= (validate[0], validate[1]),
                        epochs= epochs,
                        batch_size= batch_size,
                        verbose= verbose
    )
    # Plot the results
    acc = hist.history['accuracy']
    val = hist.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label= 'Training accuracy')
    plt.plot(epochs, val, ':', label= 'Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.plot()
    plt.show()

    scores = model.evaluate(
                            test[0], test[1],
                            verbose= 0
    )
    print(f"Accuracy est de {100 * scores[1]:.2f} %")
    return scores

# %%
# Compute one
score = create_LSTM_fit_and_display(
                                    train= (X_train, y_train),
                                    validate= (X_validate, y_validate),
                                    test= (X_test, y_test),
                                    embed_dim= 128,
                                    lstm_out= 200, 
                                    epochs= 15,
                                    batch_size= 128,
                                    verbose= 5,
)

# %%