# %% [markdown]
#
# Notebook d'exploration des models en partant du plus simple
# https://docs.microsoft.com/fr-fr/learn/modules/analyze-review-sentiment-with-keras/2-build-and-train-a-neural-network
# afin de définir le meilleur model à travailler et monter sur Azure 
# 

# %% [markdown]
# Charge les librairies et défini les constantes 


# %%
# Import the requiered libraries
from collections import defaultdict
import os
from pathlib import Path
import time
import json
import sys
from gensim.models.keyedvectors import Vocab
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from keras.backend import dtype
from keras_preprocessing.text import tokenizer_from_json
from matplotlib.pyplot import ylabel
import missingno as msno
from numpy.core.fromnumeric import trace
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

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Dropout, Dense, Flatten, LSTM, Bidirectional 
#from keras.layers.embeddings import Embedding

# %%
# Define constants
FILE_TWEETS = 'sentiment140\\training.1600000.processed.noemoticon.csv'
FILE_DF_TRAIN = 'df_train_5000-Augmented'
FILE_DF_VALIDATE = 'df_validate_1000-Augmented'
FILE_DF_TEST = 'df_test_1000-Augmented'

# Compare 
COMPARE_SOLUTIONS = True

# Dictionary size
TOP_WORDS = 5_000
# Max length of the embedding vectors. 
EMBEDDING_LENGTH = 100
# Create an offset so we have a value for unknown words.
OFFSET_FIGURE_WORD = 2
NUMBER_UNKNOWN = 1


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

# %% [markdown]
# Lit les tweets utilisés pour l'entrainement, la validation et le 
# test.
 
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
# Défini les fonctions qui seront utilisées pour pouvoir tester les
# solutions avec différents hyperparamètres et choisir la saolution
# la plus performante. 

# %%
# Work on a model
def fit_and_evaluate(model, test):
    scores = model.evaluate(
                            test[0], test[1],
                            verbose= 0
    )
    print(f"AUC est de {100 * scores[1]:.2f} %")
    return scores


# %%
# Create the NN, fit it and display some résults
def create_simple_NN_fit_and_display(
                                        train, 
                                        validate, 
                                        test, 
                                        input_dim= None, 
                                        epochs= 50, 
                                        batch_size= 32):
    # Create the neural network
    embedding_vector_length = 32
    model = Sequential()
    if input_dim == None:
        input_dim = max(train[0].flatten()) + 1
    model.add(
                Embedding(
                            input_dim= input_dim,
                            output_dim= embedding_vector_length,
                            input_length= EMBEDDING_LENGTH
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
                    metrics= ['AUC']
    )
    print(model.summary())

    hist = model.fit(
                        train[0], train[1],
                        validation_data= validate,
                        epochs= epochs,
                        batch_size= batch_size
    )
    # Plot the results
    auc = hist.history['auc']
    val = hist.history['val_auc']
    epochs = range(1, len(auc) + 1)

    plt.plot(epochs, auc, '-', label= 'Training Area Under the Curve')
    plt.plot(epochs, val, ':', label= 'Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(loc='upper left')
    plt.plot()
    plt.show()

    scores = model.evaluate(
                            test[0], test[1],
                            verbose= 0
    )
    print(f"AUC est de {100 * scores[1]:.2f} %")
    return scores


# %%
# Create the NN, fit it and display some résults
def create_simple_NN_fit_and_display(
                                        train, 
                                        validate, 
                                        test, 
                                        epochs= 50, 
                                        batch_size= 32,
                                        input_dim= None, 
                                        embedding= True):
    # Create the neural network
    embedding_vector_length = 32
    model = Sequential()
    if embedding:
        if input_dim == None:
            input_dim = max(train[0].flatten()) + 1
        model.add(
                    Embedding(
                                input_dim= input_dim,
                                output_dim= embedding_vector_length,
                                input_length= EMBEDDING_LENGTH
                    )
        )
        model.add(Flatten())
        # 16 is totally arbitrary. Other values can be 
        # used.
        model.add(Dense(16, activation= 'relu'))
    else:
        # The embedding is already done thanks to word2vec
        model.add(Dense(16, activation= 'relu', input_dim= input_dim))
    model.add(Dense(16, activation= 'relu'))
    model.add(Dense(1,  activation= 'sigmoid'))
    model.compile(
                    loss= 'binary_crossentropy',
                    optimizer= 'adam',
                    metrics= ['AUC']
    )
    print(model.summary())

    hist = model.fit(
                        train[0], train[1],
                        validation_data= validate,
                        epochs= epochs,
                        batch_size= batch_size
    )
    # Plot the results
    auc = hist.history['auc']
    val = hist.history['val_auc']
    epochs = range(1, len(auc) + 1)

    plt.plot(epochs, auc, '-', label= 'Training Area Under the Curve')
    plt.plot(epochs, val, ':', label= 'Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(loc='upper left')
    plt.plot()
    plt.show()

    scores = model.evaluate(
                            test[0], test[1],
                            verbose= 0
    )
    print(f"AUC est de {100 * scores[1]:.2f} %")
    return scores

# %%
# Create a LSTM, fit it and display some résults
def create_LSTM_fit_and_display(
                                train, validate, 
                                test, 
                                embed_dim, lstm_out, batch_size,
                                epochs, verbose, embedding= True
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
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    if embedding:
        # Embed each integer in a 128-dimensional vector
        x = Embedding(max(train[0].flatten()) + 1, embed_dim)(inputs)
    # Add 2 bidirectional LSTMs
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
                    loss= 'binary_crossentropy',
                    optimizer= 'adam',
                    metrics= ['AUC']
    )
#    model.summary()

    hist = model.fit(
                        train[0], train[1],
                        validation_data= (validate[0], validate[1]),
                        epochs= epochs,
                        batch_size= batch_size,
                        verbose= verbose
    )
    # Plot the results
    auc = hist.history['auc']
    val = hist.history['val_auc']
    epochs = range(1, len(auc) + 1)

    plt.plot(epochs, auc, '-', label= 'Training Area Under the Curve')
    plt.plot(epochs, val, ':', label= 'Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(loc='upper left')
    plt.plot()
    plt.show()

    scores = model.evaluate(
                            test[0], test[1],
                            verbose= 0
    )
    print(f"AUC est de {100 * scores[1]:.2f} %")
    return scores


# %%
# Create the neural network, fit it and display the results
def create_Glove_fit_and_display(
                                    train,
                                    validate,
                                    test,
                                    epochs,
                                    batch_size,
                                    input_shape,
                                    embedding_layer,
                                    verbose= 0
):
    layer_indices = Input(input_shape)
    embeddings = embedding_layer(layer_indices)
    x = LSTM(128, return_sequences=True)(embeddings)
    x = Dropout(0.6)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.6)(x)
    x = LSTM(128)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs= layer_indices, outputs= x)

    adam = keras.optimizers.Adam(learning_rate= 0.0001)
    model.compile(
                    optimizer= adam, 
                    loss= 'binary_crossentropy', 
                    metrics= ['AUC']
    )
#    model.summary()

    hist = model.fit(
                        train[0], train[1],
                        validation_data= (validate[0], validate[1]),
                        epochs= epochs,
                        batch_size= batch_size,
                        verbose= verbose
    )
    # Plot the results
    auc = hist.history['auc']
    val = hist.history['val_auc']
    epochs = range(1, len(auc) + 1)

    plt.plot(epochs, auc, '-', label= 'Training Area Under the Curve')
    plt.plot(epochs, val, ':', label= 'Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(loc='upper left')
    plt.plot()
    plt.show()

    scores = model.evaluate(
                            test[0], test[1],
                            verbose= 0
    )
    print(f"AUC est de {100 * scores[1]:.2f} %")
    return scores


# %%
# Read the glove vector
glove_file = os.getcwd() + '\\data\\glove.twitter.27B.50d.txt' 
with open(glove_file, 'r', encoding= 'UTF-8') as file_handler:
    words = set()
    word_to_vec_glove_vector = {}
    for line in file_handler:
        w_line = line.split()
        curr_word = w_line[0]
        word_to_vec_glove_vector[curr_word] = np.array(w_line[1:], dtype=np.float64)
glove_embed_vector_len = word_to_vec_glove_vector['moon'].shape[0]
logging.info("Glove chargé et le vecteur est réalisé.")


# %% [markdown]
# Nous testons différentes solutions pour voir les résultats. En 
# premier nous testons les Word 2 Figure, puis les Word 2 Vec et
# enfin Glove 

# %%
# Loop on the Word 2 Figure
logging.info("Travaille avec les word 2 figure")
data_to_be_tested = [
                        'w2f_clean', 
                        'w2f_stem', 
                        'w2f_lemma',
]
progress_bar = tqdm(
                    total= len(data_to_be_tested),
                    desc= 'Tests models',
                    leave= False
)
memo_score = {
                'data_kind'      : [],
                'model'          : [],
                'parameters'     : [],
                'score'          : [],
                'execution_time' : []
}
for kind_data in data_to_be_tested:
    progress_bar.update()
    print(f"Nous travaillons sur {kind_data}")
    def pad_sentence(sentence, length):
        tmp = [0] * (length - len(sentence)) + [x for x in sentence]
        return np.asarray(tmp, dtype= np.int32)
    X_train = sequence.pad_sequences(
                                        sequences= df_train[kind_data],
                                        maxlen= EMBEDDING_LENGTH
    )
    X_validate = sequence.pad_sequences(
                                        sequences= df_validate[kind_data],
                                        maxlen= EMBEDDING_LENGTH
    )
    X_test = sequence.pad_sequences(
                                    sequences= df_test[kind_data],
                                    maxlen= EMBEDDING_LENGTH
    )
    y_train = df_train['neg'].values
    y_validate = df_validate['neg'].values
    y_test = df_test['neg'].values
    logging.info("X,y sont faits")

    logging.info(f"Simple {kind_data} 128")
    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('Simple')
    memo_score['parameters'].append({'epochs' : 20, 'batch_size' : 128})
    time_in = time.time()
    score = create_simple_NN_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size']
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"Simple {kind_data} 128 fait en {memo_score['execution_time'][-1]}")

    logging.info(f"Simple {kind_data} 32")
    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('Simple')
    memo_score['parameters'].append({'epochs' : 20, 'batch_size' : 32})
    time_in = time.time()
    score = create_simple_NN_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size']
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"Simple {kind_data} 32 fait en {memo_score['execution_time'][-1]}")

    # Compute LSTM
    logging.info(f"LSTM {kind_data} 128")
    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('LSTM')
    memo_score['parameters'].append({'epochs' : 20, 'batch_size' : 128})
    time_in = time.time()
    score = create_LSTM_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        embed_dim= 128,
                        lstm_out= 200, 
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size'],
                        verbose= 4,
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"LSTM {kind_data} 128 fait en {memo_score['execution_time'][-1]}")

    logging.info(f"LSTM {kind_data} 32")
    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('LSTM')
    memo_score['parameters'].append({'epochs' : 20, 'batch_size' : 32})
    time_in = time.time()
    score = create_LSTM_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        embed_dim= 128,
                        lstm_out= 200, 
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size'],
                        verbose= 4,
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"LSTM {kind_data} 32 fait en {memo_score['execution_time'][-1]}")
logging.info("Fin de travail surles word 2 figure")




















# %%
# Loop on the Word 2 Vec
logging.info("Travaille avec les word 2 vec")
data_to_be_tested = [
                        'w2v_CBOW_clean', 
                        'w2v_CBOW_stem',
                        'w2v_CBOW_lemma', 
                        'w2v_SG_clean', 
                        'w2v_SG_stem', 
                        'w2v_SG_lemma',
                        'w2v_Google_clean', 
                        'w2v_Google_stem', 
                        'w2v_Google_lemma'
]
progress_bar = tqdm(
                    total= len(data_to_be_tested),
                    desc= 'Tests models',
                    leave= False
)
for kind_data in data_to_be_tested:
    progress_bar.update()

    print(f"Nous travaillons sur {kind_data}")
    def pad_sentence(sentence, length):
        tmp = [0] * (length - len(sentence)) + [x for x in sentence]
        return np.asarray(tmp, dtype= np.float32).astype(np.float32)
    X_train = np.asarray(
                            [
                                pad_sentence(
                                            sentence= sentence,
                                            length= EMBEDDING_LENGTH
                                )
                                for sentence in df_train[kind_data]
                            ],
                            dtype= np.ndarray
    ).astype(np.float32)
    X_validate = np.asarray(
                            [
                                pad_sentence(
                                                sentence= sentence,
                                                length= EMBEDDING_LENGTH
                                )
                                for sentence in df_validate[kind_data]
                            ],
                            dtype= np.ndarray
    ).astype(np.float32)
    X_test = np.asarray(
                            [
                                pad_sentence(
                                            sentence= sentence,
                                            length= EMBEDDING_LENGTH
                                )
                                for sentence in df_test[kind_data]
                            ],
                            dtype= np.ndarray
    ).astype(np.float32)
    input_dim = len(set(
                        X_train.flatten()
                    ).union(
                            set(
                                X_validate.flatten()
                            ).union(
                                    X_test.flatten()
                            )
                    )
    )
    y_train = df_train['neg'].values
    y_validate = df_validate['neg'].values
    y_test = df_test['neg'].values
    logging.info("X,y sont faits")

    logging.info(f"Simple {kind_data} 128")
    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('Simple')
    memo_score['parameters'].append({'epochs' : 100, 'batch_size' : 128})
    time_in = time.time()
    score = create_simple_NN_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size'],
                        input_dim= EMBEDDING_LENGTH,
                        embedding= False
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"Simple {kind_data} 128 fait en {memo_score['execution_time'][-1]}")

    logging.info(f"Simple {kind_data} 32")
    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('Simple')
    memo_score['parameters'].append({'epochs' : 20, 'batch_size' : 32})
    time_in = time.time()
    score = create_simple_NN_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size'],
                        input_dim= EMBEDDING_LENGTH,
                        embedding= False
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"Simple {kind_data} 32 fait en {memo_score['execution_time'][-1]}")

    # Compute LSTM
    logging.info(f"LSTM {kind_data} 128")
    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('LSTM')
    memo_score['parameters'].append({'epochs' : 20, 'batch_size' : 128})
    time_in = time.time()
    score = create_LSTM_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        embed_dim= 128,
                        lstm_out= 200, 
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size'],
                        verbose= 4,
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"LSTM {kind_data} 128 fait en {memo_score['execution_time'][-1]}")

    logging.info(f"LSTM {kind_data} 32")
    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('LSTM')
    memo_score['parameters'].append({'epochs' : 20, 'batch_size' : 32})
    time_in = time.time()
    score = create_LSTM_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        embed_dim= 128,
                        lstm_out= 200, 
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size'],
                        verbose= 4,
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"LSTM {kind_data} 32 fait en {memo_score['execution_time'][-1]}")

































# %%
# Loop on the tokens for Glove
logging.info("Travaille les Word 2 Vec")
data_to_be_tested = [
                        'clean', 
                        'stem', 
                        'lemma'
]
for data_kind in data_to_be_tested:
    # Create the X_
    # Compute the Glove
    logging.info(f"Glove {kind_data}")
    X_train = df_train[kind_data]
    vocab_len = len(
                    set(
                        [x for x in X_train]
                    ).union(
                            set(
                                X_validate.flatten()
                            ).union(
                                    set(
                                        X_test.flatten()
                                    )
                            )
                    ) 
    )
    emb_matrix = np.zeros((vocab_len, glove_embed_vector_len))
    progress_bar = tqdm(
                        total= len(word_to_vec_glove_vector.item()),
                        desc= 'glove',
                        leave= False
    )
    for word, index in word_to_vec_glove_vector.items():
        progress_bar.update()
        embedding_vector = word_to_vec_glove_vector.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector
    progress_bar.close()

    embedding_layer = Embedding(
                                input_dim= vocab_len,
                                output_dim= glove_embed_vector_len,
                                input_length= EMBEDDING_LENGTH,
                                weights= [emb_matrix],
                                trainable= False
    )

    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('Glove')
    memo_score['parameters'].append({'epochs' : 15, 'batch_size' : 128})
    time_in = time.time()
    score = create_Glove_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size'],
                        input_shape= (None,),
                        embedding_layer= embedding_layer
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)

    memo_score['data_kind'].append(kind_data)
    memo_score['model'].append('Glove')
    memo_score['parameters'].append({'epochs' : 15, 'batch_size' : 32})
    time_in = time.time()
    score = create_Glove_fit_and_display(
                        train= (X_train, y_train),
                        validate= (X_validate, y_validate),
                        test= (X_test, y_test),
                        epochs= memo_score['parameters'][-1]['epochs'],
                        batch_size= memo_score['parameters'][-1]['batch_size'],
                        input_shape= (None,),
                        embedding_layer= embedding_layer,
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)

# %% [markdown]
# Nous avons maintenant une comparaison des différentes solutions. 
# Nous choisissons donc de fabriquer un model de type Glove      






# %%
# Other implementation of Glove
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove2word2vec(
                Path('./data/')/'glove.twitter.27B.50d.txt',
                word2vec_output_file= 
                        Path('./data/')/'glove.twitter.27B.50d.txt.word2vec'
)

w2v_model = glove2word2vec.load_word2vec_format(
                        Path('./data/')/'glove.twitter.27B.50d.txt.word2vec',
                        binary= False
)
score = w2v_model.evaluate(
                            df_test['clean'], df_test['neg']
)
print(f"AUC est de {100 * score[1]:.2f} %")
score = w2v_model.evaluate(
                            df_test['stem'], df_test['neg']
)
print(f"AUC est de {100 * score[1]:.2f} %")
score = w2v_model.evaluate(
                            df_test['lemma'], df_test['neg']
)
print(f"AUC est de {100 * score[1]:.2f} %")

# REgarder Streamlit
# %%