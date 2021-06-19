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
from gensim.scripts.glove2word2vec import glove2word2vec
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
from keras.layers import Conv1D, GlobalMaxPool1D 
#from keras.layers.embeddings import Embedding

# %%
# Define constants
FILE_TWEETS = 'sentiment140\\training.1600000.processed.noemoticon.csv'
FILE_DF_TRAIN = 'df_train_5000-Augmented'
FILE_DF_VALIDATE = 'df_validate_1000-Augmented'
FILE_DF_TEST = 'df_test_1000-Augmented'

FILE_DICTIONARY = 'dict_simple'
FILE_WORD2VEC_CBOW = "w2v_CBOW_model_"
FILE_WORD2VEC_SKIPGRAM = "w2v_SG_model_"
FILE_WORD2VEC_GOOGLE= Path('./data/')/"GoogleNews-vectors-negative300.bin.gz"

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
VERBOSE_VALUE = 0

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
                        batch_size= batch_size,
                        verbose= VERBOSE_VALUE
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
                            verbose= VERBOSE_VALUE
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
                        batch_size= batch_size,
                        verbose= VERBOSE_VALUE
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
                            verbose= VERBOSE_VALUE
    )
    print(f"AUC est de {100 * scores[1]:.2f} %")
    return scores

# %%
# Create a LSTM, fit it and display some résults
def create_LSTM_fit_and_display(
                                train, validate, 
                                test, 
                                embed_dim, lstm_out, batch_size,
                                epochs, verbose, input_dim= None, 
                                embedding= True
):
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    if embedding:
        # Embed each integer in a 128-dimensional vector
        x = Embedding(max(train[0].flatten()) + 1, embed_dim)(inputs)
    else:
        x = Embedding(
                        input_dim= input_dim,
                        output_dim= 32,
                        input_length= EMBEDDING_LENGTH
                    )
        x = Flatten()

        x = Embedding(input_dim, embed_dim)(inputs)
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
    # model.summary()

    hist = model.fit(
                        train[0], train[1],
                        validation_data= (validate[0], validate[1]),
                        epochs= epochs,
                        batch_size= batch_size,
                        verbose= VERBOSE_VALUE
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
                            verbose= VERBOSE_VALUE
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
    x = LSTM(64, return_sequences=True)(embeddings) #128
    x = Dropout(0.6)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.6)(x)
    x = LSTM(64)(x)
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
                            verbose= VERBOSE_VALUE
    )
    print(f"AUC est de {100 * scores[1]:.2f} %")
    return scores



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
                        verbose= VERBOSE_VALUE,
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
                        verbose= VERBOSE_VALUE,
    )
    memo_score['execution_time'].append(time.time() - time_in)
    memo_score['score'].append(score)
    logging.info(f"LSTM {kind_data} 32 fait en {memo_score['execution_time'][-1]}")
progress_bar.close()    
logging.info("Fin de travail surles word 2 figure")

















# %%
# Function to turn the w2v vector into the entry for the Neurol network
def build_word_vector(tokens, w2v_model):
    try:
        w2v_model = w2v_model.wv
    except:
        pass
    size = len(w2v_model[w2v_model.index_to_key[0]])
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += w2v_model[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

# %%
# Loop on the Word 2 Vec
logging.info("Travaille direct avec les word 2 vec")
model_to_be_tested = [
                        'CBOW',
                        'Skip-Gram',
                        'Google',
                        'Glove',
]
data_to_be_tested = [
                        'clean', 
                        'stem',
                        'lemma', 
]
progress_bar_model = tqdm(
                            total= len(model_to_be_tested),
                            desc= 'Models',
                            leave= False
)

# Loop on the 2 ways to create Word2Vec
for model in model_to_be_tested:
    logging.info(f"{model}")
    progress_bar_model.update()
    progress_bar = tqdm(
                        total= len(data_to_be_tested),
                        desc= 'Tests models',
                        leave= False
    )
    # We do not need to read twice models that are computed on 
    # generic data
    model_loaded = False
    # Loop on the kind of data we have
    for kind_data in data_to_be_tested:
        logging.info(f"{kind_data}")
        progress_bar.update()
        if model == 'CBOW':
            w2v_model = Word2Vec.load(
                    f"{os.getcwd()}\data\{FILE_WORD2VEC_CBOW}{kind_data}.model"
            )
        elif model == 'Skip-Gram':
            w2v_model = Word2Vec.load(
                    f"{os.getcwd()}\data\{FILE_WORD2VEC_SKIPGRAM}{kind_data}.model"
            )
        elif model == 'Google':
            if not model_loaded:
                w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                                                            FILE_WORD2VEC_GOOGLE,
                                                            binary=True
                )
                model_loaded = True
        else:
            if not model_loaded:
                w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                                Path('./data/')/'glove.twitter.27B.50d.txt',
                                binary= False,
                                no_header= True
                )
                model_loaded = True

        logging.info(f'Le fichier du model w2v {model} pour {kind_data} chargé.')
        try:
            w2v_model = w2v_model.wv
        except:
            pass

        X_train = np.concatenate(
                        [
                            build_word_vector(
                                                tokens, 
                                                w2v_model
                            ) 
                            for tokens in map(
                                                lambda x: x.split(),
                                                df_train[kind_data]
                                            )
                        ]
        )
        X_validate = np.concatenate(
                        [
                            build_word_vector(
                                                tokens, 
                                                w2v_model
                            ) 
                            for tokens in map(
                                                lambda x: x.split(),
                                                df_validate[kind_data]
                                          )
                        ]
        )
        X_test = np.concatenate(
                        [
                            build_word_vector(
                                                tokens, 
                                                w2v_model
                            ) 
                            for tokens in map(
                                                lambda x: x.split(),
                                                df_test[kind_data]
                                            )
                        ]
        )
        y_train = df_train['neg'].values
        y_validate = df_validate['neg'].values
        y_test = df_test['neg'].values
        logging.info("X,y sont faits")

        for batch_size in [128, 32]:
            logging.info(f"Simple {kind_data} {batch_size}")
            memo_score['data_kind'].append(kind_data)
            memo_score['model'].append('Simple')
            memo_score['parameters'].append(
                                            {
                                                'epochs' : 50, 
                                                'batch_size' : batch_size
                                            }
            )
            time_in = time.time()
            score = create_simple_NN_fit_and_display(
                                train= (X_train, y_train),
                                validate= (X_validate, y_validate),
                                test= (X_test, y_test),
                                epochs= memo_score['parameters'][-1]['epochs'],
                                batch_size= memo_score['parameters'][-1]['batch_size'],
                                input_dim= len(w2v_model['love']),
                                embedding= False
            )
            memo_score['execution_time'].append(time.time() - time_in)
            memo_score['score'].append(score)
            logging.info(f"Simple {kind_data} 128 fait en {memo_score['execution_time'][-1]}")
    progress_bar.close()    
progress_bar_model.close()
logging.info('Fin de travail direct avec W2V')








































# %%
# Loop on the Word 2 Vec method LSTM
logging.info("Travaille direct avec les word 2 vec")
model_to_be_tested = [
                        'CBOW',
                        'Skip-Gram',
                        'Google',
                        'Glove'
]
data_to_be_tested = [
                        'clean', 
                        'stem',
                        'lemma', 
]
progress_bar_model = tqdm(
                            total= len(model_to_be_tested),
                            desc= 'Models',
                            leave= False
)

# Loop on the 2 ways to create Word2Vec
for kind_model in model_to_be_tested:
    progress_bar_model.update()
    progress_bar = tqdm(
                        total= len(data_to_be_tested),
                        desc= 'Tests models',
                        leave= False
    )
    # We do not have to load the model when it is not dependent on
    # the data set
    model_loaded = False
    # Loop on the kind of data we have
    for kind_data in data_to_be_tested:
        progress_bar.update()
        if kind_model == 'CBOW':
            w2v_model = Word2Vec.load(
                    f"{os.getcwd()}\data\{FILE_WORD2VEC_CBOW}{kind_data}.model"
            )
        elif kind_model == 'Skip-Gram':
            w2v_model = Word2Vec.load(
                    f"{os.getcwd()}\data\{FILE_WORD2VEC_SKIPGRAM}{kind_data}.model"
            )
        elif kind_model == 'Google':
            if not model_loaded:
                w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                                                            FILE_WORD2VEC_GOOGLE,
                                                            binary=True
                )
            model_loaded = True
        else:
            if not model_loaded:
                w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                        Path('./data/')/'glove.twitter.27B.50d.txt',
                        binary= False,
                        no_header= True
                )
            model_loaded = True

        logging.info(f'Le fichier du model w2v {kind_model} pour {kind_data} chargé.')

        try:
            w2v_model = w2v_model.wv
        except:
            pass
        def tokens_to_figures_of_w2v(tokens, w2v_model):
            tmp = [
                    w2v_model.key_to_index[token] + 3
                    if token in w2v_model.index_to_key else 2
                    for token in tokens
            ] 
            return [0] + tmp + [1]

        tmp = [ 
                tokens_to_figures_of_w2v(
                                            tokens= sentence.split(), 
                                            w2v_model=w2v_model
                )
                for sentence in df_train[kind_data]
        ]
        X_train = sequence.pad_sequences(
                                            sequences= tmp,
                                            maxlen= EMBEDDING_LENGTH,
                                            padding= 'post'
        )
        tmp = [ 
                tokens_to_figures_of_w2v(
                                            tokens= sentence.split(), 
                                            w2v_model=w2v_model
                )
                for sentence in df_validate[kind_data]
        ]
        X_validate = sequence.pad_sequences(
                                            sequences= tmp,
                                            maxlen= EMBEDDING_LENGTH,
                                            padding= 'post'
        )
        tmp = [ 
                tokens_to_figures_of_w2v(
                                            tokens= sentence.split(), 
                                            w2v_model=w2v_model
                )
                for sentence in df_test[kind_data]
        ]
        X_test = sequence.pad_sequences(
                                        sequences= tmp,
                                        maxlen= EMBEDDING_LENGTH,
                                        padding= 'post'
        )
        y_train = df_train['neg'].values
        y_validate = df_validate['neg'].values
        y_test = df_test['neg'].values
        logging.info("X,y sont faits")

        # Compute LSTM
        for batch_size in [128, 32]:
            logging.info(f"LSTM {kind_data} {batch_size}")
            memo_score['data_kind'].append(kind_data)
            memo_score['model'].append('LSTM')
            memo_score['parameters'].append(
                                            {
                                                'epochs' : 20, 
                                                'batch_size' : batch_size
                                            }
            )
            EMBEDDING_DIMENSION = 100
            embedding_matrix = np.zeros(
                                    (
                                    len(w2v_model.index_to_key) + 3,
                                    len(w2v_model['love'])
                                    )
            )
            for word, token in w2v_model.key_to_index.items():
                if w2v_model.__contains__(word):
                    embedding_matrix[token + 3] = w2v_model.__getitem__(word)
            embedding_layer = Embedding(
                                input_dim= len(w2v_model.key_to_index) + 3,
                                output_dim= embedding_matrix.shape[1],
                                weights= [embedding_matrix],
                                input_length= len(X_train[0]),
                                trainable= False
            )

            model = Sequential(
                                layers= [
                                        embedding_layer,
                                        Bidirectional(
                                                LSTM(
                                                        64,
                                                        dropout= 0.3,
                                                        return_sequences= True
                                                )
                                        ),
                                        Bidirectional(
                                                LSTM(
                                                        64,
                                                        dropout= 0.3,
                                                        return_sequences= True
                                                )
                                        ),
                                        Conv1D(
                                                100,
                                                5,
                                                activation= 'relu'
                                        ),
                                        GlobalMaxPool1D(),
                                        Dense(
                                                16, 
                                                activation= 'relu'
                                        ),
                                        Dense(
                                                1, 
                                                activation='sigmoid'
                                        ),
                            ],
                            name= "Sentiment_Model"
            )
            model.compile(
                            loss= 'binary_crossentropy', 
                            optimizer= 'adam', 
                            metrics= ['AUC']
            )
            time_in = time.time()
            history = model.fit(
                                X_train, y_train,
                                validation_data= (X_validate, y_validate),
                                batch_size= memo_score['parameters'][-1]['batch_size'],
                                epochs= memo_score['parameters'][-1]['epochs'],
                                verbose= VERBOSE_VALUE,
                            )
            # Plot the results
            auc = history.history['auc']
            val = history.history['val_auc']
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
                                    X_test, y_test,
                                    verbose= VERBOSE_VALUE
            )
            print(f"AUC est de {100 * scores[1]:.2f} %")

            memo_score['execution_time'].append(time.time() - time_in)
            memo_score['score'].append(scores)
            logging.info(f"LSTM {kind_data} {batch_size} fait en {memo_score['execution_time'][-1]}")
    progress_bar.close()    
progress_bar_model.close()
logging.info("fin des W2V en LSTM")




















# %% [markdown]
# Nous avons maintenant une comparaison des différentes solutions. 
# Nous choisissons donc de fabriquer un model de type Glove      





