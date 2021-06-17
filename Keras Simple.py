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
FILE_DF_TRAIN = 'df_train_5000'
FILE_DF_VALIDATE = 'df_validate_1000'
FILE_DF_TEST = 'df_test_1000'

FILE_DICTIONARY = 'dict_simple'
FILE_WORD2VEC_CBOW = "word2vec_CBOW"
FILE_WORD2VEC_SKIPGRAM = "word2vec_Skip_Gram"

# Compare 
COMPARE_SOLUTIONS = True

# When prototype COEF = 0.1. Otherwhize = 1
COEF = 0.1

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
def tweets_cleaner(df, stemming= False, lemmatization= False):
    """Work on the feature "tweet" to clean them (stopwords, lower ...)
    and return the result in a feature "clean"

    Args:
        df (DataFrame): Tweets must be in "tweet"
        stemming (boolean): request to add stemming. If True,
        Lemmatization is not done. Default= False
        lemmatization (boolean): request to add lemmatization.
        Is done only if stemming is at false. Delfault= False

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
                                r'(.)\1{2,}', 
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

    # Add the stemming or lemming depending on what is requiered
    if stemming:
        stemmer = PorterStemmer()
        df['clean'] = stemmer.stem_documents(df['clean'])
    elif lemmatization:
        def lemmatize_sentence(sentence, progress_bar):
            progress_bar.update()
            doc = nlp(sentence)
            return " ".join([token.lemma_ for token in doc])
        progress_bar = tqdm(
                            total= len(df),
                            desc= 'lemma',
                            leave= False
        )
        df['clean'] = [
                        lemmatize_sentence(
                                            sentence,
                                            progress_bar
                        )
                        for sentence in df['clean']
        ]
        progress_bar.close()
    return df

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


# %%
# Create X, y with Word2vec
def create_X_y_Word2Vec_CBOW(df, invert_score= True, new_dictionary= CREATE_SIMPLE_DICTIONARY):
    """Using Word2Vec, this function creates X and y for the DataFrame.
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
    if (
        not(new_dictionary) 
        and 
        os.path.exists(Path('./data/')/f"{FILE_WORD2VEC_CBOW}.model")
    ):
        w2v_model = Word2Vec.load(
                            os.getcwd() + f"\data\{FILE_WORD2VEC_CBOW}.model"
        )
    else:
        # We have to create the Word2Vec. 
        w2v_model = Word2Vec(
                                sentences= [
                                            tweet.split(' ') 
                                            for tweet in df['clean']
                                ],
                                min_count= 1,
                                vector_size= TOP_WORDS,
                                workers= WANTED_WORKERS,
                                window= WORD2VEC_WINDOW,
                                sg= 0                       # CBOW
        )
        w2v_model.save(os.getcwd() + f"\data\{FILE_WORD2VEC_CBOW}.model")
    # Create the vectors
    # Create a set so the work is faster
    w2v_model_vocab = set(w2v_model.wv.index_to_key)
    # create a function so we can see progress_bar
    def sentence_to_w2v(sentence, progress_bar):
        progress_bar.update()
        return [
                np.mean(w2v_model.wv[token]) 
                if token in w2v_model_vocab else 0
                for token in sentence.split(' ')
        ]
    
    progress_bar = tqdm(
                        total= len(df['clean']),
                        desc= 'W2V CBOW',
                        leave= False
    )
    df['w2v_CBOW'] = [
                        sentence_to_w2v(
                                        sentence= sentence, 
                                        progress_bar= progress_bar
                        )
                        for sentence in df['clean']
    ]
    progress_bar.close()
    X = [
            [
                x[i + (len(x) - EMBEDDING_LENGTH)]
                if (i + (len(x) - EMBEDDING_LENGTH)) >= 0 else 0
                for i in range(EMBEDDING_LENGTH)
            ]
            for x in df['w2v_CBOW']
    ]
    if invert_score:
        y = np.array(
                    [1 if score == 0 else 0 for score in df['score']]
        )
    else:
        y = np.array(df['score'].values)    
    return (X, y)



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
                            input_dim= max(train[0].flatten()) + 1,
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
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
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
    model.summary()

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
# Glove 
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

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as glove_file:
        words = set()
        word_to_vec_map = {}
    for line in glove_file:
        w_line = line.split()
        curr_word = w_line[0]
        word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map

def glove_matrix(df, tokenizer, glove_w2v_map):
    vocab_len = len(tokenizer.word_to_index)
    embed_vector_len = glove_w2v_map['moon'].shape[0]

    emb_matrix = np.zeros((vocab_len, embed_vector_len))

    progress_bar = tqdm(
                        total= len(glove_w2v_map.item()),
                        desc= 'glove',
                        leave= False
    )
    for word, index in glove_w2v_map.items():
        progress_bar.update()
        embedding_vector = glove_w2v_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector
    progress_bar.close()
    return Embedding(
                        input_dim= vocab_len,
                        output_dim= embed_vector_len,
                        input_length= EMBEDDING_LENGTH, 
                        weights= [emb_matrix], 
                        trainable=False
    )

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
    model.summary()

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
# First test
progress_bar = tqdm(
                    total= 3,
                    desc= "clean",
                    leave= False
)
for df in [df_train, df_validate, df_test]:
    progress_bar.update()
    df = tweets_cleaner(
                        df= df, 
                        stemming= False, 
                        lemmatization= True
    )
progress_bar.close()


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

# %%
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
                                epochs= 4,
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
                                epochs= 10,
                                batch_size= 32
)

# %%
# Compute LSTM
score = create_LSTM_fit_and_display(
                                    train= (X_train, y_train),
                                    validate= (X_validate, y_validate),
                                    test= (X_test, y_test),
                                    embed_dim= 128,
                                    lstm_out= 200, 
                                    epochs= 10,
                                    batch_size= 128,
                                    verbose= 5,
)

score = create_LSTM_fit_and_display(
                                    train= (X_train, y_train),
                                    validate= (X_validate, y_validate),
                                    test= (X_test, y_test),
                                    embed_dim= 128,
                                    lstm_out= 200, 
                                    epochs= 2,
                                    batch_size= 128,
                                    verbose= 4,
)

score = create_LSTM_fit_and_display(
                                    train= (X_train, y_train),
                                    validate= (X_validate, y_validate),
                                    test= (X_test, y_test),
                                    embed_dim= 128,
                                    lstm_out= 200, 
                                    epochs= 7,
                                    batch_size= 32,
                                    verbose= 4,
)

# %%
# Compute the requiered X, y
(X_train, y_train) = create_X_y_Word2Vec_CBOW(
                                                df= df_train,
                                                invert_score= True,
                                                new_dictionary= True
)
(X_validate, y_validate) = create_X_y_Word2Vec_CBOW(
                                                    df= df_validate,
                                                    invert_score= True,
                                                    new_dictionary= False
)
(X_test, y_test) = create_X_y_Word2Vec_CBOW(
                                            df= df_test,
                                            invert_score= True,
                                            new_dictionary= False
)

# %%
score = create_simple_NN_fit_and_display(
                                train= (X_train, y_train),
                                validate= (X_validate, y_validate),
                                test= (X_test, y_test),
                                epochs= 15,
                                batch_size= 128
)


# %%
# %%
# Glove
word_to_vec_glove_vector = read_glove_vector(
          os.getcwd() + '\\data\\glove.twitter.27B.50d.txt' 
)
tokenizer = word2figure_dict(
                                df= df_train['clean'],
                                file_name_dict= 'Word_2_Figure_Clean.json'
)
embedding_layer_clean = glove_matrix(
                                    df= df_train['clean'],
                                    tokenizer= tokenizer,
                                    glove_w2v_map= word_to_vec_glove_vector
)

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

create_Glove_fit_and_display(
                                train= (X_train, y_train),
                                validate= (X_validate, y_validate),
                                test= (X_test, y_test),
                                epochs= 15,
                                batch_size= 128,
                                input_shape= (None,),
                                embedding_layer= embedding_layer_clean,

)


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
score = fit_and_evaluate(
                            model= w2v_model,
                            test= (X_test, y_test)
)
