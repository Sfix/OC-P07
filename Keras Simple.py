# %% [markdown]
#
# Notebook du traitement des données par Keras
# https://docs.microsoft.com/fr-fr/learn/modules/analyze-review-sentiment-with-keras/2-build-and-train-a-neural-network
#  

# %%
# Import the requiered libraries
import os
import sys
import logging
from matplotlib.pyplot import ylabel
import missingno as msno
from tqdm import tqdm

import pandas as pd
import numpy as np
import re
import nltk.stem
import string


from sklearn.feature_extraction import text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding

# %%
# Define constants
FILE_TWEETS = 'sentiment140\\training.1600000.processed.noemoticon.csv'
FILE_DF_TRAIN = 'df_train'
FILE_DF_VALIDATE = 'df_validate'
FILE_DF_TEST = 'df_test'
FILE_DF_BoW = 'df_bow.pkl'

# When prototype COEF = 0.1. Otherwhize = 1
COEF = 0.1

# Dictionary size
TOP_WORDS = 10_000

# Setup the logging
logging.basicConfig(stream= sys.stdout)

# %%
# Read the dataframes
path = os.getcwd() + '\\data\\'

df_tweets = pd.read_csv(
                        path + FILE_TWEETS,
                        names= [
                                    'score',
                                    'id',
                                    'date',
                                    'query',
                                    'user',
                                    'tweet'
                        ]
)

df_tweets = df_tweets.loc[:, ['score', 'tweet']]
# Transform the scores as 0 and 1
df_tweets['score'] = df_tweets['score'].apply(lambda x: 1 if x else 0)



# %%
# Create the tensors
def analyze(text):
    # Prepare the input by removing punctuation characters, converting
    # characters to lower case, and removing words containing numbers
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower().split(' ')
    text = [word for word in text if word.isalpha()]

    # Generate an input tensor
    input = [1]
    for word in text:
        if word in word_dict and word_dict[word] < top_words:
            input.append(word_dict[word])
        else:
            input.append(2)
    padded_input = sequence.pad_sequences([input], maxlen=max_review_length)

    # Invoke the model and return the result
    result = model.predict(np.array([padded_input][0]))[0][0]
    return result

# %%
# Test it
analyze('Easily the most stellar experience I have ever had.')
analyze('Easily the most #stellar experience I have ever had.')

# %%
# Give the same length to all the tensors
max_tweets_length = max([len(x) for x in df_tweets['tensors']])
df_tweets['tensors'] = sequence.pad_sequences(
                                        sequences= df_tweets['tensors'],
                                        maxlen= max_tweets_length
)

# %%
# Create the Training samples, Validation samples and Test samples.


# %% [markdown]
# Les données sont prètes. Nous travaillons sur le réseau de neurones.

# %%
# Train
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(
                    TOP_WORDS, 
                    embedding_vector_length, 
                    input_length= max_tweets_length))
model.add(Flatten())
model.add(Dense(
                16, 
                activation='relu'
        )
)
model.add(Dense(
                16, 
                activation='relu'
        )
)
model.add(Dense(
                1, 
                activation='sigmoid'
        )
)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())