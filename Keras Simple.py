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
import sys
from gensim.models import word2vec
from gensim.parsing.preprocessing import remove_stopwords
from keras.layers import embeddings
from matplotlib.pyplot import ylabel
import missingno as msno
from tensorflow.python.util.nest import _yield_value
from tqdm.notebook import tqdm

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

# %%
# Proceed the DataFrames

# %%
# Clean the tweets
def smileys_to_words(tweet):
    for k,v in SMILEYS.items():
        tweet = tweet.replace(k, v)
    return tweet

for df in [df_train, df_test, df_validate]: 
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
                                r"('(re|d|s|m))",
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

# %%
# Simplest solution.
simple_dictionary = defaultdict(lambda: 0) 
if (
    not(CREATE_SIMPLE_DICTIONARY) 
    and 
    os.path.exists(Path('./data/')/f"{FILE_DICTIONARY}.pkl")
):
    with open(Path('./data/')/f"{FILE_DICTIONARY}.pkl", 'rb') as pickle_file:
        simple_dictionary = pickle.load(pickle_file)
else:
    df_for_simple_dictionry = [df_train]
    progress_bar = tqdm(
                        total= len(df_for_simple_dictionry),
                        desc= 'DataFrame',
                        leave= False
    )
    def create_simple_dictionary(tweet, progress_bar):
        progress_bar.update()
        for token in tweet.split(' '):
            simple_dictionary[token] += 1 
        return True

    for df in df_for_simple_dictionry:
        progress_bar_1 = tqdm(
                                total= len(df_train),
                                desc= 'Dict.',
                                leave= False
        )
        [
            create_simple_dictionary(tweet, progress_bar)
            for tweet in df['clean']
        ]
        progress_bar.update()
        progress_bar_1.close()
    progress_bar.close()
    # Translate into DataFrame to rank the words
    tmp_df = pd.DataFrame(
                            data= simple_dictionary,
                            index= ['Frequency']
    ).T
    # Delete everything that is less than 5 times
    mask = tmp_df['Frequency'] >= 5
    tmp_df = tmp_df.loc[mask]
    # Sort it
    tmp_df.sort_values(
                        by= 'Frequency',
                        ascending= False, 
                        inplace= True
    )
    # Recall 0 is padding, 1 is Start and 2 is unknown
    tmp_df['pos'] = np.arange(3, len(tmp_df) + 3, 1)
    simple_dictionary = {
                tmp_df.index[i] : tmp_df.loc[tmp_df.index[i], 'pos']
                for i in range(len(tmp_df))
    }
    # Save the dictionnary
    with open(Path('./data/')/f"{FILE_DICTIONARY}.pkl", 'wb') as pickle_file:
        pickle.dump(
                    simple_dictionary,
                    pickle_file, 
                    protocol= pickle.HIGHEST_PROTOCOL
        )

# Time to translate the clean sentences into series of numbers
progress_bar = tqdm(
                    total= len([df_train, df_validate, df_test]),
                    desc= 'total',
                    leave= False
)
for df in [df_train, df_validate, df_test]:
    progress_bar.update()
    df['embedding_simple'] = [
                                np.array([
                                    simple_dictionary[token]
                                    if token in simple_dictionary.keys() else 2
                                    for token in sentence.split(' ')
                                ],
                                dtype= np.int32
                                )
                                for sentence in df['clean'].values
    ]
progress_bar.close()

X_train = df_train['embedding_simple'].values
max_length_sentence = max([len(x) for x in X_train])
y_train = np.array(
                    [1 if score == 0 else 0 for score in df_train['score']]
)
X_validate = df_validate['embedding_simple'].values
max_length_sentence = max(
                            max_length_sentence,
                            max([len(x) for x in X_validate])
)
y_validate = np.array(
                    [1 if score == 0 else 0 for score in df_validate['score']]
)
X_test = df_test['embedding_simple'].values
max_length_sentence = max(
                            max_length_sentence,
                            max([len(x) for x in X_test])
)
y_test = np.array(
                    [1 if score == 0 else 0 for score in df_test['score']]
)



# %%
# Transform entries so they all have the same length
max_tweet_length = max_length_sentence
X_train = sequence.pad_sequences(
                                    X_train,
                                    maxlen= max_tweet_length
)
X_validate = sequence.pad_sequences(
                                    X_validate,
                                    maxlen= max_tweet_length
)
X_test = sequence.pad_sequences(
                                X_test,
                                maxlen= max_tweet_length
)

# %%
# Create the neural network
embedding_vector_length = 32
model = Sequential()
model.add(
            Embedding(
                        max(simple_dictionary.values()) + 1,
                        embedding_vector_length,
                        input_length= max_tweet_length
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

# %%
# Fit
hist = model.fit(
                    X_train,
                    y_train,
                    validation_data= (X_validate, y_validate),
                    epochs= 5,
                    batch_size= 128
)

# %%
# Plot the results
acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.plot()

# %%
# Evaluate
scores = model.evaluate(
                        X_test,
                        y_test,
                        verbose= 0
)
print(f"Accuracy est de {100 * scores[1]:.2f} %")

# %%
# Try with epoch of 3
hist = model.fit(
                    X_train,
                    y_train,
                    validation_data= (X_validate, y_validate),
                    epochs= 3,
                    batch_size= 128
)

# %%
# Plot the results
acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.plot()

# %%
# Evaluate
scores = model.evaluate(
                        X_test,
                        y_test,
                        verbose= 0
)
print(f"Accuracy est de {100 * scores[1]:.2f} %")

# %%
