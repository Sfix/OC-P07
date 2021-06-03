# %% [markdown]
#
# Notebook d'exploration des données de tweets

# %%
# Import the requiered libraries
import os
import matplotlib
from matplotlib.pyplot import ylabel
import missingno as msno
from tqdm import tqdm

import pandas as pd
import numpy as np
import re
import nltk.stem

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

%matplotlib inline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# %%
# Define constants
FILE_TWEETS = 'sentiment140\\training.1600000.processed.noemoticon.csv'
FILE_DF_TRAIN = 'df_train'
FILE_DF_VALIDATE = 'df_validate'
FILE_DF_TEST = 'df_test'
FILE_DF_BoW = 'df_bow.pkl'

# For the cleaning of the tweets
KEEP_ONLY_ONE_DUPLICATE = True

# %%
path = os.getcwd() + "\\data\\"

df_tweets = pd.read_csv(
                        path + FILE_TWEETS,
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

print(f"Le manque de données par feature est {df_tweets.isna().sum()}.")
print(f"Il y a {df_tweets.duplicated().sum()} ligne(s) dupliquée(s).")
print(f"Alors que sans le score il y a "
+ f"{df_tweets[['id', 'date', 'query', 'user', 'tweet']].duplicated().sum()}"
+ f" twettes dupliqués (même id, horaire, query, user, tweet)"
)
duplicated_tweets = df_tweets[['tweet']].duplicated().sum()
print(f"Pendant qu'en se focalisant sur les tweets il y a "
                            + f"{duplicated_tweets} tweets dupliqués"
)
duplicated_tweets_plus_score = \
                    df_tweets[['sentiment_score', 'tweet']].duplicated().sum()
print(f"et en ne regardant les score et les tweets il y a "
            + f"{duplicated_tweets_plus_score} tweets dupliqués, "
)
print(f"soit un delta de {duplicated_tweets - duplicated_tweets_plus_score}.")
msno.bar(df_tweets)


# %% [markdown]
# A priori nous sommes seulement intéressés par *sentiment_score*,
# et *tweet*. Il faut donc vérifier la cohérence de ces données. En premier nous
# regardons les doublons de tweets. Nous supprimons ceux avec des scores
# contradictoires et agissons comme dit dans les constantes pour les autres.
# Nous détruisons ensuite les autres colonnes.

# %%
# Remove the duplicated tweets. All of them if there are conflicting 
# scores, all but one otherwise. 
mask_duplicates_tweets = df_tweets['tweet'].duplicated(keep= False)
tmp_df = df_tweets.loc[mask_duplicates_tweets, ['sentiment_score', 'tweet']]
list_duplicated_tweets = set(tmp_df['tweet'])
progress_bar = tqdm(
                    total= len(list_duplicated_tweets),
                    desc= 'clear doublons',
                    leave= False
)
# Go through the duplicated tweets
index_of_issues = []
for tweet in list_duplicated_tweets:
    mask = tmp_df['tweet'] == tweet
    # See if we have the same score
    if all(tmp_df.loc[mask, 'sentiment_score']):
        # Same score for all the entries. We act accordingly
        if KEEP_ONLY_ONE_DUPLICATE:
            index_of_issues.append(list(tmp_df.loc[mask].index)[1:])
    else:
        # Two different score for the same entry. We delete them
        index_of_issues.append(list(tmp_df.loc[mask].index))
    tmp_df = tmp_df[~mask]
    progress_bar.update()
progress_bar.close()

print(f"Nous avons trouvé {len(index_of_issues)} duplicatat à détruire.")
mask = [index for index_list in index_of_issues for index in index_list]
print(f"Et ce sont  {len(mask)} tweets qui sont détruits.")
df_tweets.drop(
                index= mask,
                inplace= True
)
number_of_negative = df_tweets['sentiment_score'].value_counts().loc[0]
number_of_positive = df_tweets['sentiment_score'].value_counts().loc[4]
delta= abs(number_of_positive - number_of_negative) / max(
                                                            number_of_negative,
                                                            number_of_positive
                                                    )

print(f"Il reste donc {len(df_tweets)} avec en négatif "
        + f"{number_of_negative} et en positif {number_of_positive}, soit"
        + f" un delta de {100 * delta:.2f} %"
)


# %%
# Check that id can be used as index and then set it as such
mask = df_tweets['id'].duplicated(keep= 'first')
if any(mask):
    print(f"Il reste des id en dipuliqué")
    if KEEP_ONLY_ONE_DUPLICATE:
        df_tweets = df_tweets.loc[~mask]
    else:
        mask = df_tweets['id'].duplicated(keep= False)
        df_tweets = df_tweets.loc[~mask]

# Delete the columns we do not need
df_tweets.drop(
                labels= ['id', 'date', 'query', 'user'],
                axis= 'columns',
                inplace= True
)

tmp_df = df_tweets['sentiment_score'].value_counts()
tmp_df.sort_index(inplace= True)
print(f"Les scores fournis sont donc répartis ainsi :\n{tmp_df}")
fig, ax = plt.subplots()
fig.suptitle("Nombre de Tweets par sentiment")
ax.bar(
        x= range(len(tmp_df.index)),
        height= tmp_df.values,
        color= ["red", "green"],
)
ax.set_xticks(ticks= [0, 1])
ax.set_xticklabels(['Négatif', 'Positif'])
ax.set_ylabel('Total')
plt.show()

print(f"Change la variable pour avoir un échantillon compatible H0.")
std = np.std(df_tweets['sentiment_score'])
mean = np.mean(df_tweets['sentiment_score'])
df_tweets['sentiment_score'] = df_tweets['sentiment_score'].apply(
                                                    lambda x: 1 if x > 0 else 0
)
df_tweets.rename(
                    columns= {'sentiment_score' : 'score'},
                    inplace= True
)
display(df_tweets.head())

# Plot the two word coulds
mask = df_tweets['score'] == 0
wordcloud_negative = WordCloud(max_words= 100,).generate(
                            "\n".join(df_tweets.loc[mask, 'tweet'].values),
)
wordcloud_positive = WordCloud(max_words= 100).generate(
                            "\n".join(df_tweets.loc[~mask, 'tweet'].values),
)
fig, axs = plt.subplots(
                        nrows= 1,
                        ncols= 2,
                        figsize= (9, 2.5)
)
fig.suptitle(
                "Nuage de mots - Tweets bruts",
                fontsize= 14
)
axs[0].set_title("Négatif")
axs[0].axis('off')
axs[0].imshow(
                wordcloud_negative,
                interpolation= 'bilinear'
)
axs[1].set_title("Positif")
axs[1].axis('off')
axs[1].imshow(
                wordcloud_positive,
                interpolation= 'bilinear'
)
plt.show()

# %% [markdown]
# Pour travailler, nous devons maintenant nettoyer les tweets. Nous
# utilisons la fonction de sklearn. Le préprocessing détruit les URL
# les @ quelque chose, les ponctuations y compris le # et la répétition
# des lettres. Nous supprimons aussi les mots qui sont trop souvents
# utilisés et ceux qui ne le sont pas assez. Nous vérifions l'existence
# ou pas des smiley en ASCII  
#  

# %%
# Check ASCII smiley
SMILEYS = [
            ":)", ":-)",
            ":(", ":-(",
            ";)", ";-)",
            ",)", ",-)",
            ";p", ";-p",
            ":p", ":-p",
            ":|", ":-|",
            " :/", ":-/",
            ":'(", ":'-(",
            "<3",
]
progress_bar = tqdm(
                    total= len(SMILEYS),
                    desc= 'check smile',
                    leave= False
)
results = []
for smiley in SMILEYS:
    tmp = [smiley in x for x in df_tweets['tweet']]
    results.append(sum(tmp))
    progress_bar.update()
progress_bar.close()
if sum(results) > 0:
    fig, ax = plt.subplots()
    ax.bar(
            x= SMILEYS,
            height= results
    )
    ax.set_title("Utilisation des smiley")
    plt.show()
else:
    print("Il n'y a pas de smiley.")

# Clean the sentences
def sentence_cleaner(tweet):
    # remove the url
    tweet = re.sub(
                    r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)",
                    '',
                    tweet
    )
    # remove the @____ 
    tweet = re.sub(
                    r"((@)[^ ]*)",
                    '',
                    tweet
    )
    # remove figures
    tweet = re.sub(
                    r"[0-9]*",
                    '',
                    tweet
    )
    # go for lower case
    tweet = tweet.lower()
    # changes "n't" to " not"
    tweet = re.sub(
                    r"(n't)",
                    " not",
                    tweet
    )
    # remove the "'re" "'s" "'d" as they are to be stop words
    tweet = re.sub(
                    r"('(re|d|s))",
                    "",
                    tweet
    )
    # remove ponctuation but not smiley
    tweet = re.sub(
                    r"(([A-Za-z])|((:|,|;)(|-)(\)|\(|\||(p|\/)($|\b| ))|((<3)([a-z]| ))))\1+", 
                    "\1",
                    tweet
    )
    # remove duplicated letters
    tweet = re.sub(
                    r'([a-z])\1+', 
                    r'\1', 
                    tweet
    )
    return tweet

# %%
# Use the sklearn function to make it all at once
stop_words = text.ENGLISH_STOP_WORDS
stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

vectorizer = StemmedCountVectorizer(
                                    input= 'content',
                                    encoding= 'utf-8',
                                    decode_error= 'strict',
                                    strip_accents= 'ascii',
                                    lowercase= True,
                                    preprocessor= sentence_cleaner,
                                    tokenizer= None,
                                    stop_words= stop_words,
                                    token_pattern= r"(?u)\b\w\w+\b",
                                    ngram_range= (1, 1),
                                    analyzer= 'word',
                                    max_df= 0.9,
                                    min_df= 100,
                                    max_features= 2000,
                                    vocabulary= None,
                                    binary= False,
                                    dtype= np.int64,
)
data = vectorizer.fit_transform(df_tweets['tweet'])
df_bow = pd.DataFrame(
                        data= data.toarray(),
                        columns= vectorizer.get_feature_names(),
                        index= df_tweets.index,
                        dtype= np.uint8
)
# Free some memory
del(data)
del(vectorizer)


# %%
# Graphs to understand the data
mask = df_tweets['score'] == 0
tmp_df = df_bow.loc[mask].sum()
tmp_df.sort_values(ascending= False, inplace= True)
top_negative = tmp_df[:100]
wordcloud_negative = WordCloud().fit_words(top_negative)

tmp_df = df_bow.loc[~mask].sum()
tmp_df.sort_values(ascending= False, inplace= True)
top_positive = tmp_df[:100]
wordcloud_positive = WordCloud().fit_words(top_positive)
fig, axs = plt.subplots(
                        nrows= 1,
                        ncols= 2,
                        figsize= (9, 2.5)
)
fig.suptitle(
                "Nuage de mots - Bag of Words",
                fontsize= 14
)
axs[0].set_title("Négatif")
axs[0].axis('off')
axs[0].imshow(
                wordcloud_negative,
                interpolation= 'bilinear'
)
axs[1].set_title("Positif")
axs[1].axis('off')
axs[1].imshow(
                wordcloud_positive,
                interpolation= 'bilinear'
)
plt.show()

fig, ax = plt.subplots()
ax.plot(
        np.arange(20),
        top_negative[:20].values,
)
ax.set_xticks(np.arange(20))
ax.set_xticklabels(
                    top_negative[:20].index,
                    rotation= 50,
)
ax.set_title("Top 20 des négatifs.")
plt.show()

fig, ax = plt.subplots()
ax.plot(
        np.arange(20),
        top_positive[:20].values,
)
ax.set_xticks(np.arange(20))
ax.set_xticklabels(
                    top_positive[:20].index,
                    rotation= 50,
)
ax.set_title("Top 20 des positifs.")
plt.show()

tmp_df = df_bow.sum(axis= 'columns')
mask = tmp_df == 0
tmp_df = df_tweets.loc[mask]
print(f"Nous avons par ailleurs {len(tmp_df)} tweet qui sont à"
        + f" zéro élément dans le bag of words. \n"
        + f"{tmp_df.sample(20)}. Nous les supprimons tous."
)
tmp_df = tmp_df['score'].value_counts()
tmp_df.sort_index(inplace= True)
fig, ax = plt.subplots()
fig.suptitle("Nombre de Tweets absents du bag of words")
ax.bar(
        x= range(len(tmp_df.index)),
        height= tmp_df.values,
        color= ["red", "green"],
)
ax.set_xticks(ticks= [0, 1])
ax.set_xticklabels(['Négatif', 'Positif'])
ax.set_ylabel('Total')
plt.show()


# %% [markdown]
# Nos données sont prètes pour notre segmentation. Il nous faut 
# débuter par la création de 3 paquets : Train, Validate, Test. 
# 
# Train prendra 70% de ce que nous avons et sera équilibré. Train 
# nous servent à entrainer le model.
# Validation prendra 20% de ce que nous avons et sera équilibré.
# Validation nous sert à vérifier que le model est bien entrainé.
# Test prendra les 10% restant de ce que nous avons. Il sert à
# regarder comment notre modèle va fonctionner en production.     

# %%
# Create our bags of samples. COEF is to resuce the size. It must
# be set at 1 when production.
COEF = 1
mask = df_tweets['score'] == 0
half_size_samples_for_TV = min(
                                mask.sum(),
                                len(df_tweets) - mask.sum()
)
number_samples_needed = int(COEF * 0.7 * half_size_samples_for_TV)
index_train = df_tweets.loc[mask].sample(
                                            n= number_samples_needed,
                                            random_state= 23,
                                            replace= False
).index
index_train = index_train.union(
                                df_tweets.loc[~mask].sample(
                                                n= number_samples_needed,
                                                random_state= 23,
                                                replace= False
                    ).index
)
mask = [x not in index_train for x in df_tweets.index]
tmp_df = df_tweets.loc[mask]
mask = tmp_df['score'] == 0
number_samples_needed = int(COEF * 0.2 * half_size_samples_for_TV)
index_validation = tmp_df.loc[mask].sample(
                                            n= number_samples_needed,
                                            random_state= 23,
                                            replace= False
).index
index_validation = index_validation.union(
                                            tmp_df.loc[~mask].sample(
                                                    n= number_samples_needed,
                                                    random_state= 23,
                                                    replace= False
                    ).index
)
mask = [
        not (
                (x in index_train)
                or
                (x in index_validation)
        )
        for x in df_tweets.index
]
index_test = df_tweets.loc[mask].index


# %%
# Save the dataframes
df_bow.to_pickle(path + FILE_DF_BoW)
df_tweets.loc[index_train].to_pickle(path + FILE_DF_TRAIN + f"-{COEF}.pkl")
df_tweets.loc[index_validation].to_pickle(
                                    path + FILE_DF_VALIDATE + f"-{COEF}.pkl"
)
df_tweets.loc[index_test].to_pickle(path + FILE_DF_TEST + f"-{COEF}.pkl")

# %%
