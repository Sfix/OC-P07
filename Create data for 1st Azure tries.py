# %% [markdown]
#
# Notebook permettant de création de dataset de petite taille pour
# les expérimentations Azure  




# %%
# Import the requiered libraries
import os
import sys
import logging
from matplotlib.pyplot import ylabel
import missingno as msno
from tqdm import tqdm

import pandas as pd

# %%
# Define constants
FILE_TWEETS = 'sentiment140\\training.1600000.processed.noemoticon.csv'
SIZE_TRAIN = 2 * 2500
FILE_DF_TRAIN = f"df_train_{SIZE_TRAIN}"
SIZE_VALIDATE = 2 * 500
FILE_DF_VALIDATE = f"df_validate_{SIZE_VALIDATE}"
SIZE_TEST = 2 * 500
FILE_DF_TEST = f"df_test_{SIZE_TEST}"

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
logging.info("Les tweets sont chargés")

mask = df_tweets['score'] == 0
tmp_df = df_tweets.loc[mask]
train_index = list(tmp_df.sample(
                                    n = int(SIZE_TRAIN / 2),
                                    replace= False
                    ).index
)
mask = [x not in train_index for x in tmp_df.index]
tmp_df = tmp_df.loc[mask]
validate_index = list(tmp_df.sample(
                                    n = int(SIZE_VALIDATE / 2),
                                    replace= False
                        ).index
)
mask = [x not in validate_index for x in tmp_df.index]
tmp_df = tmp_df.loc[mask]
test_index = list(tmp_df.sample(
                                n = int(SIZE_TEST / 2),
                                replace= False
                    ).index
)
logging.info("Les samples négatifs sont sélectionnés")

mask = df_tweets['score'] == 1
tmp_df = df_tweets.loc[mask]
[
    train_index.append(x)
    for x in tmp_df.sample(
                                n = int(SIZE_TRAIN / 2),
                                replace= False
            ).index
]
mask = [x not in train_index for x in tmp_df.index]
tmp_df = tmp_df.loc[mask]
[
    validate_index.append(x)
    for x in tmp_df.sample(
                                n = int(SIZE_VALIDATE / 2),
                                replace= False
            ).index
]
mask = [x not in validate_index for x in tmp_df.index]
tmp_df = tmp_df.loc[mask]
[
    test_index.append(x)
    for x in tmp_df.sample(
                    n = int(SIZE_TEST / 2),
                    replace= False
            ).index
]
logging.info("Les samples positifs sont séletionnés.")

df_tweets.loc[train_index].to_csv(path + FILE_DF_TRAIN + ".csv", sep= "\t")
df_tweets.loc[train_index].to_pickle(path + FILE_DF_TRAIN + ".pkl")
df_tweets.loc[validate_index].to_csv(path + FILE_DF_VALIDATE + ".csv", sep= "\t")
df_tweets.loc[validate_index].to_pickle(path + FILE_DF_VALIDATE + ".pkl")
df_tweets.loc[test_index].to_csv(path + FILE_DF_TEST + ".csv", sep= "\t")
df_tweets.loc[test_index].to_pickle(path + FILE_DF_TEST + ".pkl")


# %%

