# %% [markdown]
# Création du Jupyter pour la solution sur étagère.
# 

# %%
# Import library
from dotenv import load_dotenv
from pathlib import Path
import os
import logging
from tqdm.notebook import tqdm
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# %%
# Define the constants
FILE_DF_TEST = 'df_test_'
FILE_DICT_RESULTS = "dict_results_API"
FILE_DF_RESULTS = "df_results_API"
SIZE_TEST = 1_000


# %%
# Read the results
df_results = pd.read_pickle(Path('./data/')/f"{FILE_DF_RESULTS}.pkl")
df_results.set_index(
                        keys= 'index',
                        drop= True,
                        inplace= True 
)
df_test = pd.read_pickle(Path('./data/')/f"{FILE_DF_TEST}{SIZE_TEST}.pkl")

# %%
# Score the results
# Negative well found
mask = df_test['score'] == 0
negative_index = df_test.loc[mask].index
positive_index = df_test.loc[~mask].index

df_results_of_negative = df_results.loc[negative_index]
df_results_of_positive = df_results.loc[positive_index]

negative_negative = df_results_of_negative['sentiment'] == 'negative'
positive_positive = df_results_of_positive['sentiment'] == 'positive'
negative_positive = df_results_of_negative['sentiment'] == 'positive'
negative_neutral = df_results_of_negative['sentiment'] == 'neutral'
negative_mixed = df_results_of_negative['sentiment'] == 'mixed'
positive_negative = df_results_of_positive['sentiment'] == 'negative'
positive_neutral = df_results_of_positive['sentiment'] == 'neutral'
positive_mixed = df_results_of_positive['sentiment'] == 'mixed'

print(f"Nous avons trouvé :\n"
            + f"Défini négatif, trouvé négatif {negative_negative.sum()},\n"
            + f"Défini positif, trouvé positif {positive_positive.sum()},\n"
            + f"Défini négatif, trouvé positif {negative_positive.sum()},\n"
            + f"Défini négatif, trouvé neutral {negative_neutral.sum()},\n"
            + f"Défini négatif, trouvé mixed {negative_mixed.sum()},\n"
            + f"Défini positif, trouvé negatif {positive_negative.sum()},\n"
            + f"Défini positif, trouvé neutral {positive_neutral.sum()},\n"
            + f"Défini positif, trouvé mixed {positive_mixed.sum()},\n"
)

# %%
df_results_of_negative['score_negative'].hist()
df_results_of_negative['score_neutral'].hist()
df_results_of_negative['score_positive'].hist()


# %%
histogram_negative_negative = np.histogram(
                        df_results_of_negative['score_negative'].values, 
                        bins= 100,
                        density= False
)
histogram_negative_neutral = np.histogram(
                        df_results_of_negative['score_neutral'].values, 
                        bins= 100
)
histogram_negative_positive = np.histogram(
                        df_results_of_negative['score_positive'].values, 
                        bins= 100
)
fig, ax = plt.subplots()
ax.plot(
        np.arange(0, 100, 1),
        histogram_negative_negative[0],
        label= "Negative",
        color= "Green"
)
ax.plot(
        np.arange(0, 100, 1),
        histogram_negative_neutral[0],
        label= "Neutral",
        color= "Orange"
)
ax.plot(
        np.arange(0, 100, 1),
        histogram_negative_positive[0],
        label= "Positive",
        color= "Red"
)
fig.legend()
plt.show()
# %%
