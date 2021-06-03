# %% [markdown]
#
# Notebook d'exploration des données de tweets

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

from sklearn.feature_extraction import text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import shap
shap.initjs()

import statsmodels.formula.api as smf
import statsmodels.api as sm

# %%
# Define constants
FILE_TWEETS = 'sentiment140\\training.1600000.processed.noemoticon.csv'
FILE_DF_TRAIN = 'df_train'
FILE_DF_VALIDATE = 'df_validate'
FILE_DF_TEST = 'df_test'
FILE_DF_BoW = 'df_bow.pkl'

# When prototype COEF = 0.1. Otherwhize = 1
COEF = 0.1

# Setup the logging
logging.basicConfig(stream= sys.stdout)

# %%
# Read the dataframes
path = os.getcwd() + '\\data\\'

df_train = pd.read_pickle(path + FILE_DF_TRAIN + f"-{COEF}.pkl")
df_bow = pd.read_pickle(path + FILE_DF_BoW)

df_bow_train = df_bow.loc[df_train.index]
df_bow_train['_score_'] = df_train['score']
del(df_bow)

# %%
columns_for_X = list(df_bow_train.columns)
columns_for_X.pop(columns_for_X.index('_score_'))
model = smf.ols(
                formula= "_score_ ~ " \
                            + ' + '.join(columns_for_X[:50]),
                data= df_bow_train 
)
results = model.fit()
print(results.summary())

# %%
CDI - 
But

Skilltofit : lien de rencontre profressionnelle
Faire des rencontres qui n'auraient pas lieu sans eux.
Peur de se tromper. 

Poste de consultant en financement de l'innovation et R&D
Cabinet spécialisé 
Consultant confirmé : double compétence scientifique 
10 personnes. Siége en Alsace en développement en maillage sur 
le territoire. Comptable de l'ensemble des livrable
Veulent développer l'actualité sur le territoire. Donc parfois
création d'une agence physique avec consultant junior pour développer

Poste sur Lyon. Il y avait un consultant. L'entreprise a commencé
sa période d'essais en Novembre. Il a reçu une offre de DAF et il a
du coup mis fin à la période d'essais. 
Accent la période d'essais sur un business développement. Donc devoir 
trouver des nouveaux clients 

80% de productions et 20% de développement commercial
Donc phase de départ purement commercial. 

--------
Aide amont - 
   subvention de la BPI - 
   Nationaux et 
   ADEME
   Européenne <- Une des consultante confirmé est spécialiste du type
   d'aide. Donc pas un besoin. 


 # %%

# %%
