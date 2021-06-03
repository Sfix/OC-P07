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


# %%
# perform an PCA
sc = StandardScaler()
df_bow_train = sc.fit_transform(df_bow.loc[df_train.index])

pca = PCA(
            n_components= 0.95
)
pca.fit(df_bow_train)
n_dimension = 10
X = pca.transform(df_bow_train)[:,:n_dimension]

# %%
# Analyse the PCA
# Draw the scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'rx-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion de la variance expliquée')
plt.show()

# See how the variance is evolving
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
plt.fill_between(np.arange(len(exp_var_cumul)), exp_var_cumul)
plt.title('Variance cumulée')
plt.show()

# Project on the axes and planes
tmp_df = pd.DataFrame(
                        data= X,
                        columns= [f'PC_{i+1}' for i in range(n_dimension)],
                        index= df_train.index
)
tmp_df['score'] = df_train['score']
# Take only part of it for the plot
tmp_df_sample = tmp_df.sample(
                                n= 1_000,
                                random_state= 23,
                                replace= False,
) 
sns.pairplot(
                data= tmp_df_sample,
                hue= 'score',
                plot_kws= {'alpha' : 0.2},
                height= 4,

)

# %%
# Look at the variance explained vs size of PC
to_explain = [
                5, 10, 25, 50, 
                75, 80, 90, 95, 97.5, 99, 
                None
]
progress_bar = tqdm(
                    total= len(to_explain),
                    desc= 'Compute PCA',
                    leave= False
)
results = []
for percent in to_explain[:-1]:
    pca = PCA(
                n_components= percent / 100,
    )
    pca.fit(df_bow_train)
    results.append(len(pca.components_))
    progress_bar.update()
pca = PCA(n_components= len(df_bow.columns))
pca.fit(df_bow_train)
results.append(len(pca.components_))
progress_bar.update()
progress_bar.close()
plt.plot(
            results,
            to_explain[:-1] + [100],
            'rx-',
            linewidth= 1
)
plt.title(f"Variance expliquée")
plt.xlabel(f"Nombre de features")
plt.ylabel(f"% de variance expliquée")
plt.show()


# %%
# Compute a logistic regression
# define the validations bow to test the training
df_validate = pd.read_pickle(path + FILE_DF_VALIDATE + f"-{COEF}.pkl")
df_bow_validate = df_bow.loc[df_validate.index]
# first model
lr = LogisticRegression()
model = lr.fit(df_bow_train, df_train['score'])
logging.info("Fit de la réfression Logistic réalisée")
score = model.score(df_bow_validate, df_validate['score'])
print(f"Le score avec les paramètre de base est {score}.")

explainer = shap.Explainer(
                            model,
                            df_bow_train,
                            feature_names= df_bow.columns
)
shap_values = explainer(df_bow_validate)
shap.plots.beeswarm(shap_values)#, df_bow_validate, feature_names= df_bow_train.columns)

ind = 5
shap.plots.force(shap_values[ind])
del(model)
del(explainer)

# %%
# LogisticCV (take a sample due to memmory constraints)
sample_coef = 0.1
sample = df_train.sample(int(sample_coef * len(df_train))).index
df_bow_train = pd.DataFrame(
                            data= df_bow_train,
                            columns= df_bow.columns,
                            index= df_train.index
)
clf = LogisticRegressionCV(
                            cv= 5,
                            random_state= 23,
                            n_jobs= -1,
                            verbose= 5,
).fit(df_bow_train.loc[sample], df_train.loc[sample, 'score'])
print(f"Nous avons un score de {clf.score(df_bow_validate, df_validate['score'])}")


# %%
# Optimize the hyper-parameters
param_grid_ = [
                {'C': [1e-5, 2.5E-5, 5E-5]},
                {'C': [6E-5, 7E-5, 7.5E-5]},
                {'C': [8E-5, 9E-5, 1e-4]},
                {'C': [1.1E-4, 1.2E-4, 1.3E-4]},
                {'C': [1.4E-4, 1.5E-4, 2.5E-4]},
                {'C': [5E-4, 1E-3, 1e-2]},
                {'C': [1E-1, 1, 10]},
                {'C': [1E2, 1E3]},
]
results = {
            'score_mean' : [],
            'score_std'  : [],
            'params'     : []
}
progress_bar = tqdm(
                    total= len(param_grid_),
                    desc= 'GridSearchCV',
                    leave= False
)
for iteration in range(len(param_grid_)):
    bow_search = GridSearchCV(
                                LogisticRegression(), 
                                cv=5, 
                                param_grid=param_grid_[iteration],
                                verbose= 1,
                                n_jobs= -1,
    ).fit(df_bow_train, df_train['score'])
    [results['score_mean'].append(x) for x in bow_search.cv_results_['mean_test_score']]
    [results['score_std'].append(x)  for x in bow_search.cv_results_['std_test_score']]
    [results['params'].append(x)     for x in bow_search.cv_results_['params']]
    logging.info(f"{iteration} batch fait")
    progress_bar.update()
progress_bar.close()
df_results = pd.DataFrame(results)
df_results['param'] = [x['C'] for x in df_results['params']]
df_results.sort_values(by= 'param', inplace= True)
fig, ax = plt.subplots()
ax.plot(
        [x['C'] for x in df_results['params']],
        df_results['score_mean'],
)
ax.set_xscale('log')
ax.set_xticks([x['C'] for x in df_results['params']])
plt.show()

# %%
