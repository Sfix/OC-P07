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

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time



# %%
# Define the constants
# For requests : https://docs.microsoft.com/fr-fr/azure/cognitive-services/text-analytics/concepts/data-limits?tabs=version-3
MAX_DOCUMENTS_PER_REQUEST = 10
MAX_SIZE_DOCUMENT = 125_000
# If F0 -> 100 & 300. If S -> 1_000 & 1_000
# MAX_REQUEST_PER_SECOND_F0 = 100
MAX_REQUEST_PER_SECOND  = 100
MAX_REQUEST_PER_MINUTE  = 300

# For data
FILE_DF_TEST = 'df_test_'
FILE_DICT_RESULTS = "dict_results_API"
FILE_DF_RESULTS = "df_results_API"
SIZE_TEST = 1_000


# %%
# Authenticate
# Read the keys
env_path = Path('./data')/'.env'
load_dotenv(dotenv_path= env_path)

# Function do authenticate the client
def authenticate_client():
    ta_credential = AzureKeyCredential(os.getenv("KEY1"))
    text_analytics_client = TextAnalyticsClient(
                                        endpoint= os.getenv("ENDPOINT"),
                                        credential= ta_credential)
    return text_analytics_client

# Authenticate the client
client = authenticate_client()


# %%
# Load the samples for test
df_test = pd.read_pickle(Path('./data/')/f"{FILE_DF_TEST}{SIZE_TEST}.pkl")


# %%
# Loop on the tweets to call the API
# Start by defining the variables
results = {
            'index' : [],
            'response' : [],
            'id' : [],
            'sentiment' : [],
            'warnings' : [],
            'statistics' : [],
            'score_negative' : [],
            'score_neutral' : [],
            'score_positive' : [],
            'sentence' : [],
            'errors' : []
}

index_tweets = list(df_test.index)
current_loop = 0
now = datetime.now()
stack_of_time_per_second = MAX_REQUEST_PER_SECOND * [now]
stack_of_time_per_minute = MAX_REQUEST_PER_MINUTE * [
                    now - timedelta(seconds= 60)
]

progress_bar = tqdm(
                    total= len(df_test),
                    desc= "API",
                    leave= False
)
while (current_loop * MAX_DOCUMENTS_PER_REQUEST) < len(df_test):
    # Wait the 1 second between requests
    while (
            (
                datetime.now() 
                - stack_of_time_per_second[
                                        current_loop % MAX_REQUEST_PER_SECOND
                    ]
            ).seconds < 1
    ):
        must_wait = (
                        1 
                        - (
                            datetime.now() 
                            - stack_of_time_per_second[
                                        current_loop % MAX_REQUEST_PER_SECOND
                                ]
                        ).seconds
        )
        logging.debug(msg= f"Attends la seconde {must_wait}s")
        if must_wait > 0:
            time.sleep(must_wait)

    # Wait the minute between requests
    while (
            (
                datetime.now() 
                - stack_of_time_per_minute[
                                        current_loop % MAX_REQUEST_PER_MINUTE
                    ]
            ).seconds < 60
    ):
        must_wait = (
                        60 
                        - (
                            datetime.now() 
                            - stack_of_time_per_minute[
                                        current_loop % MAX_REQUEST_PER_MINUTE
                                ]
                        ).seconds
        )
        logging.debug(msg= f"Attends la minute : {must_wait}s")
        if must_wait > 0:
            time.sleep(must_wait)

    # We create the documents
    starting_index = current_loop * MAX_DOCUMENTS_PER_REQUEST
    ending_index = (current_loop + 1) * MAX_DOCUMENTS_PER_REQUEST
    mask = index_tweets[starting_index:ending_index]
    documents = df_test.loc[mask, "tweet"].values

    # Make the request
    response = client.analyze_sentiment(list(documents))
    response = response

    for i in range(len(response)):
        results['index'].append(mask[i])
        current_response = response[i]
        results['response'].append(current_response)
        results['id'].append(current_response['id'])
        results['sentiment'].append(current_response['sentiment'])
        results['warnings'].append(current_response['warnings'])
        results['statistics'].append(current_response['statistics'])
        results['score_negative'].append(
                            current_response['confidence_scores']['negative']
        )
        results['score_neutral'].append(
                            current_response['confidence_scores']['neutral']
        )
        results['score_positive'].append(
                            current_response['confidence_scores']['positive']
        )
        tmp_sentences = current_response['sentences']
        value = {
                    'text' : [],
                    'sentiment' : [],
                    'confidence_score_negative' : [],
                    'confidence_score_neutral' : [],
                    'confidence_score_positive' : [],
                    'offset' : [],
                    'mined_opinions' : []
        }
        for j in range(len(tmp_sentences)):
            value['text'].append(tmp_sentences[j]['text'])
            value['sentiment'].append(tmp_sentences[j]['sentiment'])
            value['confidence_score_negative'].append(
                            tmp_sentences[j]['confidence_scores']['negative']
            )
            value['confidence_score_neutral'].append(
                            tmp_sentences[j]['confidence_scores']['neutral']
            )
            value['confidence_score_positive'].append(
                            tmp_sentences[j]['confidence_scores']['positive']
            )
            value['offset'].append(tmp_sentences[j]['offset'])
            value['mined_opinions'].append(tmp_sentences[j]['mined_opinions'])
        results['sentence'].append(value)
        results['errors'].append(current_response['is_error'])



    

    # Update the data
    stack_of_time_per_second[current_loop % MAX_REQUEST_PER_SECOND] = \
                                                                datetime.now()
    stack_of_time_per_minute[current_loop % MAX_REQUEST_PER_MINUTE] = \
                                                                datetime.now()
    
    # Finish the loop.
    current_loop += 1
    progress_bar.update(MAX_DOCUMENTS_PER_REQUEST)
progress_bar.close()

# %%
# Save the results
with open(Path('./data')/f"{FILE_DICT_RESULTS}.pkl", 'wb') as handle:
    pickle.dump(
                results, 
                handle, 
                protocol= pickle.HIGHEST_PROTOCOL,
    )

# %%
# Create dataframe
df_results = pd.DataFrame(
                    data= {
                            key : value for key, value in results.items()
                            if key in [
                                        'index', 
#                                        'response',
                                        'id',
                                        'sentiment',
                                        'score_negative',
                                        'score_neutral',
                                        'score_positive',
                                        'sentence',
                                        'warnings',
                                        'statistics',
                                        'errors'
                                    ]
                    }
)
df_results.to_pickle(Path('./data')/f"{FILE_DF_RESULTS}.pkl")

