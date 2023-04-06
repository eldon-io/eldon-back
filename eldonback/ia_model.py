###### Importation des librairies
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from detoxify import Detoxify
import string
from tqdm import tqdm
import nltk
import keras
import openai
from eldonback.secret import SECRET_KEY_OPEN_AI
nltk.download("punkt")
from nltk.stem import PorterStemmer
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


### Traitement de la prédiction renvoyé par l'algorithme gpt-3
def traitement_prediction(prediction):
  msg = prediction.lower().split()
  response = 0

  if 'negative' in msg or 'negative.' in msg:
    response = 1
  
  # pour vérifier 
  # print(msg, response)

  return response

### Prédiction de la nature du message
def algorithm_gpt3(x_to_classify):
    x_to_classify = x_to_classify.replace("_", " ")
    openai.api_key = SECRET_KEY_OPEN_AI
    
    response = openai.Completion.create(
    model="text-babbage-001",
    prompt="Classify the following text with the words positive or negative: \n"+ x_to_classify +"\nCategory:")
    
    msg = response.choices[0].text
    y_predicted = traitement_prediction(msg)
    
    return y_predicted

'''
######  Modèle GPT3
def Model_GPT3(y_to_predict):
    openai.api_key = secret.SECRET_KEY_OPEN_AI
    response = openai.Completion.create(
        model="text-babbage-003",
        prompt="Classify the following text as either positive or negative: \n"+ x_test.iloc[i] +"\nCategory:")
    response = response.choices[0].text.lower()
    if("positiv" in response):
      response = "positive"
    elif ("negativ" in response):
      reponse = "negative"
    elif("neutral" in response):
      response = "neutral"
    else:
      response = openai.Completion.create(
        model="text-babbage-003",
        prompt="Does " + response + " means 'positive', 'negative' or 'neutral' ?")
      response = response.choices[0].text.lower()
      if("positiv" in response):
        response = "positive"
      elif ("negativ" in response):
        reponse = "negative"
      elif("neutral" in response):
        response = "neutral"
      else:
        response = "error : " + response
    y_predicted = response
    
    return y_predicted
'''