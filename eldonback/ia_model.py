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
    openai.api_key = SECRET_KEY_OPEN_AI
    
    response = openai.Completion.create(
    model="text-babbage-001",
    prompt="Classify the following text with the words positive or negative: \n"+ x_to_classify +"\nCategory:")
    
    msg = response.choices[0].text
    y_predicted = traitement_prediction(msg)
    
    return y_predicted