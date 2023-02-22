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
import secret
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
nltk.download("punkt")
from nltk.stem import PorterStemmer
nltk.download('omw-1.4')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

###### Pré traitement
def normalize_text(mot):
  # Tokenize : I'm hiring right now --> ['I', ''m', 'hiring', 'right', 'now']
  mot = word_tokenize(mot)
  # Steeming : ['I', 'hiring', 'right', 'now'] --> ['i', 'hire', 'right', 'now']
  ps = PorterStemmer()
  mot = [ps.stem(i) for i in mot]
  return mot


def preTraitementRandomForest(dataset):
    # Remove any non-non normal letter : ['I', ''m', 'hiring', 'right', 'now'] --> ['I', 'hiring', 'right', 'now']
    dataset.text = dataset.text.str.lower().replace('[^a-zA-Z\n ]', '', regex=True).replace('[\n]', ' ', regex=True).replace('  ', ' ', regex=True)
    for i in tqdm(range(len(dataset))):
        dataset.text.iloc[i] = normalize_text(df.text.iloc[i])
    dataset.text.str.split("").explode().unique()

    x_train, x_test, y_train, y_test = train_test_split(dataset.text, dataset.label, test_size=0.2, random_state=0, shuffle=True)
    
    # Classe qui va permettre de tokenizer un text
    # num_words = nombre maximal de mots à conserver 
    # oov_token = ajout à word_index pour remplacer les mots hors vocabulaire pendant les appels texts_to_sequence
    tokenizer = Tokenizer(num_words=50000,oov_token="unk")

    # Mise à jour du vocabulaire interne en fonction d'une liste de textes
    tokenizer.fit_on_texts(x_train)

    # On transforme chaque mot du texte en une séquence d’entiers
    x_train = np.array(tokenizer.texts_to_sequences(x_train) )
    x_test = np.array(tokenizer.texts_to_sequences(x_test) )
    
    # On structure nos séquences d'entiers sous la forme d'une matrice carré
    x_train = pad_sequences(x_train, padding='post', maxlen=400)
    x_test = pad_sequences(x_test, padding='post', maxlen=400)

    return (x_train, x_test, y_train, y_test)


###### Modèle Random Forest
def Model_randomForest(dataset, k, y_to_predict):
    (x_train, x_test, y_train, y_test) = preTraitementRandomForest(dataset)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    
    # Utilisation de Random Forest
    rfc = RandomForestClassifier(n_estimators=k) #, max_depth=5, min_samples_split=10, min_samples_leaf=5)
    # On entraine notre modèle
    rfc.fit(x_train, y_train)
    # On récupère les prédictions effectué à partir du modèle sur le train set 
    y_predicted = rfc.predict(y_to_predict)
    # On calcule la précision et le score f1
    # p, f = accuracy_score(y_test, y_to_predict), f1_score(y_test, y_to_predict)
    # print(p, " ", f)
    
    return y_predicted


######  Modèle GPT3
def Model_GPT3(y_to_predict):
    openai.api_key = secret.SECRET_KEY_OPEN_AI
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Classify the following text as either positive or negative: \n"+ x_test.iloc[i] +"\nCategory:")
    y_predicted = response.choices[0].text
    
    return y_predicted