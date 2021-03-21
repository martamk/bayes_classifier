# krok 2 stworzenie tablic wektorów cech i labelek na podstawie pozyskanego wczesniej słownictw i naszego zbioru danyh
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

data = pd.read_csv('emails_data3.csv')
file = open('vocabulary.txt', 'r')
contents = file.read()
vocab = ast.literal_eval(contents)

X = np.zeros((data.shape[0], len(vocab)))
y = np.zeros((data.shape[0]))

corpus = []
vocabulary = list(vocab.keys())

for i in range(data.shape[0]):
    corpus.append((data.iloc[i, 0]))

pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)), ('tfid', TfidfTransformer())]).fit(corpus)
idf = np.array(pipe['tfid'].idf_)

for i in range(data.shape[0]):
    email = data.iloc[i, 0].split()

    for email_word in email:
        if email_word.lower() in vocab:
            X[i, vocab[email_word]] += 1
            y[i] = data.iloc[i, 1]

X = X * idf[:, np.newaxis]
print(X.shape)
print(len(y))
np.save('X.npy', X)
np.save('y.npy', y)

