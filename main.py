# krok 3 załadowanie danych i uzycie klasy NaiveBayes do wykonania testu
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from naive_Bayes import NaiveBayes


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X = np.load('X.npy')
y = np.load('y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

precision = precision_score(y_test, predictions, average="binary")
recall = recall_score(y_test, predictions, average="binary")

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
print("Naive Bayes classification precision", precision)
print("Naive Bayes classification recall", recall)

