# krok 1. pozyskanie słownictwa występującego w naszych danych
import pandas as pd
import nltk
from nltk.corpus import words

vocabulary = {}
data = pd.read_csv('emails_data.csv')
nltk.download('words')
set_words = set(words.words())


def build_vocabulery(curr_email):
    i = len(vocabulary)

    for word in curr_email:
        if word.lower() not in vocabulary and word.lower() in set_words:
            vocabulary[word] = i
            i += 1


if __name__ == "__main__":
    for i in range(data.shape[0]):
        curr_email = data.iloc[i, 0].split()
        build_vocabulery(curr_email,)
        print(f'Current email is {i}/{data.shape[0]} and the length of vocab is {len(vocabulary)}')

    file = open("vocabulary.txt", 'w')
    file.write(str(vocabulary))
    file.close()
