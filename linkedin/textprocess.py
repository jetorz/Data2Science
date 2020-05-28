import nltk
import pandas as pd

jpath = 'sjobs.txt'
raw = open(jpath, encoding='utf8').read().lower()
tokens = sorted(set(nltk.wordpunct_tokenize(raw)))
text = nltk.Text(tokens[tokens.index('a'):])
porter = nltk.PorterStemmer()
stem = [porter.stem(t) for t in text]
stem