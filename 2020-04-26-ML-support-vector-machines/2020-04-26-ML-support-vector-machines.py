import numpy as np
import re
from sklearn import svm
from sklearn.model_selection import train_test_split

# setup a vocabulary list.

vocablist = [line.strip().split('\t')[-1] for line in open('vocab.txt', 'r', encoding='UTF8')]
vocab = {}
for i in range(len(vocablist)):
    vocab[vocablist[i]] = i

with open('emailSample1.txt', 'r', encoding='UTF8') as email:
    email = email.read()

# Tokenrize email

email = email.lower()
email = re.sub('[0-9]+', 'number', email)
email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
email = re.sub('[$]+', 'dollar', email)
email = re.sub('<[^<>]+>', ' ', email)
email = re.sub('[^0-9a-z]', ' ', email)
email = re.sub('(\s+)', ' ', email).strip()
email = email.split()
email

# Extract feature

wordidx = []
for i in range(len(email)):
    if email[i] in vocablist:
        wordidx.append(vocab[email[i]])

feature = [0 for i in range(len(vocab))]
for i in range(len(wordidx)):
    feature[wordidx[i]] = 1

# Training and testing

x = [line for line in open('X.txt', 'r', encoding='UTF8')]
y = [line for line in open('y.txt', 'r', encoding='UTF8')]

data = []
for i in range(len(y)):
    data.append(x[i].strip() + y[i])

with open('svmSpam.txt', 'w', encoding='UTF8') as spamfile:
    spamfile.writelines(data)

data = np.loadtxt('svmSpam.txt', delimiter=' ')
X = data[:, 0:-1]; y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = svm.SVC()
clf.fit(X_train, y_train)

print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))