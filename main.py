import re

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import csv

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

from wordcloud import WordCloud
from tqdm import tqdm

def cleanText(text: str) -> str:
    text = text.lower()
    text = re.sub('\d', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]', '', text)
    return text

def stemming(wordsList: list) -> list:
    ps = PorterStemmer()
    return [ps.stem(word) for word in wordsList]

def stemmingStr(input: str) -> list:
    wordsList = []
    ps = PorterStemmer()
    for word in [slowo for slowo in re.split('; |, | ', input.lower())]:
        wordsList.append(ps.stem(word))
    return wordsList

def textFiltering(input: list) -> list:
    stopWords = stopwords.words('english')
    return [word for word in input if word not in stopWords and len(word) > 3]

def text_tokenizer(input: str) -> list:
    text = cleanText(input)
    wordsList = text.split(" ")
    wordsList = stemming(wordsList)
    wordsList = textFiltering(wordsList)
    return wordsList

def createBow(words: list) -> dict:
    bow = {}
    for word in words:
        if word not in bow.keys():
            bow[word] = 1
        else:
            bow[word] += 1
    return bow

dataset = pd.read_csv('movie.csv')
print(dataset.head())
dataset.info()
dataset = dataset.drop_duplicates()
print(dataset.groupby('label').describe())

reviews = dataset['text']

txt = ""
for i in tqdm(range(len(reviews))):
    txt += reviews.iloc[i] + " "
stemmed = textFiltering(stemming(cleanText(txt).split()))
bow = createBow(stemmed)
wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

tf = TfidfVectorizer(tokenizer=text_tokenizer)
transform_tf = tf.fit_transform(reviews)

x_train, x_test, y_train, y_test = train_test_split(transform_tf, dataset['label'], test_size=0.5, random_state=0)

model = LogisticRegression(max_iter=1000, random_state=0)
model.fit(x_train, y_train)
disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=['Negative', 'Positive'], cmap='Blues', xticks_rotation='vertical')
plt.show()

text1 = 'The long time of displaying ads was annoying'
score1 = model.predict_proba(tf.transform([text1]))[0][1]
print(score1)

text2 = 'The movie was great and the service was excellent'
score2 = model.predict_proba(tf.transform([text2]))[0][1]
print(score2)
