import re

import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

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

logisticRegressionModel = LogisticRegression(max_iter=1000, random_state=0)
logisticRegressionModel.fit(x_train, y_train)
disp = ConfusionMatrixDisplay.from_estimator(logisticRegressionModel, x_test, y_test, display_labels=['Negative', 'Positive'], cmap='Blues', xticks_rotation='vertical')
plt.show()

preds = logisticRegressionModel.predict(x_test)
print("Logistic regression accuracy score:")
print(accuracy_score(y_test, preds))

textNegative = 'The long time of displaying ads was annoying'
textPositive = 'The movie was great and the service was excellent'

scoreNegative = logisticRegressionModel.predict_proba(tf.transform([textNegative]))[0][1]
print("Logistic regression negative text score:")
print(scoreNegative)
scorePositive = logisticRegressionModel.predict_proba(tf.transform([textPositive]))[0][1]
print("Logistic regression positive text score:")
print(scorePositive)

lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
disp = ConfusionMatrixDisplay.from_estimator(lsvc, x_test, y_test, display_labels=['Negative', 'Positive'], cmap='Blues', xticks_rotation='vertical')
plt.show()

preds = lsvc.predict(x_test)
print("LinearSVC accuracy score:")
print(accuracy_score(y_test, preds))

clf = CalibratedClassifierCV(lsvc)
clf.fit(x_train, y_train)

scoreNegative = clf.predict_proba(tf.transform([textNegative]))[0][1]
print("LinearSVC negative text score:")
print(scoreNegative)
scorePositive = clf.predict_proba(tf.transform([textPositive]))[0][1]
print("LinearSVC positive text score:")
print(scorePositive)

