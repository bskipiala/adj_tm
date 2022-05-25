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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

from wordcloud import WordCloud
from tqdm import tqdm

def cleanText(input):
    emoticons = re.findall('[;:][^\w|\s|;|:]?[^\w|\s|;|:]', input)
    result = input.lower()
    result = re.sub('\d', '', result)
    result = re.sub('<[^>]*>', '', result)
    result = re.sub('[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]', '', result)
    result = " ".join(result.split())
    for emoticon in emoticons:
        result += emoticon
    return result

def stemming(input: str) -> list:
    wordsList = []
    ps = PorterStemmer()
    for word in [slowo for slowo in re.split('; |, | ', input.lower())]:
        wordsList.append(ps.stem(word))
    return wordsList

def textFiltering(input: list):
    stopWords = stopwords.words('english')
    return [word for word in input if word not in stopWords and len(word) > 3]

def text_tokenizer(input: str) -> list:
    text = cleanText(input)
    wordsList = stemming(text)
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


df = pd.read_csv('reviews.csv')
print(df.head())
df.info()
df = df.drop_duplicates()
print(df.groupby('Sentiment').describe())

vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=20)
x = vectorizer.fit_transform(df['Text'])
y = df['Sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

model = LogisticRegression(max_iter=1000, random_state=0)
model.fit(x_train, y_train)

disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=['Negative', 'Positive'], cmap='Blues', xticks_rotation='vertical')
plt.show()

text1 = 'The long lines and poor customer service really turned me off'
score1 = model.predict_proba(vectorizer.transform([text1]))[0][1]
print(score1)

text2 = 'The food was great and the service was excellent'
score2 = model.predict_proba(vectorizer.transform([text2]))[0][1]
print(score2)




# rating = input['rating']
# verified_reviews = input['verified_reviews']
#
# verified_reviews = " ".join(verified_reviews)
# tokenized_text = text_tokenizer(verified_reviews)
# print(tokenized_text)
#
# bow = createBow(tokenized_text)
# print(bow)
#
# wc = WordCloud()
# wc.generate_from_frequencies(bow)
#
# matplotlib.pyplot.imshow(wc, interpolation='bilinear')
# matplotlib.pyplot.axis("off")
# matplotlib.pyplot.show()
