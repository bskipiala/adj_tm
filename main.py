import re
import pandas as pd
import matplotlib.pyplot
import sklearn

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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

input = 'This is the first document This document is the second document And this is the third one Is this the first document'
vectorizer = CountVectorizer()
tokenized_text = text_tokenizer(input)
print(tokenized_text)
X_transform = vectorizer.fit_transform(tokenized_text)
print(X_transform.toarray())

