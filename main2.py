import re
import pandas as pd
import matplotlib.pyplot

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

def textFiltering(input: str):
    # text = ' '.join([slowo for slowo in re.split('; |, | ', text.lower()) if slowo not in stop_words])
    # return text
    stopWords = stopwords.words('english')
    wordsList = input.split(" ")
    return [word for word in wordsList if word not in stopWords and len(word) > 3]

def text_tokenizer(input: str) -> list:
    text = cleanText(input)
    text = stemming(text)
    wordsList = textFiltering(text)
    return wordsList

