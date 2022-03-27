import re
import pandas as pd
import matplotlib.pyplot

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from wordcloud import WordCloud
from tqdm import tqdm

def stemming(text: str) -> list:
    wordsList = []
    ps = PorterStemmer()
    for word in [slowo for slowo in re.split('; |, | ', text.lower())]:
        wordsList.append(ps.stem(word))
    return wordsList

def text_tokenizer(text: str) -> list:
