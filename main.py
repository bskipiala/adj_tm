import re
from nltk.tokenize import sent_tokenize, word_tokenize


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

text = "<div><h2>  :)  Header333,,  </h2> <p>article1<b>strong   ;-)   text2!</b> <a href="">link     </a>:(</p></div>"
cleanedText = cleanText(text)
print(cleanedText)

