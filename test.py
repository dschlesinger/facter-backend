from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer) # cuda = 0,1 based on gpu availability

from newspaper import Article
import nltk
nltk.download('punkt')

def extract(url):

    # Creates article object
    article = Article(url)

    # Downloads and parses article
    article.download()
    article.parse()

    # Returns
    return article

url = "https://www.reuters.com/world/us/us-house-republicans-eye-plan-avert-government-shutdown-moodys-warns-2023-11-11/"

# Extracts article from URL
article = extract(url)

info = article.text.split("\n")

while("" in info):
    info.remove("")

print(info)

for part in info:
    detect = classifier(part)
    print(detect)