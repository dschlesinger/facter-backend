from fastapi import FastAPI

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("mediabiasgroup/roberta_mtl_media_bias")
model = TFAutoModelForSequenceClassification.from_pretrained("mediabiasgroup/roberta_mtl_media_bias")

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

def process(data):
    total_score = 0

    for i in data:
        if i[0]["label"] == "Biased":
            total_score -= i[0]["score"]
        else:
            total_score += i[0]["score"]
    return total_score

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict_smth(url:str):

    detect = []

    article = extract(url)

    info = article.text.split("\n")

    while("" in info):
        info.remove("")

    print(info)

    for part in info:
        detect.append(classifier(part))

        print(f"{part}|{detect}")

    final_score = process(detect)

    return [
        final_score, info
    ]