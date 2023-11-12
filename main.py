# Imports
from fastapi import FastAPI
from pydantic import BaseModel
# Python imports
from python_util.extract_article import extract
from python_util.model import get_classifier

# Set input classes
class URL(BaseModel):
    url: str

class Article(BaseModel):
    text: str

# Creates classifier object
classifier = get_classifier()

#Finds mean score based on model predictions
def process(data):
    total_score = 0

    for i in data:
        if i[0]["label"] == "Biased":
            total_score -= i[0]["score"]
        else:
            total_score += i[0]["score"]
    return total_score/len(data)

#Init fast api
app = FastAPI()

#Root directory
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/text/")
async def predict_smth(text: Article):
    score = classifier(text.text)

    print(score)

    if score[0]["label"] == "Biased":
        score = -score[0]["score"]
    else:
        score = score[0]["score"]

    #returns final score + list for highlights
    return [
        score, text
    ]

#prediction root
@app.post("/predict/url/")
async def predict_smth(url: URL):

    #store predictions
    detect = []

    #extract article from url
    article = extract(url.url)

    #split into processable chunks
    info = article.text.split("\n")

    #removes blanks
    while("" in info):
        info.remove("")

    #for debuging extract
    #print(info)

    #run model over each chunk
    for part in info:
        detect.append(classifier(part))

        print(f"{part}|{detect}")


    #generates final score
    final_score = process(detect)


    #returns final score + list for highlights
    return [
        final_score, detect
    ]