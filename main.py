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

@app.post("/analyze/text/")
async def predict_smth(text: Article):
    #store predictions
    detect = []

    #split into processable chunks
    info = text.text.split("\n")

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
        final_score, info, detect, "Custom Input"
    ]

#prediction root
@app.post("/analyze/url/")
async def predict_smth(url: URL):

    #store predictions
    detect = []

    #extract article from url
    try:
        article = extract(url.url)
    except:
        return [
            0, [], [], "Error: Invalid URL"
        ]

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
        final_score, info, detect, article.title
    ]
