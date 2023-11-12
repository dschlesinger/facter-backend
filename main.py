# Imports
from fastapi import FastAPI
from pydantic import BaseModel
# Python imports
from python_util.extract_article import extract
from python_util.model import get_classifier,get_tone_analysis,get_tone_screen
import numpy as np

# Set input classes
class URL(BaseModel):
    url: str

class Article(BaseModel):
    text: str

# Creates classifier object
classifier = get_classifier()

tone_analyzer =  get_tone_analysis()

tone_screener =  get_tone_screen()

#Finds mean score based on model predictions
def process_bias(data):
    total_score = 0

    for i in data:
        if i[0]["label"] == "Biased":
            total_score -= i[0]["score"]
        else:
            total_score += i[0]["score"]
    return total_score/len(data)

def process_tone(data):
    total_score = 0

    for i in data:
        if i[0]["label"] == "NEGATIVE":
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
    try:
        #store predictions
        detect_bias = []
        detect_tone = []
        detect_screen = []

        #split into processable chunks
        info = text.text.split("\n")

        #removes blanks
        while("" in info):
            info.remove("")

        #for debuging extract
        #print(info)

        #run model over each chunk
        for part in info:
            detect_bias.append(classifier(part))
            detect_tone.append(tone_analyzer(part))

            print(f"{part}|{detect_bias}")


        #generates final score
        final_score = process_bias(detect_bias)

        final_tone_score = process_tone(detect_tone)

        #returns final score + list for highlights
        return [
            final_score, info, detect_bias, detect_tone, final_tone_score, "Custom Input"
        ]
    except:
        return [
                0, [], [], "Error: Server error, try again"
            ]

#prediction root
@app.post("/analyze/url/")
async def predict_smth(url: URL):
    #try:
    #store predictions
    detect_bias = []
    detect_tone = []
    detect_screen = []

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

    if len(info) == 0:
            return [
            0, [], [], "Error: No article found for the URL"
        ]

    #for debuging extract
    #print(info)

    #run model over each chunk
    for part in info:
        detect_bias.append(classifier(part))
        detect_tone.append(tone_analyzer(part))

        print(f"{part}|{tone_analyzer(part)}")

    #generates final score
    final_score = process_bias(detect_bias)

    final_tone_score = process_tone(detect_tone)

    #returns final score + list for highlights
    return [
        final_score, info, detect_bias, detect_tone, final_tone_score, article.title
    ]
    # except:
    #     return [
    #                 0, [], [], "Error: Server error, try again"
    #             ]
