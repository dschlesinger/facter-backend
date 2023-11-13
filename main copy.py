# Imports
from fastapi import FastAPI
from pydantic import BaseModel
# Python imports
from python_util.extract_article import extract
from python_util.model import get_bias,get_tone
import numpy as np
from analyze_article import analyze_article

# Set input classes
class URL(BaseModel):
    url: str

class Article(BaseModel):
    text: str

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

        #split into processable chunks
        info = text.text.split("\n")

        #removes blanks
        while("" in info):
            info.remove("")

        #for debuging extract
        #print(info)

        #run model over each chunk
        for part in info:
            detect_bias.append(get_bias(part))
            detect_tone.append(get_tone(part))

            print(f"{part}|{detect_bias}")


        final_bias = np.array(detect_bias).mean()
        final_tone_score = np.array(detect_tone).mean()

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
    try:
        #store predictions
        detect_bias = []
        detect_tone = []

        #extract article from url
        try:
            article = analyze_article(url.url)
        except:
            return [
            0, [], [], "Error: Invalid URL"
            ]

        #split into processable chunks
        info = article.paragraphs

        if len(info) == 0:
                return [
                0, [], [], "Error: No article found for the URL"
            ]

        #for debuging extract
        #print(info)

        #run model over each chunk
        for part in info:
            detect_bias.append(get_bias(part))
            detect_tone.append(get_tone(part))

            print(f"{part}|{tone_analyzer(part)}")

        final_bias = np.array(detect_bias).mean()
        final_tone_score = np.array(detect_tone).mean()

        #returns final score + list for highlights
        return [
            final_score, info, detect_bias, detect_tone, final_tone_score, article.title
        ]
    except:
        return [
                    0, [], [], "Error: Server error, try again"
             ]
