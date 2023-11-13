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
            final_bias, info, detect_bias, detect_tone, final_tone_score, "Custom Input"
        ]
    except:
        return [
                0, [], [], "Error: Server error, try again"
            ]

#prediction root
@app.post("/analyze/url/")
async def predict_smth(url: URL):
    #try:
    #extract article from url
    #try:
    article = analyze_article(url.url)
    # except:
    #     return [
    #     0, [], [], "Error: Invalid URL"
    #     ]

    # Checks if it can find an article
    if len(article.paragraphs) == 0:
            return [
            0, [], [], "Error: No article found for the URL"
        ]

    #returns final score + list for highlights
    return [
        # Bias and tone info (as well as paragraphs)
        article.final_bias_score, 
        [p.text for p in article.paragraphs], 
        [p.bias for p in article.paragraphs], 
        [p.tone for p in article.paragraphs], 
        article.final_tone_score, 

        # Article info
        article.article.title,

        # Info on similar articles
        [p.similar_paragraphs[0]["p"].text for p in article.paragraphs if len(p.similar_paragraphs) > 0], # Most similar paragraph for each paragraph 
        [[a["p"].domain_name for a in p.similar_paragraphs] for p in article.paragraphs if len(p.similar_paragraphs) > 0], # All domain names for each paragraph
        [[a["p"].url for a in p.similar_paragraphs] for p in article.paragraphs if len(p.similar_paragraphs) > 0], # All urls for each paragraph




    ]
    # except:
    #     return [
    #                 0, [], [], "Error: Server error, try again"
    #          ]
