# Imports
import numpy as np
# Python imports
from utils import domain_name_from_url
from python_util.model import get_tone,get_bias


class Paragraph:
    def __init__(self, text, article):
        #
        self.text = text
        self.url = article.url
        #self.article = article
        self.domain_name = domain_name_from_url(self.url)
        #
        self.similar_paragraphs = []

    def find_bias(self):
        self.bias = get_bias(self.text)
        self.tone = get_tone(self.text)


class NewsArticle:
    def __init__ (self, article):
        # Gets a list of all paragraphs within the article as a paragraph object
        self.paragraphs = [Paragraph(text=text, article=article) for text in article.text.split("\n") if text != ""]
        
        # Finds bias for each paragraph within the news article
        for paragraph in self.paragraphs:
            paragraph.find_bias()
        
        # Adds other important information
        self.article = article
        self.domain_name = domain_name_from_url(article.url)

        # Calculates mean bias and tone
        self.final_bias_score = np.array([p.bias for p in self.paragraphs]).mean()
        self.final_tone_score = np.array([p.tone for p in self.paragraphs]).mean()