# Imports
from python_util.extract_article import extract
import time
# Python Imports
from utils import search, get_text_distance
from article_structure import Paragraph, NewsArticle


def analyze_article(url):

    # Extracts article
    original_article = NewsArticle(extract(url=url))

    # Searches for other articles with similar titles
    links = search(query=original_article.article.title)

    # Adds all titled articles it can find to a list of article objects
    article_list = []
    for link in links:
        try:
            article = extract(link)
            if article.title and (link not in url):
                article_list.append(article)
        except:
            pass


    # Creates a list of paragraph objects made up of every article combined
    paragraphs = []
    for article in article_list:
        for paragraph in article.text.split("\n"):
            paragraphs.append(Paragraph(text=paragraph, article=article))


    # Finds similar paragraphs for each paragraph
    SIMILARITIY_THRESHOLD = 0.8
    # Finds similar paragraphs
    for original_article_paragraph in [original_article.paragraphs[i] for i in range(len(original_article.paragraphs)) if original_article.paragraphs[i].bias < 0]:
        for p in paragraphs:
            sim = get_text_distance(original_article_paragraph.text, p.text)
            if sim > SIMILARITIY_THRESHOLD:
                original_article_paragraph.similar_paragraphs.append({"similarity_score": sim, "p": p})

        original_article_paragraph.similar_paragraphs.sort(key=lambda x: x["similarity_score"], reverse=True)
        original_article_paragraph.similar_paragraphs = original_article_paragraph.similar_paragraphs[:3]


    # Returns
    return original_article