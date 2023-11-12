# Imports
from newspaper import Article

# NLTK needs to be downloaded if it isn't part of the env
import nltk
nltk.download('punkt')


# Extracts article from URL
def extract(url):

    # Creates article object
    article = Article(url)

    # Downloads and parses article
    article.download()
    article.parse()

    # Returns
    return article




##########################################
if __name__ == "__main__":

        
    # Example URL
    url = "https://www.reuters.com/world/us/us-house-republicans-eye-plan-avert-government-shutdown-moodys-warns-2023-11-11/"

    # Extracts article from URL
    article = extract(url)

    #print(article.authors)
    #print(article.text)

    print(article.publish_date)
    print(article.title)


    article.nlp()
    #print(article.summary)
    print(article.keywords)