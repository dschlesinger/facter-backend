# Imports
from urllib.parse import urlparse
from scipy.spatial import distance
import requests
from bs4 import BeautifulSoup
import re
import spacy

def domain_name_from_url(url):
    '''
    Input: url (ex. https://www.reuters.com/world/us/us-house....)
    Output: just the domain name (ex. reuters.com)
    '''

    # Gets domain name using urlparse
    domain_name = urlparse(url).netloc
    # Removes prefix (like "www.")
    remove_prefix = '.'.join(domain_name.split('.')[-2:])
    # Returns
    return remove_prefix


#nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("en_core_web_md")
def get_text_distance(text1, text2):
    '''
    Input: 2 strings
    Output: cosine similarity between the two strings
    '''

    text1_nlp = nlp(text1)
    text2_nlp = nlp(text2)
    return text1_nlp.similarity(text2_nlp)



def search(query):
    '''
    Input: search query
    Output: a list of links for that search query
    '''
    
    # Gets soup for google search
    page = requests.get(f"https://www.google.com/search?q={query}")
    soup = BeautifulSoup(page.content, features="lxml")

    # Gets main div with all search results
    main = soup.find("div", id="main")

    link_list = []
    try:
        for link in main.find_all("a",href=re.compile("(?<=/url\?q=)(htt.*://.*)")):
            # Cleans links
            clean_link = link['href'].split("&sa=")[0].replace("/url?q=", "")
            # Adds to main list
            link_list.append(clean_link)

        # Removes "support.google.com" and "accounts.google.com" from link list
        link_list = [link for link in link_list if not ("support.google.com" in link or "accounts.google.com" in link)][:5]
    except:
        link_list = []

    # Returns
    print(len(link_list))
    return link_list



####################################################
if __name__ == "__main__":
    print(domain_name_from_url("https://www.reuters.com/world/us/us-house-republicans-eye-plan-avert-government-shutdown-moodys-warns-2023-11-11/"))