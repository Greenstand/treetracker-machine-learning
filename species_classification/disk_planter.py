# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:07:48 2020

@author: A-Kees.Brekelmans
"""


from bs4 import BeautifulSoup
import requests
import urllib
import os
import imghdr 

#word_to_id = {"Willow" : "n12725940", "Bonsai" : "n13112035"}



def id_to_urls(word_net_id):
    """Accepts a word net id and returns the related urls to the images as list"""
    
    page = requests.get(f"http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={word_net_id}")    
    # BeautifulSoup is an HTML parsing library
    
    soup = BeautifulSoup(page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line
    
    str_soup=str(soup)#convert soup to string so it can be split

    split_urls=str_soup.split('\r\n')#split so each url is a different possition on a list
    
    return split_urls
    
def urls_to_dir(word, urls):
    """Accepts the category word and a list of urls. Creates a directory if it
    does not exists and writes images to the directory."""
    if not os.path.exists(word):
        print(f"Creating directory for {word}")
        os.makedirs(word)
    
    path = os.path.join(os.getcwd(), word)
    
    for c, url in enumerate(urls):
        try:
            resource = urllib.request.urlopen(url)
            output = open(os.path.join(path, f"{c}.jpg"),"wb")
            output.write(resource.read())
            output.close()

        except Exception as exc:
            print(f"Exception occured while downloading image from url {url} {str(exc)}")
    print(f"Finished writing {word} images. Total of {len(list(os.listdir(path)))} images.")
        
def word_to_dir(word_net_dic):
    """Accepts a dictionary with words and related word_net_ids, fetches image urls
    and writes images to a category word id."""
    for word, word_net_id in zip(word_net_dic.keys(), word_net_dic.values()):
        print(f"Fetching URLS for {word}")
        urls = id_to_urls(word_net_id)
        
        urls_to_dir(word, urls)
        
#word_to_urls(word_to_id)        
        
        
    
    