import pickle as pkl
import datetime
import csv
# Scraping items
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
from IPython.core.display import display, HTML
import concurrent.futures
from collections import Counter
import numpy as np
from scipy import stats
import pandas as pd
from subprocess import call
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#Path on Ubuntu
# chromedriver = "/home/williamcottrell72/Downloads/chromedriver_linux64/chromedriver" # path to the chromedriver executable
#Path on Mac
chromedriver="/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)

#In general the template below needs to get replaced for each new city.

sf_attraction_template='https://www.tripadvisor.com/Attractions-g60713-Activities-oa{}-San_Francisco_California.html#FILTERED_LIST'


def make_url_list(sf_attraction_template,pages=25):

    #The inital and other pages are slightly different so we just break these up

    initial_page=sf_attraction_template.format('')
    other_pages=[sf_attraction_template.format(30*i) for i in range(1,pages)]
    all_pages=[initial_page]+other_pages

    urls_final=[]

    for p in all_pages:

        try:
            driver.get(p)
            soup=BeautifulSoup(driver.page_source,'html.parser')
            individual_urls=soup.find_all('div', class_='listing_title')

            for iu in individual_urls:
                partial_url=iu.find('a')['href']
                full_url='https://www.tripadvisor.com'+partial_url
                urls_final.append(full_url)
        except:
            pass

    return urls_final


def make_loc_info(urls_final)
    location_info2={}
    for url in urls_final:
        driver.get(url)
        soup=BeautifulSoup(driver.page_source,'html.parser')
        try:
            rating=soup.find('span',class_='ui_bubble_rating bubble_40')['alt'][0]
        except:
            rating='None'
        try:
            tags=[x.text for x in soup.find('div',class_='detail').find_all('a',href=True)]
        except:
            tags=['None']
        name=soup.find(id='HEADING').text
        location_info2[name]=[rating,tags]
        #Should be inserting into sql.
    return location_info2
