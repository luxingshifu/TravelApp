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
la_attraction_template='https://www.tripadvisor.com/Attractions-g32655-Activities-oa{}-Los_Angeles_California.html#FILTERED_LIST'

cwd=os.getcwd()
approot=cwd.strip('preprocessing')



def make_url_list(sf_attraction_template,offset=0,pages=25):

    #The inital and other pages are slightly different so we just break these up

    initial_page=sf_attraction_template.format('')
    other_pages=[sf_attraction_template.format(30*i+offset) for i in range(1,pages)]
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
    pkl.dump(urls_final,open(approot+'good_data/Los_Angeles/urls','wb'))
    return urls_final

#Output of above function goes into the next function to build a place profiles.


def make_loc_info(urls_final):
    location_info2={}
    ct=0
    for url in urls_final:
        print(f"Process is {round(100*ct/len(urls_final),3)} % through",end="\r")
        ct+=1
        driver.get(url)
        soup=BeautifulSoup(driver.page_source,'html.parser')
        try:
            rating=float(soup.find('span',class_='overallRating').text)
        except:
            rating='None'
        try:
            tags=[x.text for x in soup.find('div',class_='detail').find_all('a',href=True)]
        except:
            tags=['None']
        name=soup.find(id='HEADING').text
        location_info2[name]=[rating,tags]
        #Should be inserting into sql.
    pkl.dump(location_info2,open(approot+'good_data/Los_Angeles/loc_info.pkl','wb'))

    return location_info2

#The following function gets all the reviews for each attraction by going
#through all the pages assoicated with an attraction.  We really only need
#to scrape the number of pages associated with each attraction, the rest
#can be done with the template.

def make_attraction_reviews(urls):
    tail='-'+urls[1].split('-')[-1]
    attraction_reviews={}
    ct=0
    for url in urls:
        print(ct)
        ct+=1
        try:
            driver.get(url)
            soup=BeautifulSoup(driver.page_source,'html.parser')
            url_list=url.split('-')
            name='-'.join(url_list[-2:])
            template='-'.join(url_list[:-2])+'-or{}-'+name
            last_num=soup.find(class_="pageNum last taLnk ")
            last_page=int(last_num['data-page-number'])
            review_urls=[template.format('')]
            for i in range(1,last_page):
                review_urls.append(template.format(i*10))
            attraction_reviews[name.strip(tail)]=review_urls
        except:
            pass
    return attraction_reviews


#Finally, we need a list of the individual users who have reviewed a given city.

def make_user_ids(attraction_reviews):
    user_ids=[]
    keys=list(attraction_reviews.keys())
    L=len(keys)
    ct=0
    for k in keys:
        ct+=1
        print(f"Process is {round(100*ct/L,3)} % through",end="\r")
        urls=attraction_reviews[k]
        for url in urls:
            try:
                driver.get(url)
                soup=BeautifulSoup(driver.page_source,'html.parser')
                ids=soup.find_all(class_="info_text")
                for ID in ids:
                    t1=ID.text
                    t2=ID.find('strong').text
                    t3=t1.strip(t2)
                    user_ids.append(t3)
            except:
                pass
    return user_ids

def scrape_reviews(urls):
    count=0
    tmp={}
    users={}
    ct=0
    for url in urls:
        print(f"Process is {round(100*ct/len(urls),3)} % through",end="\r")
        ct+=1
        name=url[36:]
        driver.get(url)
        soup=BeautifulSoup(driver.page_source,'html.parser')
#         pages=int(soup.find_all(class_="cs-paginate-goto")[-1].text)
        reviews=soup.find_all('li',class_='cs-review')
        ranks={}
        for review in reviews:

            try:
                attraction_name=review.find_all(class_='cs-review-location')[0].text
                rating=int([str(tag) for tag in review.find_all()][-3][-10:-9])
            except:
                pass
            try:
                points=int(soup.find(class_="points").text.strip(' ').replace(',',''))
            except:
                points='None'
            try:
                level=int(soup.find(class_="level tripcollectiveinfo").text.split(' ')[1])
            except:
                level='None'
            try:
                readers=int(soup.find(class_="currentBadge").find(class_="badgeSubtext").text.split(' ')[1].replace(',',''))
            except:
                readers='None'
            try:
                stats={'points':points,'level':level,'readers':readers}
            except:
                stats={}

            if 'Los Angeles:' == attraction_name[:12]:
                ranks[attraction_name]=rating

        if len(ranks)!=0:
            users[name]={'ratings':ranks,'stats':stats}
            tmp[name]={'ratings':ranks,'stats':stats}
            count+=1
        if count%100==0:
            with open(approot+f'good_data/Los_Angeles/file_{count}.pkl','wb') as f:
                pkl.dump(tmp,f)
            print(count)
            tmp={}

    return users
