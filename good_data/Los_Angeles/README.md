# File Descriptions

This directory contains various files that are used to build the data necessary for the recommender system.  The files are described as follows

## urls

***urls***:  This is a raw *list* of urls of reviews describing each attraction.  A typical element of the list looks like:

'https://www.tripadvisor.com/Attraction_Review-g32655-d147966-Reviews-The_Getty_Center-Los_Angeles_California.html'
In this file, there is one url per attraction, representing the first page in a series of reviews for each attraction.

## loc_info.pkl
***loc_info.pkl***:  This is a dictionary whose keys are attractions and whose values are lists describing the attraction.  The first element of the list is
an overall rating.  The second element is another list of adjectives describing
the site.  The head of this file looks like:

```python
{'The Getty Center': [5.0, ['Museums', 'Specialty Museums']],
 'Griffith Observatory': [4.5, ['Museums', 'Observatories & Planetariums']],
 'Universal Studios Hollywood': [4.5,
  ['Water & Amusement Parks', 'Theme Parks']],....}
```

## attraction_reviews_urls.pkl

***attraction_reviews_urls.pkl***:  This file contains the complete list of urls_final describing each site so it basically contains **urls**.  The format
is different, however.  **attraction_reviews_urls.pkl** is a dictionary whose
keys are attractions and whose values are the associated list of urls.

## user_ids.pkl

***user_ids.pkl***:  This is just a list of ids for users who have reviewed some
attraction in LA.
