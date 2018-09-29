# File Descriptions

## city_scraper.py

***city_scraper.py***: This contains some functions that are useful for scraping TripAdvisor.  Eventually, it would make sense to create a separate directory for each scraped website.

## build_data.py

***build_data.py***: This takes the data from the scraper and converts it to something that an be fed into the neural network.  This function will need to call both `style_mapper.pkl` and `ta_feature_dct.pkl` now both living in ```good_data```

## raw_data

***raw_data***: Contains raw data from each user review.  The format is as follows:

```python
{'user_idn': {'ratings':ratings_dct,'stats':stats_dct}}
```
where ```ratings_dct``` and ```stats_dct``` are dictionaries with the following structure:

```python
ratings_dct={'attraction_name':score, 'another_attraction':score....}

stats_dct={'points':score,'level':score,'readers':score}
```
