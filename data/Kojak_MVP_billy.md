# Project Kojak: MVP Proposal

## Objective

The goal of this project is to develop a travel app.  In particular, I will be focusing on the recommender system which takes historical data from **TripAdviser** and **SFTravel** and predicts which sites a new user would like to visit in San Francisco.  

## Methodology

I plan to use a deep learning based collaborative filter for making recomendations.  The input layer will consist of a sparse array of user rankings of past experiences.  (These are 1-5 ranking off of **TripAdviser**.)  The data is then squashed through a few dense layers and dropout layers before being re-expanded again.  The loss function is chosen to enforce an alignment between the input preferences and the output while masking out the site-user pairs for which there is no data available.

In addition to this, I plan to use descriptive labels for each site (e.g., 'culture', 'museum', 'food') scraped from **SFTravel** in order to build a profile for the travel preferences of each user.  I will need to only focus on those users who have left recommendations for labeled attractions, which will reduce the dataset somewhat.  I will then build a profile vector for each user and then retrain the network with the profile vector added in.  When a new user arrives, he can just input his travel style and the network will should be able to automatically predict attraction preferences even with no past history.  

## Datasets

I will primarily be using **TripAdviser** and **SFTravel**.

## Tools

My main tool here is Pytorch.

## Known Unknowns

The biggest unknown is how well the neural network can be trained with the amount of data that I have available.  Another long-term unknown is how likely people are to use a new travel app.
