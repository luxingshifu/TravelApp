
import numpy as np
import pickle as pkl
import os
import math
from collections import Counter
import random
import recsys
import torch
import torch.nn as nn

#from google.appengine.api import app_identity
# use the following line to load from bucket.
# import google.cloud.storage
#
# import logging
# #import webapp2
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
# storage_client = google.cloud.storage.Client("TravelApp")

# def create_recsys(matrix,dropout=.1,latent_features=4,max_iter=10,lr=.001,epochs=3,temperature=1,batch_size=50):
#     return recsys.recsys(matrix,len(matrix),len(matrix[0]),latent_features,dropout,max_iter,epochs,temperature,lr,batch_size=batch_size)

# # TODO (Developer): Replace this with your Cloud Storage bucket name.
# bucket_name = 'travelapp_luxingshifu'
# bucket = storage_client.get_bucket(bucket_name)
#
# # TODO (Developer): Replace this with the name of the local file to upload.
place_light = 'full_place_list_light.pkl'
model_light = 'model_cdf_light.pkl'
file3 = 'processed_data.pkl'

torch.set_num_threads(1)

# blob1 = bucket.blob(os.path.basename(file1))
# blob1.download_to_filename('full_place_list.pkl')
#
# blob2 = bucket.blob(os.path.basename(file2))
# blob2.download_to_filename('model.pkl')
#
# blob3 = bucket.blob(os.path.basename(file3))
# blob3.download_to_filename('processed_data.pkl')



#
# with open('model/model.pkl','rb') as f:
#     model=pkl.load(f)

# with open('model/processed_data.pkl','rb') as f:
#     data=pkl.load(f)



with open('good_data/full_place_list_light.pkl','rb') as f:
    full_place_list=pkl.load(f)

with open('good_data/place_profiles_light.pkl','rb') as f:
    full_profiles=pkl.load(f)

#note, the +4 is due to the length of prefs.

#Note, need to automatically update latent_feature size!

model=recsys.recsys(latent_features=12,sites=len(full_place_list)+4)
model.load_state_dict(torch.load('model/model_cdf_light'))

# things_to_remove=pkl.load(open('good_data/things_to_remove.pkl','rb'))

site_index=pkl.load(open('good_data/site_index.pkl','rb'))
def site_check(x):
    return x in site_index



#Just need full_profiles

def condition(x):
    bad_words=['cafe', 'restaurant', 'bar', 'deli', 'diner','donald', 'panera','canteen','chophouse','saloon','eatery',\
               'dive','fast-food','coffee','canteen','pizzeria','grill','drive-in','hamburger','caffe','kitchen',\
                'taqueria','crepe','bagel','comer','bakery','steak','noodle','pizza','burger','sushi','hotel','cuisine',\
              'bakeshop','bistro','donuts','baskin robbins','ice cream','jamba juice','pollo','subs','food','soft serve',\
              'taco','ramen','shabu','jerry','holiday inn','chowder','gyros','cantina','fish','juice','taste','inn','pub',\
              'mrs. fields','cebicheria','boulangerie','le marais','nijiya','uma casa','cupcakes','chicken','resturant',\
              'sushirrito','motel','hotel','tavern','biscuit','warming','rv park','lounge','ihop','stinking','pizzetta',\
              'puccini','tortilla']
    return all([w not in x.lower() for w in bad_words])

def site_check(x):
    return x in site_index

mask=[site_check(x) for x in full_place_list]
#
# def condition_vivien(x):
#     return x not in things_to_remove


def rescale(array):
    Max=max(array)
    Min=min(array)
    return (10/(Max-Min))*(np.array(array) - (Max+Min)/2)+5



def preferences_to_placescores(preferences,weight,num_results=10,full_profiles=full_profiles):

    #Idea, add result of neural network to a feature based recommender.

    torch.set_num_threads(1)

    s=1/(np.exp(-2*weight)+1)
    new_user_preferences=np.array(preferences)
    initialization=np.zeros(len(full_profiles))
    new_user=np.concatenate((new_user_preferences,initialization),axis=0)

    print("Drumroll for the prediction", flush = True)
    predictions_raw=model.predict(new_user)
    print("....and done", flush = True)
    predictions=predictions_raw.detach().numpy()[len(preferences):]
    offset=np.array([sum(new_user_preferences*x) for x in full_profiles])

    predictions_filtered=rescale(predictions[mask])
    offset_filtered=rescale(offset[mask])
    sites_filtered=dict(enumerate(np.array(full_place_list)[mask]))


    # predictions_norm=(10/max(np.abs(predictions)))*predictions
    # predictions_norm=(20/(B-S))*(predictions-(B+S)/2)
    # offset=np.array([sum(new_user_preferences*x) for x in full_profiles])
    # offset_norm=(10/max(np.abs(offset)))*offset
    # combo_predictions=(1-s)*predictions_norm+s*offset_norm
    combo_predictions=rescale((1-s)*predictions_filtered+s*offset_filtered)

    # final_predictions=(20/(big-small))*(combo_predictions-(big+small)/2)
    place_prediction=sorted(list(enumerate(combo_predictions)),key=lambda x: x[1],reverse=True)
    potentials=[[sites_filtered[x[0]],x[1]] for x in place_prediction]
    # filtered=list(filter(lambda x: condition(x[0])  and condition_vivien(x[0]),potentials))[:num_results]
    # filtered = list(filter(lambda x: site_check(x[0]),potentials))[:num_results]
    return potentials[:num_results]
