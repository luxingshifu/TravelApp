
import numpy as np
import pickle as pkl
import os
import math
from collections import Counter
import random
import recsys
import boto3
#from google.appengine.api import app_identity
# use the following line to load from bucket.
# import google.cloud.storage
#
# import logging
# #import webapp2
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
# storage_client = google.cloud.storage.Client("TravelApp")

def create_recsys(matrix,dropout=.1,latent_features=4,max_iter=10,lr=.001,epochs=3,temperature=1,batch_size=50):
    return recsys.recsys(matrix,len(matrix),len(matrix[0]),latent_features,dropout,max_iter,epochs,temperature,lr,batch_size=batch_size)

# # TODO (Developer): Replace this with your Cloud Storage bucket name.
# bucket_name = 'travelapp_luxingshifu'
# bucket = storage_client.get_bucket(bucket_name)
#
# # TODO (Developer): Replace this with the name of the local file to upload.
file1 = 'full_place_list.pkl'
file2 = 'model.pkl'
file3 = 'processed_data.pkl'

# blob1 = bucket.blob(os.path.basename(file1))
# blob1.download_to_filename('full_place_list.pkl')
#
# blob2 = bucket.blob(os.path.basename(file2))
# blob2.download_to_filename('model.pkl')
#
# blob3 = bucket.blob(os.path.basename(file3))
# blob3.download_to_filename('processed_data.pkl')


with open('model/model.pkl','rb') as f:
    model=pkl.load(f)

# with open('model/processed_data.pkl','rb') as f:
#     data=pkl.load(f)

with open('model/place_profiles.pkl','rb') as f:
    full_profiles=pkl.load(f)

with open('model/full_place_list.pkl','rb') as f:
    full_place_list=pkl.load(f)

#Just need full_profiles

def preferences_to_placescores(preferences,weight,num_results=10,full_profiles=full_profiles):

    s=1/(np.exp(-2*weight)+1)

    new_user_preferences=np.array(preferences)
    initialization=np.zeros(len(full_profiles))
    new_user=np.concatenate((new_user_preferences,initialization),axis=0)
    predictions_raw=model.predict(new_user)
    predictions=predictions_raw.detach().numpy()[len(preferences):]
    predictions_norm=(10/max(predictions))*predictions
    offset=np.array([sum(new_user_preferences*x) for x in full_profiles])
    offset_norm=(10/max(offset))*offset
    final_predictions=(1-s)*predictions_norm+s*(10/max(offset))*offset
    place_prediction=sorted(list(enumerate(final_predictions)),key=lambda x: x[1],reverse=True)[:num_results]

    return [[full_place_list[x[0]],x[1]] for x in place_prediction]
