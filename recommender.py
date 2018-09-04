
import numpy as np
import pickle as pkl
import os
import math
from collections import Counter
import nltk
import random
import recsys
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def create_recsys(matrix,dropout=.1,latent_features=4,max_iter=10,lr=.001,epochs=3,temperature=1,batch_size=50):
    return recsys.recsys(matrix,len(matrix),len(matrix[0]),latent_features,dropout,max_iter,epochs,temperature,lr,batch_size=batch_size)

with open('model/model.pkl','rb') as f:
    model=pkl.load(f)

with open('data/processed_data.pkl','rb') as f:
    data=pkl.load(f)


def preferences_to_placescores(preferences,weight,num_results=10,user_profiles=data[0],new_matrix=data[1],places=data[2],mpp=data[3],style_mapper=data[4]):

    full_profiles, full_place_list, urp = mpp
    zeros=np.zeros((len(new_matrix),len(urp)))
    new_user_preferences=(10/sum(preferences))*np.array(preferences)
    initialization=np.zeros(len(new_matrix[0])-len(preferences))
    new_user=np.concatenate((new_user_preferences,initialization),axis=0)
    predictions_raw=model.predict(new_user)
    predictions=predictions_raw.detach().numpy()[len(preferences):]
    offset=[sum(new_user_preferences*x) for x in full_profiles]
    final_predictions=predictions+weight*np.array(offset)
    place_prediction=sorted(list(enumerate(final_predictions)),key=lambda x: x[1],reverse=True)[:num_results]

    return [[full_place_list[x[0]],x[1]] for x in place_prediction]
