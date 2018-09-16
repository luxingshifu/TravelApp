import numpy as np
import pickle as pkl
import bisect
import os
import pandas as pd
import math
from collections import Counter
import random
import recsys
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



#Results is a list of user rankings of various attractions in SF.  It is structured as a list of json
# like files of the form [{user1:{attraction1:rating1,attraction2:rating2,attraction3:rating3}, user2:...}]

#original range is 19400
def get_res():
    results=[]
    for k in range(100,400,100):
        with open(f'raw_data/file_{k}.pkl','rb') as f:
            likes=pkl.load(f)
            results.append(likes)

    # Now we need to combine all the results above into one big dictionary.

    res=results[0]
    for i in range(1,len(results)):
        res={**res,**results[i]}
    return res

with open('good_data/loc_info.pkl','rb') as f:
    loc_info=pkl.load(f)

with open('good_data/style_mapper.pkl','rb') as f:
    style_mapper=pkl.load(f)

site_index=pkl.load(open('good_data/site_index.pkl','rb'))

def remove_reviews(bla):
    return [x for x in bla if 'Reviews' not in x]

def make_matrix(res):
    user_list=list(res.keys())
#     places=[]
#     for user_key in user_list:
#         for place in res[user_key].keys():
#             place_minus_sf=place[15:]
#             #We are removing the conditions on the places for now.  These will be added later after training.
# #             if condition(place.lower()):
#             places.append(place)
#
#     #this is the same as uvp_set above.
#     places=list(set(places))


    matrix=[]
    for u in user_list:
        user_likes=[]
        for p in site_index:
            if p in res[u].keys():
                user_likes.append(res[u][p])
            else:
                user_likes.append(float('NaN'))
        matrix.append(user_likes)

    return np.array(matrix), user_list


# get_data takes in the results 'res' above and spits out a 'clean' matrix of user preferences
# as well as a list of users and places.  This is basically a cleaned version of 'make_matrix'

def get_data(res):

    matrix, user_list = make_matrix(res)

    good_list=[]
    for i in range(len(matrix)):
        if sum(np.nan_to_num(matrix[i],0))!=0:
            good_list.append(i)

    users_clean=np.array(user_list)[good_list]
    matrix_clean=matrix[good_list]
    return matrix_clean, users_clean

def make_profiles_new(matrix,loc_info_clean,site_index):
    profiles={}
    for u in range(len(matrix)):
        features_list=[]
        for i in range(len(matrix[0])):
            try:
                features_list+=int(matrix[u][i])*(loc_info_clean[site_index[i]])
            except:
                features_list+=''

        new_features=[]
        for f in features_list:
            try:
                nf=style_mapper[f]
                if type(nf)==str:
                    new_features.append(nf)
                else:
                    new_features+=nf
            except:
                pass

        ct=Counter(new_features)
        norm=np.linalg.norm(list(ct.values()))
        tot=sum(ct.values())

        # res={k:10*ct[k]/norm for k in ct.keys()}
        #Use the line above if we want to normalize everything to a total of 10, else, use below.
        #Not normalizing kinda makes sense because we can give more weight to those who have travelled
        #more and reviewed more places... these people are more likely to be traveling and use the app
        #anyway.

        res={k:ct[k] for k in ct.keys()}
        profiles[u]=res
    return profiles



def convert_profiles_to_np(profiles,attributes=['Nature','History','Culture','Life']):

    #Where does matrix come from?
    np_profiles=[]
    for u in range(len(profiles)):
        user_profile=[]
        for a in attributes:
            if a in profiles[u].keys():
                user_profile.append(profiles[u][a])
            else:
                user_profile.append(0)
        np_profiles.append(user_profile)

    return np.array(np_profiles)

#Flatten is just a helper function for 'make_profiles'

def flatten(testlist):
    acc=[]
    for x in testlist:
        if type(x)==str:
            acc.append(x)
        elif type(x)==list:
            acc+=x
    return acc

def make_place_profile(places_clean,loc_info_clean,style_mapper):
    place_profile=[]
    for p in places_clean:
        try:
            loc_info=loc_info_clean[p]
            new_labels=[]
            for label in loc_info:

                sml=style_mapper[label]

                if type(sml)==list:
                    new_labels+=sml
                else:
                    new_labels+=[sml]
                    pass
                #print(new_labels)
            ct=Counter(new_labels)
            place_profile.append(np.array([ct['Nature'],ct['History'],ct['Culture'],ct['Life']]))
        except:
            place_profile.append(np.array([0,0,0,0]))
    return place_profile

def create_recsys(matrix,dropout=.1,latent_features=4,max_iter=10,lr=.001,epochs=3,temperature=1,batch_size=50):
    return recsys.recsys(matrix,len(matrix),len(matrix[0]),latent_features,dropout,max_iter,epochs,temperature,lr,batch_size=batch_size)

def make_place_arrays(site_index,loc_info_clean,style_mapper):

    #loc_info_clean is apparently the full place list

    # places_clean=[x[15:] for x in places]
    unreviewed_places=[x for x in loc_info_clean.keys() if x not in places_clean and 'tour' not in x.lower()]
    site_index=places_clean+unreviewed_places


    place_profile = make_place_profile(places_clean,loc_info_clean,style_mapper)
    unreviewed_place_profile = make_place_profile(unreviewed_places,loc_info_clean,style_mapper)

    reviewed_places = np.vstack(place_profile)
    unreviewed_places = np.vstack(unreviewed_place_profile)

    full_profiles = np.concatenate((reviewed_places,unreviewed_places),axis=0)

    return full_profiles, site_index, unreviewed_places


def convert_to_cdf(array,minimum,maximum):
    new=[]
    for x in array:
        index=(maximum-minimum)/(len(array))*(bisect.bisect_left(sorted(array),x)-(len(array))/2)
        new.append(index)
    return np.array(new)



def config_data(res, loc_info, style_mapper):



    # first, use get_data to construct a matrix, user list and place list appearing in the
    # user reviews.

    matrix, user_list= get_data(res)

    #next, build the user profiles based off of their reviews.

    loc_info_clean={x: remove_reviews(loc_info[x][1]) for x in loc_info.keys()}

    # The function 'make_place_arrays' is essentially just `passing through'. The reason
    # this is done is to ensure that the input and output is consistent with the other
    # data.


    # mpp = make_place_arrays(places,loc_info_clean,style_mapper)
    mpp=make_place_profile(site_index,loc_info_clean,style_mapper)
    # full_profiles, site_index, unreviewed_places = mpp
    full_profiles = mpp




    user_profiles=make_profiles_new(matrix,loc_info_clean,site_index)
    np_profiles=convert_profiles_to_np(user_profiles)

    np_profiles_normalized=pd.DataFrame(np_profiles).apply(lambda x: convert_to_cdf(x,0,10),axis=0)

    # next, we clean up places and add in a string of zeros for all the places which the
    # users (all of them) have not been to.

    # places_clean=[x[15:] for x in places]
    # padding=len([x for x in loc_info_clean.keys() if x not in places_clean and 'tour' not in x.lower()])
    # zeros=np.zeros((len(matrix),padding))
    new_matrix=np.concatenate((np_profiles_normalized,matrix),axis=1)



    return user_profiles, new_matrix, places, mpp, style_mapper
