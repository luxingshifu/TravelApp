import numpy as np
import pickle as pkl
import bisect
import os
# import sys
approot=os.getcwd().strip('preprocessing')
# sys.path.append(approot+'recommender_files')
import pandas as pd
import math
from collections import Counter
import random
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


#First we get the important data.  We should make the 'city'
#a variable determined by the user.

city='Los_Angeles'
data_dir=approot+'good_data/'+city
style_mapper=pkl.load(open(approot+'good_data/style_mapper.pkl','rb'))
site_index=pkl.load(open(data_dir+'/site_index.pkl','rb'))
loc_info=pkl.load(open(data_dir+'/loc_info.pkl','rb'))

if city=='San_Francisco':
    #Will write some code here to make SF compatible with
    #new data structure.  Basically, need to modify 'res'
    #in the function 'get_res' so that the extra layer
    #of dictionaries makes sense.
    pass

def remove_reviews(bla):
    return [x for x in bla if 'Reviews' not in x]
loc_info_clean={x: remove_reviews(loc_info[x][1]) for x in loc_info.keys()}




#Results is a list of user rankings of various attractions in 'city'.  It is structured as a list of json
# like files of the form [{user1:{attraction1:rating1,attraction2:rating2,attraction3:rating3}, user2:...}]

def get_res():
    results=[]
    check=True
    k=0
    while check:
        try:
            with open(approot+'preprocessing/raw_data/'+city+f'/file_{k}.pkl','rb') as f:
                likes=pkl.load(f)
                results.append(likes)
            k+=100
        except:
            check=False

    # Now we need to combine all the results above into one big dictionary.
    res=results[0]
    for i in range(1,len(results)):
        res={**res,**results[i]}
    return res

def clean_res():
    res=get_res()
    keys=list(res.keys())
    new_dct={}
    for i in range(len(keys)):
        dct=res[keys[i]]['ratings']
        n=len(city)+2
        new={x[n:]:dct[x] for x in dct if x[n:] in site_index}
        if len(new)!=0:
            new_dct[keys[i]]=new
    return new_dct

def make_matrix(res):
    user_list=list(res.keys())
    matrix=[]
    for u in user_list:
        user_likes=[]
        for p in site_index:
            if p in res[u].keys():
                user_likes.append(res[u][p])
            else:
                user_likes.append(float('NaN'))
        if sum(np.nan_to_num(user_likes,0))!=0:
            matrix.append(user_likes)
    return np.array(matrix), user_list


def make_user_profiles(matrix,loc_info_clean):
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

#         res={k:10*ct[k]/norm for k in ct.keys()}
        #Use the line above if we want to normalize everything to a total of 10, else, use below.
        #Not normalizing kinda makes sense because we can give more weight to those who have travelled
        #more and reviewed more places... these people are more likely to be traveling and use the app
        #anyway.

        res={k:ct[k] for k in ct.keys()}
        profiles[u]=res
    return profiles


def convert_profiles_to_np(profiles,attributes=['Nature','History','Culture','Life']):

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



def make_place_profile(site_index,loc_info_clean,style_mapper):
    place_profile=[]
    for p in site_index:
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


def convert_to_cdf(array,minimum,maximum):
    new=[]
    for x in array:
        index=(maximum-minimum)/(len(array))*(bisect.bisect_left(sorted(array),x))
        new.append(index)
    return np.array(new)


def config_data(res, loc_info_clean, style_mapper):
    # first, use get_data to construct a matrix, user list and place list appearing in the
    # user reviews.

    matrix, user_list = make_matrix(res)

    # The function 'make_place_profile' is essentially just `passing through'. The reason
    # this is done is to ensure that the input and output is consistent with the other
    # data.

    place_profiles=make_place_profile(site_index,loc_info_clean,style_mapper)
    user_profiles=make_user_profiles(matrix,loc_info_clean)
    np_profiles=convert_profiles_to_np(user_profiles)
    np_profiles_normalized=pd.DataFrame(np_profiles).apply(lambda x: convert_to_cdf(x,0,10),axis=0)
    new_matrix=np.concatenate((np_profiles_normalized,matrix),axis=1)

    return user_profiles, new_matrix, place_profiles


res=clean_res()
user_profiles, matrix, place_profiles=config_data(res, loc_info_clean, style_mapper)


pkl.dump(place_profiles,open(data_dir+'/place_profiles.pkl','wb'))
pkl.dump(matrix,open(data_dir+'/matrix.pkl','wb'))
pkl.dump(user_profiles,open(data_dir+'/user_profiles.pkl','wb'))
