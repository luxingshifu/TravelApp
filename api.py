import numpy as np
import pickle as pkl
import recommender_files.recommender_v2 as recommender_v2
# import datetime
import os
import itin_gen.api_itin as api_itin
import itin_gen.api_itin_3 as api_itin_3

#Need to re-generate final_attractions_dict for Los_Angeles.

with open('good_data/San_Francisco/final_attractions_dict.pkl','rb') as f:
    fad=pkl.load(f)


def make_prediction(features):
    # print(features.keys())
    # for k in list(features.keys()):
    #     print(k, type(k))
    for k in list(features.keys()):
        print(k,features[k])
    traveler=features['name']
    nat=float(features['nature'])
    hist=float(features['history'])
    cult=float(features['culture'])
    life=float(features['life'])
    st=float(features['starttime'])
    en=float(features['finishtime'])
    budget=float(features['budget'])
    end=features['endhotel']
    start=features['starthotel']


    preferences=[nat,hist,cult,life]
    if np.linalg.norm(np.array(preferences))<.01:
        preferences = [1,1,1,1]



    recs=recommender_v2.preferences_to_placescores(preferences,num_results=200,weight=.01)



    # progress, routes, best_route, names, diagnostics=api_itin.itin_generator(recs,budget=budget,alpha=.8,ambition=[st,en],max_iterations=1000)

    # actual_route_3=api_itin_3.get_itinerary(recs,start,end, budget,ambition=[st,en])



    actual_route_3 = api_itin_3.get_itinerary(recs, start, end, budget=budget, ambition=[st,en])
    # actual_route=[names[val] for val in routes[best_route][0]]

    key=os.environ['GOOGLEMAPS_KEY']
    rec_photo=[]
    for place in recs:
        try:
            photoref=fad[place[0]][0]['photos'][0]['photo_reference']
            url='https://maps.googleapis.com/maps/api/place/photo?key='+key+'&maxwidth=200&photoreference='+photoref
        except:
            if type(photoref)==list:
                photoref=photoref[0]
                url='https://maps.googleapis.com/maps/api/place/photo?key=' + key + '&maxwidth=200&photoreference='+photoref
            else:
                url='https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iJ913gPoMsl0/v0/200x-1.jpg'

        newrow=[place[0],round(place[1],2)]+[url]
        rec_photo.append(newrow)





    result = {
        'recommendations':recs,
        'actual_route':actual_route_3,
        'rec_photo':rec_photo}

    return result


if __name__ == '__main__':
    print(make_prediction(example))
