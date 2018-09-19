# Kickstarter!!!!!!!!!!!!!!!!!!
import numpy as np
import pickle
import recommender_v2
import datetime
import itin_gen.api_itin as api_itin


def make_prediction(features):

    nat=float(features['nature'])
    hist=float(features['history'])
    cult=float(features['culture'])
    life=float(features['life'])
    st=features['starttime']
    en=features['finishtime']

    preferences=[nat,hist,cult,life]

    recs=recommender_v2.preferences_to_placescores(preferences,num_results=200,weight=.01)


    progress, routes, best_route, names=api_itin.itin_generator(recs,alpha=.8,ambition=[3,22],max_iterations=1000)

    actual_route=[names[val] for val in routes[best_route][0]]


    result = {
        'recommendations':recs,
        'actual_route':actual_route,
        'progress':progress}

    return result


if __name__ == '__main__':
    print(make_prediction(example))
