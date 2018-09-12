# Kickstarter!!!!!!!!!!!!!!!!!!
import numpy as np
import pickle
import recommender
import itin_gen.api_itin as api_itin
from api_itin import itin_generator
#import pandas







def make_prediction(features):

    nat=features['nature']
    hist=features['history']
    cult=features['culture']
    life=features['life']

    preferences=[nat,hist,cult,life]
    recs=recommender.preferences_to_placescores(preferences,num_results=20,weight=.01)
    # rec_list=recs

    progress, routes, best_route, names=itin_generator(recs,alpha=.8,max_iterations=1000)

    actual_route=[names[val] for val in routes[best_route][0]]
    print(f'This is the actual route {actual_route})')

    result = {
        'prediction': int(0 > 0.5),
        'prob_succeed': 0,
        'nature':nat,
        'recommendations':recs,
        'actual_route':actual_route}

    return result




if __name__ == '__main__':
    print(make_prediction(example))
