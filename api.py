# Kickstarter!!!!!!!!!!!!!!!!!!
import numpy as np
import pickle
import recommender
#import pandas







def make_prediction(features):

    nat=features['nature']
    hist=features['history']
    cult=features['culture']
    life=features['life']

    preferences=[nat,hist,cult,life]
    recs=recommender.preferences_to_placescores(preferences,num_results=20,weight=.01)
    # rec_list=recs

    result = {
        'prediction': int(0 > 0.5),
        'prob_succeed': 0,
        'nature':nat,
        'recommendations':recs}

    return result




if __name__ == '__main__':
    print(make_prediction(example))
