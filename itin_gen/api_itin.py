def itin_generator(user_preferences, budget=2000, ambition = [9, 17], temperature=100, stopping_temperature=0.00000001, max_iterations=10000, alpha=0.995):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    # from data_gen import get_fake_data
    # from initial_route import create_initial_route
    # from test_locations import get_start_and_stop
    # from Path import Path
    # from Fitness import Fitness
    import test2

    np.random.seed(55)

    # for current testing purposes
    tour_length = 60*(ambition[1] - ambition[0])

    site_name_lookup, SF_sites, travel_matrix = test2.get_fake_data(user_preferences)

    iteration = 1

    start, stop = test2.get_start_and_stop(SF_sites)

    '''create an initial route'''
    current_route = test2.create_initial_route(start, stop, budget, tour_length, SF_sites, travel_matrix)

    progress = []
    routes = []

    ''' PRINT THINGS '''
    print('budget:', budget)
    print('ambition:', ambition)
    print('initial route', current_route)

    '''keep track of score from initial selection'''
    initial_score = current_route[1]
    progress.append(current_route[1])
    print(progress)

    '''keep track of route from initial selection'''
    routes.append(current_route)


    '''modify and iterate until temperature has cooled or iterations have been hit'''
    while (temperature > stopping_temperature) & (iteration < max_iterations):
        '''get a candidate'''

        current_route = test2.propagate_change(current_route, budget, tour_length, SF_sites, travel_matrix, temperature)
        if (iteration % 50 == 0) :
            print(iteration, temperature, progress[-1], routes[-1])

            '''not bonanza!!!!! raise temperature by '''
            if len(progress) > 50 and (progress[-50] == progress[-1]):
                temperature *= 2-alpha/(2*50)
        else:
            '''decrease per usual'''
            temperature *= alpha

        '''iter-plus'''
        iteration += 1

        '''keep track of the best score & route from the current generation'''
        progress.append(current_route[1])

        routes.append(current_route)

    best_score_index = np.argmax(np.array(progress))
    best_score = progress[best_score_index]
    best_route = routes[best_score_index]
    print(best_score, best_route)


    '''create the plot'''
    # plt.plot(progress)
    # plt.ylabel('Fitness Score')
    # plt.xlabel('Iteration')
    # plt.show()

    return progress, route, best_score_index
