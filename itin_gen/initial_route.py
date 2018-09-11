def create_initial_route(start, stop, budget, tour_length, SF_sites, travel_matrix):
    import numpy as np
    from test2 import Path
    from test2 import Fitness

    sites = list(SF_sites.index)
    path = [start]
    not_valid = []

    o_budget = budget
    o_tour_length = tour_length

    while ((budget > 0.1*o_budget) | (tour_length > 0.1*o_tour_length)) & ((budget > 0) & (tour_length > 0)):

        if len(path) == 1:
            from_attraction = start

        possibilities = list(set(sites).difference(set(path + not_valid + [stop])))

        '''if there aren't any options left, break'''
        if len(possibilities) == 0:
                break
        to_attraction = np.random.choice(possibilities)

        '''budget - ticket_prices - cost of traveling from location A to location B'''
        budget = budget - SF_sites['visit_cost'][to_attraction] - travel_matrix[1][from_attraction][to_attraction]

        '''tour_length - visit_length - time to travel from location A to location B'''
        tour_length = tour_length - SF_sites['visit_length'] [to_attraction]- travel_matrix[0][from_attraction][to_attraction]

        '''check that there is enough budget/time left to get to get to the stop location'''
        if (budget > travel_matrix[1][to_attraction][stop]) & (tour_length > travel_matrix[0][to_attraction][stop]):
            path.append(to_attraction)
            from_attraction = to_attraction

        else:
            '''if not, go back to the drawing board, and add the current site to the discard pile'''
            not_valid.append(to_attraction)
            '''add the budget and time back'''
            budget = budget + SF_sites['visit_cost'][to_attraction] + travel_matrix[1][from_attraction][to_attraction]
            tour_length = tour_length + SF_sites['visit_length'][to_attraction]+ travel_matrix[0][from_attraction][to_attraction]

    path.append(stop)

    available_budget = budget - travel_matrix[1][path[-1]][stop]
    available_tour_length = tour_length - travel_matrix[0][path[-1]][stop]

    score = Fitness(path, o_budget, o_tour_length, available_budget, available_tour_length, SF_sites, travel_matrix).route_fitness()

    return path, score, [available_budget, available_tour_length]
