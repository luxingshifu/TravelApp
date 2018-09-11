import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math
# from system_limits import recursionlimit

def get_fake_data(user_preferences):

    V = len(user_preferences)

    '''generate SF_sites data'''
    similarities = [pref[1] for pref in user_preferences]
    print(f'similarities is ..... {similarities})')
    visit_lengths = 60*np.random.randint(1,4, size=V)
    ticket_prices = np.random.randint(0, 50, size = V)

    highlight_reel = np.random.randint(0,V,10)
    highlight_bonuses = np.ones(V)

    for highlight in highlight_reel:
        highlight_bonuses[highlight] = 1.25

    All_SF_sites = dict(zip(range(V), list(zip(similarities, highlight_bonuses, visit_lengths, ticket_prices))))

    #SF_sites = {key: value for (key, value) in All_SF_sites.items() if value[0] >= } #this may need to be altered pending input
    SF_sites = All_SF_sites
    SF_sites = pd.DataFrame.from_dict(SF_sites).T
    SF_sites.columns = ['similarity', 'highlight_bonus', 'visit_length', 'visit_cost']
    #SF_sites.index.names = list(SF_sites.keys())

    '''assign ID to sites'''
    names = [pref[0] for pref in user_preferences]
    site_names = [site for site in names]
    site_name_lookup = dict(zip(range(V), site_names))

    '''generate travel_matrix'''
    D = 2
    positions = np.random.randint(2,300, size=(V, D))
    differences = positions[:, None, :] - positions[None, :, :]
    travel_times = np.sqrt(np.sum(differences**2, axis=-1))

    costs = np.random.randint(0,15, size=(V, D))
    distances = costs[:, None, :] - costs[None, :, :]
    travel_costs = np.sqrt(np.sum(distances**2, axis=-1))

    travel_times_df = pd.DataFrame(travel_times)[list(SF_sites.index)].iloc[list(SF_sites.index)]
    travel_costs_df = pd.DataFrame(travel_costs)[list(SF_sites.index)].iloc[list(SF_sites.index)]
    travel_matrix = [travel_times_df, travel_costs_df]

    return site_name_lookup, SF_sites, travel_matrix

class Path:
    def __init__(self, from_attraction, to_attraction, SF_sites, travel_matrix):

        self.similarity = SF_sites['similarity'][to_attraction]
        self.highlight_bonus = SF_sites['highlight_bonus'][to_attraction]
        self.visit_length = SF_sites['visit_length'][to_attraction]
        self.visit_cost = SF_sites['visit_cost'][to_attraction]

        self.travel_times = travel_matrix[0]
        self.travel_costs = travel_matrix[1]

        self.to_attraction = to_attraction
        self.from_attraction = from_attraction
        self.name = "(" + str(self.from_attraction) + "->" + str(self.to_attraction) + ")"

    def travel_time(self):
        travel_time = self.travel_times[self.from_attraction][self.to_attraction]
        return travel_time

    def travel_cost(self):
        travel_cost = self.travel_costs[self.from_attraction][self.to_attraction]
        return travel_cost

    def score(self, B1=1, B2=2):
        # print(f'type highlight_bonus is {type(self.highlight_bonus)})')
        # print(f'type similarity is {type(self.similarity)})')
        # print(f'type visit length is {type(self.visit_length)})')
        # print(f'type travel time is {type(self.travel_time())})')
        # print(f'type travel cost is {type(self.travel_cost())})')
        score = float(self.highlight_bonus)*float(self.similarity)*float(self.visit_length) - float(self.travel_time()) - float(self.travel_cost())
        return score

    def __repr__(self):
        return self.name

class Fitness:

    def __init__(self, route, budget, tour_length, available_budget, available_tour_length, SF_sites, travel_matrix):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
        self.SF_sites = SF_sites
        self.travel_matrix = travel_matrix
        self.o_budget = budget
        self.o_tour_length = tour_length
        self.available_budget = available_budget
        self.available_tour_length = available_tour_length

    def route_fitness(self):
#         from Path import Path

        if self.fitness == 0:
            path_fitness = 0
            for i in range(len(self.route)-1):
                from_attraction = self.route[i]
                to_attraction = self.route[i+1]
                path = Path(from_attraction, to_attraction, self.SF_sites, self.travel_matrix)
                path_fitness += path.score()

            spent_budget = self.o_budget - self.available_budget
            spent_tour_length = self.o_tour_length - self.available_tour_length

            if (spent_budget < self.o_budget):
                budget_score = .3*path_fitness
            else:
                budget_score = -.5*path_fitness

            if (spent_tour_length < self.o_tour_length):
                tour_length_score = .3*path_fitness
            else:
                tour_length_score = -.5*path_fitness

        self.fitness = path_fitness + budget_score + tour_length_score
        return self.fitness

def get_start_and_stop(SF_sites):

    sites = list(SF_sites.index)
    start = np.random.choice(sites)
    stop = start
    return start, stop

def create_initial_route(start, stop, budget, tour_length, SF_sites, travel_matrix):

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

##### STRATEGY: Next SA Move

def get_route_locations(node_index, current_route_0):
    node = current_route_0[node_index]
    before_node = current_route_0[node_index-1]
    after_node = current_route_0[node_index+1]
    return node, before_node, after_node

def get_available_resources(node, before_node, after_node, current_path, leftover_budget, leftover_tour_length, SF_sites, travel_matrix):

    '''add back the cost of travel to the site, travel from the site, and money spent at the site'''
    available_budget = leftover_budget + travel_matrix[1][before_node][node] + travel_matrix[1][node][after_node] + SF_sites['visit_cost'][node]

    '''add back the time of travel to the site, travel from the site, and time spent at the site'''
    available_tour_length = leftover_tour_length + travel_matrix[0][before_node][node] + travel_matrix[0][node][after_node] + SF_sites['visit_length'][node]
    return available_budget, available_tour_length

def get_neighbors(node_index, current_path, budget, tour_length, leftover_budget, leftover_tour_length, SF_sites, travel_matrix):
    ''' neighbors are any individuals that when inserted for a node, do not exceed the defined cost limits'''

    '''**available resources are the following'''

    '''current_state locations'''
    node, before_node, after_node = get_route_locations(node_index, current_path)

    '''$ and time resources'''
    available_budget, available_tour_length = get_available_resources(node, before_node, after_node, current_path, leftover_budget, leftover_tour_length, SF_sites, travel_matrix)

    '''**prune the available total possible solutions to a more workable list'''

    '''remove all node locations that are in the current path'''
    all_sites = list(SF_sites.index)
    possible_sites = set(all_sites).difference(set(current_path))

    '''**prune: for insertions: remove all node locations that are outside the available budget and tour length'''

    new_travel_costs = (travel_matrix[1][before_node] + travel_matrix[1][after_node]).drop(labels=[before_node, node, after_node])
    new_final_budget = np.array([available_budget]*len(new_travel_costs)) - (SF_sites['visit_cost'].drop([before_node, node, after_node]) + new_travel_costs)
    new_final_budget.name = 'final_budget'

    new_travel_lengths = (travel_matrix[0][before_node] + travel_matrix[0][after_node]).drop(labels=[before_node, node, after_node])
    new_final_tour_length = np.array([available_tour_length]*len(new_travel_lengths)) - (SF_sites['visit_length'].drop([before_node, node, after_node]) + new_travel_lengths)
    new_final_tour_length.name = 'final_tour_length'

    nodes_to_check = pd.concat([new_final_budget, new_final_tour_length], axis=1)

    '''check neighbors for viability'''
    neighbors = nodes_to_check[((nodes_to_check['final_budget'] >= 0) & (nodes_to_check['final_tour_length'] >= 0))]
    return neighbors.index


class get_new_route():

    def __init__(self, current_route, budget, tour_length, SF_sites, travel_matrix):
        self.name = 'routing'

        self.current_route = current_route[0]
        self.score = current_route[1]
        self.o_budget = budget
        self.o_tour_length = tour_length
        self.leftover_budget = current_route[2][0]
        self.leftover_tour_length = current_route[2][1]
        self.SF_sites = SF_sites
        self.travel_matrix = travel_matrix

        self.length = len(current_route[0])
        self.max_loops = self.length

    def roulette(self):
        num_options = 0
        if self.length == 3:
            wait_and_see = 0
            node_index = [1,]
            return wait_and_see, node_index
        if self.length < 4:
            num_options = 2
        else:
            num_options = 3

        wait_and_see = np.random.choice(range(num_options))
        node_index = np.random.choice(range(1, self.length-1), size=(num_options-1,), replace=False)

        return wait_and_see, node_index

    def next_move(self, wait_and_see, node_index):
        if wait_and_see == 0:
            return self.insertion(node_index[0], self.current_route)
        elif wait_and_see == 1:
            return self.deletion(node_index[0], self.current_route)
        elif wait_and_see == 2:
            return self.mutation(node_index, self.current_route)

    def get_neighbors(self, node_index, current_route):
        neighbors = get_neighbors(node_index, current_route, self.o_budget, self.o_tour_length, self.leftover_budget, self.leftover_tour_length, self.SF_sites, self.travel_matrix)
        return neighbors

    def insertion(self, node_index, current_route):
        wait_and_see = 0
        neighbors = self.get_neighbors(node_index, current_route)

        trials = 0
        while ((len(neighbors) == 0) & (trials < self.max_loops)):
            node_index = np.random.choice(range(1, self.length-1))
            neighbors = self.get_neighbors(node_index, current_route)
            trials += 1

        if trials == self.max_loops:
            return current_route
#             wait_and_see, node_index = self.roulette()
#             return self.next_move(wait_and_see, node_index)

        neighbor = np.random.choice(neighbors)
        candidate_0 = current_route[:node_index] + [neighbor] + current_route[node_index:]
        return candidate_0

    def deletion(self, node_index, current_route):
        candidate_0 = current_route[:node_index] + current_route[(node_index+1):]
        return candidate_0

    def mutation(self, indices, current_route):
        indices = sorted(indices)
        candidate_0 = current_route
        for index in range(indices[0], indices[1]+1):
            candidate_0 = self.insertion(index, candidate_0)
            if len(candidate_0) > len(current_route):
                candidate_0 = self.deletion((index+1), candidate_0)
        return candidate_0

    def score_candidate(self, candidate):
        budget = self.o_budget
        tour_length = self.o_tour_length

        for index in range(len(candidate)-1):
            site_A = candidate[index]
            site_B = candidate[index+1]

            '''budget - ticket_prices - cost of traveling from location A to location B'''
            budget = budget - self.SF_sites['visit_cost'][site_B] - self.travel_matrix[1][site_A][site_B]

            '''tour_length - visit_length - time to travel from location A to location B'''
            tour_length = tour_length - self.SF_sites['visit_length'][site_B] - self.travel_matrix[0][site_A][site_B]

            leftover_budget = budget
            leftover_tour_length = tour_length

        route_score = Fitness(candidate, self.o_budget, self.o_tour_length, self.leftover_budget, self.leftover_tour_length, self.SF_sites, self.travel_matrix).route_fitness()
        return candidate, route_score, [leftover_budget, leftover_tour_length]

def get_candidate(current_route, budget, tour_length, SF_sites, travel_matrix):
    current_route = get_new_route(current_route, budget, tour_length, SF_sites, travel_matrix)
    wait_and_see, node_index = current_route.roulette()
    candidate = current_route.next_move(wait_and_see, node_index)
    current_route = current_route.score_candidate(candidate)
    return current_route

def probability_move(current_path, candidate, temperature):

    '''fitness is recorded in <path>[1]'''
    '''if the candidate is better, return a probability of 1 for updating'''
    if candidate[1] - current_path[1] > 0:
        return 1
    elif candidate[1] < 0:
        print(candidate[1])
        return 0
    else:
        ''' let the cooling function handle the updating probability'''
        return math.exp(-(current_path[1] - candidate[1])/temperature)

def update_path(current_path, candidate, temperature):
    p_prob = probability_move(current_path, candidate, temperature)
    print(p_prob)
    if np.random.random() < p_prob:
        return candidate
    return current_path

def propagate_change(current_route, budget, tour_length, SF_sites, travel_matrix, temperature):
    candidate = get_candidate(current_route, budget, tour_length, SF_sites, travel_matrix)
    updated_route = update_path(current_route, candidate, temperature)
    return updated_route
