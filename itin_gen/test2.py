import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

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
        score = self.highlight_bonus*self.similarity*self.visit_length - self.travel_time() - self.travel_cost()
        return score

    def __repr__(self):
        return self.name

class Fitness:

    def __init__(self, route, SF_sites, travel_matrix):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
        self.SF_sites = SF_sites
        self.travel_matrix = travel_matrix

    def route_fitness(self):
        from Path import Path

        if self.fitness == 0:
            path_fitness = 0
            for i in range(len(self.route)-1):
                from_attraction = self.route[i]
                to_attraction = self.route[i+1]
                path = Path(from_attraction, to_attraction, self.SF_sites, self.travel_matrix)
                path_fitness += path.score()
        self.fitness = path_fitness
        return self.fitness

def get_start_and_end(SF_sites):
    sites = list(SF_sites.index)
    start = np.random.choice(sites)
    stop = start
    return start, stop

def create_initial_route(start, stop, budget, tour_length, SF_sites, travel_matrix):
    import numpy as np
    from Path import Path
    from Fitness import Fitness

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

    score = Fitness(path, SF_sites, travel_matrix).route_fitness()

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
    neighbors = nodes_to_check[((nodes_to_check['final_budget'] >= 0) & (nodes_to_check['final_tour_length'] >= 0)) & ((nodes_to_check['final_budget'] < .15*budget) | (nodes_to_check['final_tour_length'] < .15*tour_length))]
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
        if self.length < 4:
            num_options = 2
        else:
            num_options = 3

        wait_and_see = np.random.choice(range(num_options))
        node_index = np.random.choice(range(1, self.length-1), size=(num_options - 1,), replace=False)

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
            candidate_0 = self.deletion((index+1), candidate_0)
        return candidate_0

    def get_surviving_candidate(self, node_index, wait_and_see, candidate_0):
        candidate = tag_survival(candidate_0, self.o_budget, self.o_tour_length, self.SF_sites, self.travel_matrix)
        trials = 0

        while ((candidate == '') & (trials < self.max_loops)):
            wait_and_see, node_index = self.roulette()
            candidate = self.next_move(wait_and_see, node_index)
            candidate = tag_survival(candidate, self.o_budget, self.o_tour_length, self.SF_sites, self.travel_matrix)
            trials += 1

        if (trials == self.max_loops):
            wait_and_see, node_index = self.roulette()
            candidate = self.next_move(wait_and_see, node_index)
            candidate = self.get_surviving_candidate(node_index, wait_and_see, candidate)

#         new_route = get_new_route(candidate, SF_sites_df, travel_matrix)
#         print(new_route.score)
        return candidate


def get_candidate(current_route, budget, tour_length, SF_sites, travel_matrix):
    current_route = get_new_route(current_route, budget, tour_length, SF_sites, travel_matrix)
    wait_and_see, node_index = current_route.roulette()
    candidate = current_route.next_move(wait_and_see, node_index)
    current_route = current_route.get_surviving_candidate(node_index, wait_and_see, candidate)
    return current_route

def tag_survival(candidate, budget, tour_length, SF_sites, travel_matrix):
    o_budget = budget
    o_tour_length = tour_length

    for index in range(len(candidate)-1):
        site_A = candidate[index]
        site_B = candidate[index+1]

        '''budget - ticket_prices - cost of traveling from location A to location B'''
        budget = budget - SF_sites['visit_cost'][site_B] - travel_matrix[1][site_A][site_B]

        '''tour_length - visit_length - time to travel from location A to location B'''
        tour_length = tour_length - SF_sites['visit_length'][site_B] - travel_matrix[0][site_A][site_B]

        if (budget < 0) | (tour_length < 0):
            return ''

    leftover_budget = budget
    leftover_tour_length = tour_length

    if ((leftover_budget <= 0.15*o_budget) | (leftover_tour_length <= 0.15*o_tour_length)):
        score = Fitness(candidate, SF_sites, travel_matrix).route_fitness()
        return candidate, score, [leftover_budget, leftover_tour_length]
    else:
        return ''



def probability_move(current_path, candidate, temperature):

    '''fitness is recorded in <path>[1]'''
    '''if the candidate is better, return a probability of 1 for updating'''
    if (current_path[1] - candidate[1]) < 0.001:
        return 1
    else:
        ''' let the cooling function handle the updating probability'''
        return math.exp(-abs(current_path[1] - candidate[1])/temperature)

def update_path(current_path, candidate, temperature):
    p_prob = probability_move(current_path, candidate, temperature)
    if np.random.random() < p_prob:
        current_path = candidate
        return candidate
    return current_path

def propagate_change(current_route, budget, tour_length, SF_sites, travel_matrix, temperature):
    candidate = get_candidate(current_route, budget, tour_length, SF_sites, travel_matrix)
    updated_route = update_path(current_route, candidate, temperature)
    return updated_route
# def simulated_annealing_plot(budget, tour_length, SF_sites, travel_matrix, temperature=100, stopping_temperature=0.00000001, max_iterations=10000, alpha=0.995):
#     #np.random.seed(52)
#     iteration = 1
#
#     start, stop = get_start_and_end(SF_sites)
#     print(start, stop)
#
#     '''create an initial route'''
#     current_route = create_initial_route(start, stop, budget, tour_length, SF_sites, travel_matrix)
#
#     progress = []
#     routes = []
#
#     '''keep track of score from initial selection'''
#     initial_score = current_route[1]
#     progress.append(current_route[1])
#     print(progress)
#
#     '''keep track of route from initial selection'''
#     routes.append(current_route)
#
#
#     '''modify and iterate until temperature has cooled or iterations have been hit'''
#     while (temperature > stopping_temperature) & (iteration < max_iterations):
#         '''get a candidate'''
#
#         current_route = propagate_change(current_route, budget, tour_length, SF_sites, travel_matrix, temperature)
#
#         if (iteration % 50 == 0) :
#             print(iteration, temperature, progress[-1])
#
#             '''not bonanza!!!!! raise temperature by '''
#             if len(progress) > 50 and (progress[-50] == progress[-1]):
#                 temperature *= 2-alpha/(2*50)
#         else:
#             '''decrease per usual'''
#             temperature *= alpha
#
#         '''iter-plus'''
#         iteration += 1
#
#         '''keep track of the best score & route from the current generation'''
#         progress.append(current_route[1])
#
#         routes.append(current_route)
#
#
#     '''create the plot'''
#     plt.plot(progress)
#     plt.ylabel('Fitness Score')
#     plt.xlabel('Iteration')
#     plt.show()
#
#     best_score_index = np.argmax(np.array(progress))
#     best_score = progress[best_score_index]
#     best_route = routes[best_score_index]
#     print(best_score, best_route)
#
#     return progress, route, best_score_index
