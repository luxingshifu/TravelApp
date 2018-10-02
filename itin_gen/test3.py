'''Last edited: Vivien Tsao 9/19/18'''

import pickle as pkl
import numpy as np

'''SHARED CLASSES'''
'''This sets up all the classes used by the final model(s).'''

user_preferences = pkl.load(open('austin.pkl', 'rb'))
start = 'St. Regis'
stop = 'Hotel Vitale'

'''Traveler'''
class Traveler:
    def __init__(self, user_preferences, start, stop, budget=200, ambition=[9,17], travel_dates=0, must_see=[], already_seen=[]):
        self.user_preferences = user_preferences
        self.start = start
        self.stop = stop
        self.budget = budget
        self.ambition = ambition
        self.travel_dates = travel_dates
        self.must_see = must_see
        self.already_seen = already_seen
        self.tour_length = 60*(ambition[1]-ambition[0])
        self.site_name_lookup, self.SF_sites, self.travel_matrix, self.hotel_index, self.start_hotel_index, self.stop_hotel_index, self.hotel_travel_matrix = self.gen_real_data()

    def gen_real_data(self):
        '''open all the pre-calculated pickle files'''

        '''sites'''
        site_index = pkl.load(open('site_index.pkl', 'rb'))
        highlight_bonuses = pkl.load(open('highlight_bonuses.pkl', 'rb'))

        visit_length = pkl.load(open('visit_length.pkl', 'rb'))
        visit_cost = pkl.load(open('entrance_fees.pkl', 'rb'))

        '''paths'''
        travel_times = pkl.load(open('travel_times.pkl', 'rb'))
        travel_costs = pkl.load(open('travel_costs.pkl', 'rb'))

        '''utilize user_preferences to get the needed data structures'''

        site_names = [pref[0] for pref in self.user_preferences][:200]
        similarities = [pref[1] for pref in self.user_preferences][:200]

        '''sites to dis-include'''
        '''includes adding sights that are must seen & others that are already seen'''
        remove = list(set(site_index).difference(set(site_names + self.must_see))) + self.already_seen
        remove_indices = [site_index.index(r_) for r_ in remove]

        '''manifest the returned data structures with the user preferences in mind'''

        '''SF_sites'''
        user_highlight_bonuses = [highlight_bonuses[n] for n in range(len(highlight_bonuses)) if n not in remove_indices]
        user_visit_length = [visit_length[n] for n in range(len(visit_length)) if n not in remove_indices]
        user_visit_cost = [visit_cost[n] for n in range(len(visit_cost)) if n not in remove_indices]

        SF_sites = np.asarray(list(zip(similarities, user_highlight_bonuses, user_visit_length, user_visit_cost)))

        '''travel_matrix'''
        mask = np.ones(np.shape(travel_times), dtype=bool)
        mask[:,remove_indices] = False
        mask[remove_indices,:] = False
        one_d = len(travel_times) - len(remove_indices)

        user_travel_times = travel_times[mask].reshape([one_d, one_d])
        user_travel_costs = travel_costs[mask].reshape([one_d, one_d])

        travel_matrix = [user_travel_times, user_travel_costs]

        '''site_name_lookup'''
        user_site_indices = range(len(site_index)-len(remove_indices))
        user_sites = [site_index[id_] for id_ in user_site_indices]
        site_name_lookup = dict(zip(user_site_indices, user_sites))

        '''hotel information'''
        '''Possible hotels: ['The Fairmont San Francisco', 'Hotel Boheme', 'St. Regis', 'Nineteen 06 Mission', 'Oceanview Motel', 'Inn at the Presidio', 'Hotel Vitale', 'The Wharf Inn', 'Hotel Zephyr', 'Seaside Inn', 'Hotel Nikko', 'Hotel Zelos', "Noe's Nest Bed and Breakfast in San Francisco"]'''

        hotel_index = pkl.load(open('hotel_index.pkl', 'rb'))
        hotel_times = pkl.load(open('hotel_times.pkl', 'rb'))
        hotel_travel_costs = pkl.load(open('hotel_travel_costs.pkl', 'rb'))

        start_hotel_index = hotel_index.index(self.start) + len(user_site_indices)
        stop_hotel_index = hotel_index.index(self.stop) + len(user_site_indices)

        mask2 = np.ones(np.shape(hotel_times), dtype=bool)
        mask2[:, remove_indices] = False
        hotel_one_d = len(user_site_indices)
        user_hotel_times = hotel_times[mask2].reshape([len(hotel_index), hotel_one_d])
        user_hotel_travel_costs = hotel_travel_costs[mask2].reshape([len(hotel_index), hotel_one_d])

        hotel_travel_matrix = [user_hotel_times, user_hotel_travel_costs]

        return site_name_lookup, SF_sites, travel_matrix, hotel_index, start_hotel_index, stop_hotel_index, hotel_travel_matrix

'''Path'''
class Path:
    def __init__(self, from_attraction, to_attraction, traveler):

        self.SF_sites = traveler.SF_sites
        self.hotel_travel_matrix = traveler.hotel_travel_matrix
        self.travel_matrix = traveler.travel_matrix

        self.from_attraction = int(from_attraction)
        self.to_attraction = int(to_attraction)

        self.name = f'({self.from_attraction}, {self.to_attraction})'

        self.similarity, self.highlight_bonus, self.visit_length, self.visit_cost, self.travel_time, self.travel_cost = self.get_hotel_or_attraction()

        self.path_cost = self.visit_cost + self.travel_cost
        self.path_length = self.visit_length + self.travel_time

        self.path_fitness = self.get_fitness()

    ''' define this to understand whether the point assessed is a hotel or an attraction'''
    def get_hotel_or_attraction(self):
        similarity = 0
        highlight_bonus = 0
        visit_length = 0
        visit_cost = 0
        travel_time = 0
        travel_cost = 0

        if (self.from_attraction >= len(self.SF_sites) and self.to_attraction >= len(self.SF_sites)) :

            similarity = 0
            highlight_bonus = -10000
            visit_length = 10
            visit_cost = 0
            travel_time = 10000
            travel_cost = 10000

        elif self.from_attraction >= len(self.SF_sites):

            similarity = self.SF_sites[self.to_attraction, 0]
            highlight_bonus = self.SF_sites[self.to_attraction, 1]

            visit_length = self.SF_sites[self.to_attraction, 2]
            visit_cost = self.SF_sites[self.to_attraction, 3]

            travel_time = self.hotel_travel_matrix[0][self.from_attraction - len(self.SF_sites)][self.to_attraction]
            travel_cost = self.hotel_travel_matrix[1][self.from_attraction - len(self.SF_sites)][self.to_attraction]

        elif self.to_attraction >= len(self.SF_sites):

            similarity = 1
            highlight_bonus = 1

            visit_length = 10
            visit_cost = 0

            travel_time = self.hotel_travel_matrix[0][self.to_attraction - len(self.SF_sites)][self.from_attraction]
            travel_cost = self.hotel_travel_matrix[1][self.to_attraction - len(self.SF_sites)][self.from_attraction]

        else:
            similarity = self.SF_sites[self.to_attraction, 0]
            highlight_bonus = self.SF_sites[self.to_attraction, 1]

            visit_length = self.SF_sites[self.to_attraction, 2]
            visit_cost = self.SF_sites[self.to_attraction, 3]

            travel_time = self.travel_matrix[0][self.from_attraction][self.to_attraction]
            travel_cost = self.travel_matrix[1][self.from_attraction][self.to_attraction]

        return similarity, highlight_bonus, visit_length, visit_cost, travel_time, travel_cost

    def get_fitness(self, B1=0.5, B2=1.5, B3=10):

        similarity = self.similarity

        if self.similarity > 5:
            similarity_bonus = .75 + (self.similarity/10)
            similarity = similarity**similarity_bonus

        if self.highlight_bonus > 1:
            highlight = 2*self.highlight_bonus
        else:
            highlight = self.highlight_bonus

        path_fitness = -(float(highlight)*B3*float(similarity)) - B2*float(self.travel_time) - (B2*float(self.travel_cost + self.visit_cost))

        return path_fitness

    def __repr__(self):
        return self.name

'''Route'''
class Route:

    def __init__(self, route, traveler):
        self.traveler = traveler
        self.route = route
        self.name = route
        self.route_nodes = len(route)

        self.hotel_travel_matrix = traveler.hotel_travel_matrix

        self.SF_sites = traveler.SF_sites
        self.travel_matrix = traveler.travel_matrix
        self.budget = traveler.budget
        self.tour_length = traveler.tour_length

        self.start = traveler.start_hotel_index
        self.stop = traveler.stop_hotel_index

        self.paths = self.get_paths()

        self.route_cost = sum([path.path_cost for path in self.paths])
        self.route_length = sum([path.path_length for path in self.paths])

        self.available_budget = self.budget - self.route_cost
        self.available_tour_length = self.tour_length - self.route_length

        self.route_fitness = self.fitness()

    def get_paths(self):
        interim = []
        for n in range(self.route_nodes-1):
            path = Path(self.route[n], self.route[n+1], self.traveler)
            interim.append(path)
        return interim

    def fitness(self, B1=1.3, B2=2):

        '''get score for path'''
        paths_fitness = 0.1*sum([path.path_fitness for path in self.paths])

        if (self.route_cost < self.budget):
            budget_score = -(self.route_cost/self.budget)*paths_fitness

            if (self.route_cost > 0.75*self.budget):
                budget_score = 4*budget_score
        else:
            budget_score = B1*(abs(paths_fitness))

        if (self.route_length < self.tour_length):
            tour_length_score = -0.5*(self.route_length/self.tour_length)*paths_fitness

            if (self.route_length > 0.75*self.tour_length):
                tour_length_score = 4*tour_length_score

        else:
            tour_length_score = B1*(abs(paths_fitness))

        fitness_score = paths_fitness + budget_score + tour_length_score

        return fitness_score

    def __repr__(self):
        return str(self.route)
                
'''******'''
'''Initial Routes'''
class initial_routes():
    def __init__(self, traveler, num_routes=1):
        self.traveler = traveler
        self.tour_length = traveler.tour_length
        self.SF_sites = traveler.SF_sites
        self.budget = traveler.budget
        self.start = traveler.start_hotel_index
        self.stop = traveler.stop_hotel_index
        self.num_routes = num_routes

    def create_initial_routes(self):
        routes = []
        count = 0

        while count < self.num_routes:

            sites = range(len(self.SF_sites))
            route = [self.start]
            not_valid = []

            available_budget = self.budget
            available_tour_length = self.tour_length

            while ((available_budget > 0.1*self.budget) | (available_tour_length > 0.1*self.tour_length)) & ((available_budget > 0) & (available_tour_length > 0)):

                '''define the from attraction'''
                if len(route) == 1:
                    from_attraction = self.start

                '''define the to attraction'''
                possibilities = list(set(sites).difference(set(route + not_valid + [self.stop])))

                if len(possibilities) == 0:
                        break
                to_attraction = np.random.choice(possibilities)

                '''information needed'''
                candidate_path = Path(from_attraction, to_attraction, self.traveler)

                path_to_stop = Path(to_attraction, self.stop, self.traveler)

                '''check that there is enough budget/time left to get to get to the stop location'''
                if (available_budget > (candidate_path.path_cost + path_to_stop.path_cost)) & (available_tour_length > (candidate_path.path_length + path_to_stop.path_length)):
                    available_budget = available_budget - candidate_path.path_cost
                    available_tour_length = available_tour_length - candidate_path.path_length

                    route.append(to_attraction)
                    from_attraction = to_attraction

                else:
                    '''if not, go back to the drawing board, and add the current site to the discard pile'''
                    not_valid.append(to_attraction)
                    '''add the budget and time back'''

            route.append(self.stop)
            route_class = Route(route, self.traveler)

            routes.append(route_class)
            count +=1

        return routes

'''Update Route'''
class get_updated_route():

    def __init__(self, current_route):
        self.name = 'get_new_route'
        self.current_route = current_route
        self.length = len(current_route.route)

    def roulette(self):
        next_ = 0
        node_index =[1]

        if self.length >= 4:
            next_ = np.random.choice(range(4), p = [.3, .3, .3, .1])
            node_index = np.random.choice(list(range(1, self.length-1)), size=np.random.randint(2, self.length-1), replace=False)

        return next_, node_index

    def next_move(self, next_, node_index):
        if self.length < 3:
            return self.current_route
        if next_ == 0:
            return self.insertion(node_index[0])
        elif next_ == 1:
            return self.deletion(node_index[0])
        elif next_ == 2:
            return self.mutation(node_index)
        else:
            return self.diversification(node_index)

    def softmax(self, x):
        e_x = np.exp(np.min(x)-x)
        return e_x / e_x.sum()

    def get_neighbors(self, node_index):
        ''' neighbors are any individuals that when inserted for a node, do not exceed the defined cost limits'''

        current_route = self.current_route

        node = current_route.route[node_index]
        before_node = current_route.route[node_index - 1]
        after_node = current_route.route[node_index + 1]

        all_sites = set(range(len(current_route.SF_sites)))
        possible_sites = all_sites.difference(set(current_route.route))

        '''**prune: for insertions: remove all node locations that are outside the available budget and tour length'''
        mask = np.ones(len(current_route.SF_sites), dtype=bool)
        mask_values = [node_ for node_ in current_route.route if node_ < len(current_route.SF_sites)]
        mask[mask_values] = False

        if (before_node >= len(current_route.SF_sites)) & (after_node >= len(current_route.SF_sites)):

            mask = np.ones(np.shape(current_route.hotel_travel_matrix[0][1,:]), dtype=bool)
            mask_values = [node_ for node_ in current_route.route if node_ < len(current_route.SF_sites)]
            mask[mask_values] = False

            new_travel_costs = (current_route.hotel_travel_matrix[1][before_node - len(current_route.SF_sites)] + current_route.hotel_travel_matrix[1][after_node  - len(current_route.SF_sites)])[mask]
            new_final_budget = np.array([current_route.available_budget]*len(new_travel_costs)) - current_route.SF_sites[:,3][mask] + new_travel_costs

            new_travel_lengths = (current_route.hotel_travel_matrix[0][before_node - len(current_route.SF_sites)] + current_route.hotel_travel_matrix[0][after_node - len(current_route.SF_sites)])[mask]
            new_final_tour_length = np.array([current_route.available_tour_length]*len(new_travel_lengths)) - current_route.SF_sites[:,2][mask] + new_travel_lengths

        elif before_node >= len(current_route.SF_sites):

            new_travel_costs = (current_route.hotel_travel_matrix[1][before_node - len(current_route.SF_sites)] + current_route.travel_matrix[1][after_node])[mask]
            new_final_budget = np.array([current_route.available_budget]*len(new_travel_costs)) - current_route.SF_sites[:,3][mask] + new_travel_costs

            new_travel_lengths = (current_route.travel_matrix[0][before_node - len(current_route.SF_sites)] + current_route.travel_matrix[0][after_node])[mask]
            new_final_tour_length = np.array([current_route.available_tour_length]*len(new_travel_lengths)) - current_route.SF_sites[:,2][mask] + new_travel_lengths

        elif after_node >= len(current_route.SF_sites):

            new_travel_costs = (current_route.travel_matrix[1][before_node] + current_route.hotel_travel_matrix[1][after_node - len(current_route.SF_sites)])[mask]
            new_final_budget = np.array([current_route.available_budget]*len(new_travel_costs)) - current_route.SF_sites[:,3][mask] + new_travel_costs

            new_travel_lengths = (current_route.travel_matrix[0][before_node] + current_route.hotel_travel_matrix[0][after_node - len(current_route.SF_sites)])[mask]
            new_final_tour_length = np.array([current_route.available_tour_length]*len(new_travel_lengths)) - current_route.SF_sites[:,2][mask] + new_travel_lengths
        else:

            new_travel_costs = (current_route.travel_matrix[1][before_node] + current_route.travel_matrix[1][after_node])[mask]
            new_final_budget = np.array([current_route.available_budget]*len(new_travel_costs)) - current_route.SF_sites[:,3][mask] + new_travel_costs

            new_travel_lengths = (current_route.travel_matrix[0][before_node] + current_route.travel_matrix[0][after_node])[mask]
            new_final_tour_length = np.array([current_route.available_tour_length]*len(new_travel_lengths)) - current_route.SF_sites[:,2][mask] + new_travel_lengths

        '''final indices to check'''
        nodes_indices = [num for num in range(len(current_route.SF_sites)) if num not in current_route.route]
        nodes_to_check = list(zip(nodes_indices, new_final_budget, new_final_tour_length))
        nodes_to_check = np.array(nodes_to_check, dtype=int)

        neighbors_list = [nodes_to_check[n][0] for n in range(len(nodes_to_check)) if (nodes_to_check[n][1] >= 0) & (nodes_to_check[n][2] >= 0)]

        neighbors_fitness = [sum([Path(before_node, neighbor, self.current_route.traveler).path_fitness, Path(neighbor, after_node, self.current_route.traveler).path_fitness]) for neighbor in neighbors_list]

        neighbors = dict(zip(neighbors_list, neighbors_fitness))

        return neighbors

    def insertion(self, node_index):
        neighbors = self.get_neighbors(node_index)
        trials = 0

        while ((len(neighbors) == 0) & (trials < self.length)):
            node_index = np.random.choice(range(1, self.length-1))
            neighbors = self.get_neighbors(node_index)
            trials += 1
            if trials == self.length:
                return self.current_route

        neighbor = np.random.choice(list(neighbors.keys()), p=self.softmax(list(neighbors.values())))
        candidate_0 = self.current_route.route[:node_index] + [neighbor] + self.current_route.route[node_index:]
        candidate_route = Route(candidate_0, self.current_route.traveler)
        return candidate_route

    def deletion(self, node_index):
        candidate_0 = self.current_route.route[:node_index] + self.current_route.route[(node_index+1):]
        candidate_route = Route(candidate_0, self.current_route.traveler)
        return candidate_route

    def diversification(self, indices, mutation_rate=0.15):
        indices = sorted(indices)
        candidate_route = self.current_route
        route_length = self.current_route.route_nodes
        for index in range(indices[0], indices[1]+1):
            if (np.random.random() < mutation_rate):
                candidate_route = get_updated_route(candidate_route).insertion(index)
                if candidate_route.route_nodes > route_length:
                    candidate_route = get_updated_route(candidate_route).deletion(index+1)
        return candidate_route

    def mutation(self, indices, mutation_rate=.8):
        indices = sorted(indices)
        route = self.current_route.route
        n = 0
        while n < len(indices):
            index = indices[n]
            if (np.random.random() < mutation_rate):
                swap_partners = list(set(indices).difference([index]))
                index2 = np.random.choice(swap_partners)
                sswap = sorted([index, index2])
                route = route[:sswap[0]] + [route[sswap[1]]] + route[sswap[0]+1:sswap[1]] + [route[sswap[0]]] + route[sswap[1]+1:]
            n +=1
        candidate_route = Route(route, self.current_route.traveler)
        return candidate_route

    def __repr__(self):
        return self.name

'''GA - Update Generation'''
class get_updated_generation():

    def __init__(self, current_generation, traveler, mutation_rate = 0.01, elite_size = 4):

        self.current_generation = current_generation
        self.traveler = traveler
        self.budget = traveler.budget
        self.tour_length = traveler.tour_length
        self.SF_sites = traveler.SF_sites
        self.travel_matrix = traveler.travel_matrix

        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generation_size = len(current_generation)
        self.ranked_gen = self.get_ranked_gen()
        self.mating_pool = self.get_mating_pool()
        self.new_generation = self.get_new_generation()
        self.mutations = self.get_mutations()
        self.next_generation = self.get_next_generation()

    def get_ranked_gen(self):
        fitness_results = {}
        for i in range(self.generation_size):
            fitness_results[i] = self.current_generation[i].route_fitness
        ranked_gen = sorted(fitness_results.items(), key = lambda x: x[1])
        return ranked_gen

    def get_mating_pool(self):
        selected_parents = []

        for i in range(self.elite_size):
            selected_parents.append(self.ranked_gen[i])

        ranked_array = np.array(self.ranked_gen)
        new_ranked = ranked_array[:,1] - max(ranked_array[:,1])
        cum_sum = np.cumsum(new_ranked).reshape([len(ranked_array), 1])
        cum_perc = 100*cum_sum/sum(new_ranked)
        ranked_array = np.concatenate((ranked_array, cum_sum, cum_perc),1)

        for j in range(min(self.generation_size,200) - self.elite_size):
            pick = np.random.randint(0,101)
            if (pick >= ranked_array[j,3]):
                candidate = self.ranked_gen[j]
                selected_parents.append(candidate)

        mating_pool = []

        for i in range(len(selected_parents)):
            index = selected_parents[i][0]
            mating_pool.append(self.current_generation[index])
        return mating_pool

    def make_children(self, parent1, parent2):
        children = create_children(parent1, parent2).get_children()
        return children

    def get_new_generation(self):
        new_generation = []

        for i in range(self.elite_size):
            new_generation.append(self.mating_pool[i])

        for i in range((len(self.mating_pool)-self.elite_size)):
            parents_indices = np.random.choice(range(len(self.mating_pool)), 2, replace=False)
            parent1 = self.mating_pool[parents_indices[0]]
            parent2 = self.mating_pool[parents_indices[1]]

            children = self.make_children(parent1, parent2)
            new_generation.extend(children)
        return new_generation

    def get_mutations(self):
        mutations = []
        for current_route in self.new_generation:
            if current_route.route_nodes < 3:
                pass
            elif current_route.route_nodes == 3:
                mutation = get_updated_route(current_route).insertion(1)
                mutations.append(mutation)
            else:
                indices = np.random.choice(list(range(1, current_route.route_nodes-1)), size=np.random.randint(2, current_route.route_nodes-1), replace=False)
                chance_index = np.random.choice([0,1])
                if chance_index == 0:
                    mutation = get_updated_route(current_route).mutation(indices)
                    mutations.append(mutation)
                else:
                    mutation = get_updated_route(current_route).diversification(indices)
                    mutations.append(mutation)

        return mutations

    def get_next_generation(self):
        return self.new_generation + self.mutations

'''GA - Create Children'''
class create_children():
    def __init__(self, parent1, parent2):
        self.traveler = parent1.traveler
        self.parent1 = parent1
        self.parent2 = parent2
        self.chromosome1 = parent1.route
        self.chromosome2 = parent2.route
        self.SF_sites = parent1.SF_sites
        self.travel_matrix = parent1.travel_matrix
        self.budget = parent1.budget
        self.tour_length = parent1.tour_length
        self.c1_length = parent1.route_nodes
        self.c2_length = parent2.route_nodes
        self.start = self.chromosome1[0]
        self.stop = self.chromosome1[-1]
        self.crossover_vals = list(set(self.chromosome1[1:-1]).intersection(self.chromosome2[1:-1]))
        self.children =[]

    def remove_duplicates(self, child):
            indices = {site:[i for (i,x) in enumerate(child[1:-1]) if x == site] for site in child[1:-1]}
            indices_to_remove =[]
            for key, value in indices.items():
                if len(value) > 1:
                    index_to_remove = np.random.choice(value)
                    indices_to_remove.append(index_to_remove)
                    value.remove(index_to_remove)
            new_child = [child[0]] + [child[1:-1][index] for index in range(len(child[1:-1])) if index not in indices_to_remove] + [child[-1]]
            return new_child

    def get_children(self):
        if self.parent1.route_nodes <= self.parent2.route_nodes:
            shorter = self.chromosome1
            longer = self.chromosome2
        else:
            shorter = self.chromosome2
            longer = self.chromosome1

        if len(shorter) < 3:
            return self.get_child_routes([shorter, longer])
        if len(shorter) == 3:
            return self.extend_stock()
        elif (self.chromosome1 == self.chromosome2):
            return(self.diversify_stock(shorter))
        elif (len(self.crossover_vals) == 0):
            return self.random_crossover(shorter, longer)
        else:
            return self.ordered_crossover()

    def extend_stock(self):
        index = 1
        child1 = get_updated_route(self.parent1).insertion(index)
        child2 = get_updated_route(self.parent2).insertion(index)
        children = [child1, child2]
        return children

    def diversify_stock(self, shorter):
        indices = np.random.choice(list(range(1, len(shorter)-1)), 2, replace=False)
        child1 = get_updated_route(self.parent1).diversification(indices)
        child2 = get_updated_route(self.parent2).diversification(indices)
        children = [child1, child2]
        return(children)

    def random_crossover(self, shorter, longer):

        rc_1 = np.random.choice(range(1, len(shorter)-1), np.random.randint(1,(len(shorter)-2)), replace=False)
        rc_2 = np.random.choice(range(1, len(longer)-1), len(rc_1), replace=False)

        for n in range(len(rc_1)):
            swap1 = shorter[rc_1[n]]
            swap2 = longer[rc_2[n]]

            shorter[rc_1[n]] = swap2
            longer[rc_2[n]] = swap1

        children = self.get_child_routes([shorter, longer])
        return children

    def ordered_crossover(self):
        n = 0
        child1 = self.chromosome1
        child2 = self.chromosome2
        used = []
        crossover_vals = self.crossover_vals

        while (len(crossover_vals) > 0) & (n < 5):
            if (len(child1) > 3) & (len(child2) > 3):
                cross_val = np.random.choice(crossover_vals)
                used.append(cross_val)

                swap1 = child1.index(cross_val)
                swap2 = child2.index(cross_val)

                interim_c1 = child1[:swap1]+child2[swap2+1:]
                interim_c2 = child2[:swap2]+child1[swap1+1:]

                child1 = interim_c1
                child2 = interim_c2

                crossover_vals=list(set(child1[1:-1]).intersection(set(child2[1:-1]+used)))

                n+=1

                if len(self.crossover_vals) == 0:
                    break
            else:
                n=5

        child1 = self.remove_duplicates(child1)
        child2 = self.remove_duplicates(child2)

        children = self.get_child_routes([child1, child2])
        return children

    def get_child_routes(self, children):
        children = [self.remove_duplicates(child) for child in children]
        children = [Route(child, self.traveler) for child in children]
        return children

'''GA - GA Move'''
class ga_move():
    def __init__(self, current_generation):
        self.current_generation = current_generation
        self.gen_size = len(current_generation)
        self.SF_sites = current_generation[0].SF_sites
        self.travel_matrix = current_generation[0].travel_matrix

    def update_generation(self):
        next_generation = get_updated_generation(self.current_generation, self.current_generation[0].traveler).next_generation
        return next_generation

'''GA - Score Generation'''
class score_generation():
    def __init__(self, current_generation):
        self.current_generation = current_generation
        self.generation_size = len(current_generation)
        self.routes_fitness = [route.route_fitness for route in current_generation]
        self.average_fitness = self.average_fitness()
        self.best_route = self.best_route()
        self.best_fitness = self.best_fitness()

    def average_fitness(self):
        average_fitness = sum(self.routes_fitness)/self.generation_size
        return average_fitness

    def best_route(self):
        best_route_index = np.argmin(self.routes_fitness)
        return self.current_generation[best_route_index]

    def best_fitness(self):
        best_score_index = np.argmin(self.routes_fitness)
        return self.routes_fitness[best_score_index]

'''GA - ga_plot'''
class ga_plot():

    def __init__(self, traveler, gen_size=15, elite_size=2, mutation_rate=0.1, num_generations=100):

        '''traveler deets'''
        self.traveler = traveler
        self.budget = traveler.budget
        self.tour_length = traveler.tour_length
        self.SF_sites = traveler.SF_sites
        self.travel_matrix = traveler.travel_matrix
        self.start = traveler.start_hotel_index
        self.stop = traveler.stop_hotel_index

        self.gen_size = gen_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations

    def evolve(self):
        gen_count = 0

        current_generation = initial_routes(self.traveler, 20).create_initial_routes()

        '''keep track of score from initial selection'''
        progress = []
        routes = []
        routes_val = []

        gen_performance = score_generation(current_generation)

        progress.append(gen_performance.best_fitness)
        routes.append(gen_performance.best_route)

        while gen_count < self.num_generations:
            print(gen_count, gen_performance.generation_size, gen_performance.best_fitness)

            current_generation = ga_move(current_generation).update_generation()
            gen_performance = score_generation(current_generation)

            progress.append(gen_performance.best_fitness)
            routes.append(gen_performance.best_route)
            gen_count +=1

        starting_score = progress[0]
        best_score_index = np.argmin(np.array(progress))
        best_score = progress[best_score_index]
        best_route = routes[best_score_index].route

        print((best_score - starting_score), best_score, best_route)

        return progress, routes, best_score_index
