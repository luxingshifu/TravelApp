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
