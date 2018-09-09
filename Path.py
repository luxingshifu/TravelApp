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
