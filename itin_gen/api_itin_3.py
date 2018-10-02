import set_up_classes
from set_up_classes import Traveler as Traveler
from set_up_classes import Path as Path
from set_up_classes import Route as Route
from set_up_classes import initial_routes as initial_routes
from set_up_classes import create_children as create_children
from set_up_classes import ga_move as ga_move
from set_up_classes import score_generation as score_generation

from set_up_classes import ga_plot as ga_plot

def get_itinerary(user_preferences, start, stop, budget=200, ambition=[9,17]):
    Austin = Traveler(user_preferences, start, stop, budget, ambition)
# ga_plot(traveler, gen_size=35, elite_size=2, mutation_rate=0.1, num_generations=100)
    GA = ga_plot(Austin, 35, 4, 0.1, 30)
    progress, routes, best_score_index = GA.evolve()

    best_route = routes[best_score_index]

    best_route_attractions = []

    for attraction_index in best_route.route:
        print(attraction_index)
        if attraction_index > len(Austin.SF_sites):
            attraction = Austin.hotel_index[attraction_index -len(Austin.SF_sites)]
            best_route_attractions.append(attraction)
        else:
            attraction = Austin.site_name_lookup[attraction_index]
            best_route_attractions.append(attraction)
    return best_route_attractions