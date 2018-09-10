def get_fake_data(user_preferences):

    import numpy as np
    import pandas as pd

    V = len(user_preferences)

    '''generate SF_sites data'''
    similarities = [pref[1] for pref in user_preferences]
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
    site_names = [site for site in SF_sites.index]
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
