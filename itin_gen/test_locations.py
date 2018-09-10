def get_start_and_stop(SF_sites):
    import numpy as np

    sites = list(SF_sites.index)
    start = np.random.choice(sites)
    stop = start
    return start, stop
