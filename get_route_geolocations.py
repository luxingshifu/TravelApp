import pickle as pkl

proxy_dct=pkl.load(open('proxy_dct.pkl','rb'))

def get_route_geolocations(route):
    import pickle as pkl
    _, attr_by_geolocations = pkl.load(open('important_items.pkl', 'rb'))
    hotels_index = pkl.load(open('hotel_geocoords_dct.pkl', 'rb'))

    start = route[0]
    waypts = route[1:-1]
    stop = route[-1]
    dct_route = {}

    try:
        start_geocoord = hotels_index[start]
    except:
        try:
            start_geocoord = attr_by_geolocations[start]
        except:
            start_geocoord = attr_by_geolocations[proxy_dct[start]]

    dct_route['start'] = {'location' : start_geocoord, 'name': start}

    try:
        stop_geocoord = hotels_index[stop]
    except:
        stop_geocoord =  attr_by_geolocations[stop]

    dct_route['stop'] = {'location' : stop_geocoord, 'name': stop}

    waypoint_geocoords = [attr_by_geolocations[waypt] for waypt in waypts]
    waypoint_names = [waypt for waypt in waypts]

    dct_route['waypoints'] = {'locations':[{'location': waypoint_geocoord} for waypoint_geocoord in waypoint_geocoords], 'names':[waypoint_name for waypoint_name in waypoint_names]}

    return dct_route

    #waypoints
