###hello
import pickle as pkl
import string

proxy_dct=pkl.load(open('good_data/proxy_dct.pkl','rb'))


# We need to 1) flip and 2) shift by 1.
def reindex(names):
    names2=names[::-1]
    names3=[]
    for i in range(len(names)):
        names3.append(names2[((i-1)%len(names))])
    return names3


def get_route_geolocations(route):
    import pickle as pkl
    _, attr_by_geolocations = pkl.load(open('important_items.pkl', 'rb'))
    hotels_index = pkl.load(open('hotel_geocoords_dct.pkl', 'rb'))

    alphabet = list(string.ascii_uppercase)

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

    dct_route['start'] = {'location' : start_geocoord, 'name': start, 'tbl_name': (alphabet[0], start)}

    waypoint_geocoords = []
    for waypt in waypts:
        try:
            waypoint_val = attr_by_geolocations[waypt]
        except:
            waypoint_val = attr_by_geolocations[proxy_dct[waypt]]
        waypoint_geocoords.append(waypoint_val)

    waypoint_names = list(zip([waypt for waypt in waypts], alphabet[1:len(route)-1]))

    for n in range(len(waypoint_names)):
        try:
            stop_geocoord = hotels_index[stop]
        except:
            try:
                stop_geocoord  = attr_by_geolocations[stop]
            except:
                stop_geocoord  = attr_by_geolocations[proxy_dct[stop]]

    dct_route['stop'] = {'location' : stop_geocoord, 'name': stop, 'tbl_name': (alphabet[len(route)-1], start)}

    dct_route['waypoints'] = {'locations':[{'location': waypoint_geocoord} for waypoint_geocoord in waypoint_geocoords], 'names':[waypoint_name for waypoint_name in waypoint_names], 'tbl_names': waypoint_names}

    return dct_route
