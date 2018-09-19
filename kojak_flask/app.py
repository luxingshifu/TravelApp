from flask import Flask, abort, render_template, jsonify, request
from flask import Flask, render_template
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map, icons
from get_route_geolocations import get_route_geolocations

dict_directions = {}

app = Flask('Kojak_Flask')

app.config['GOOGLEMAPS_KEY'] = "AIzaSyB-9ZI7M3DneS6lPAZAItAlnNPZ5TpbgdU"

GoogleMaps(app)

@app.route('/')
def index():
    actual_route = ['The Fairmont San Francisco', 'New United States Mint', 'Vinicola de Coppola',
    'Japantown', 'Madame Tussauds San Francisco', 'San Francisco State University',
    'Cellarmaker Brewing Company', 'Autodesk Gallery', 'Sea Lion Center', 'Alta Plaza Park',
    'San Francisco Opera', 'Golden Gate Bridge', 'San Francisco City Hall', 'Louise M. Davies Symphony Hall',
    'Labyrinth of Cultures', 'Sing Fat Co. building', 'Hotel Zelos']
    dct = get_route_geolocations(actual_route)
    return render_template('gmaps2.html',results = dct)

app.run(debug=True)
