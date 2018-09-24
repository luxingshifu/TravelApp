from flask import Flask, abort, render_template, jsonify, session, request, redirect, url_for, flash, make_response
from api import make_prediction
from flask_googlemaps import GoogleMaps, Map
import pandas as pd
import os
from flask_googlemaps import GoogleMaps, Map
# from boto.s3.connection import S3Connection
from werkzeug.datastructures import ImmutableMultiDict
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
import itin_gen.get_route_geolocations as get_route_geolocations
from get_route_geolocations import get_route_geolocations

GOOGLEMAPS_KEY = os.environ['GOOGLEMAPS_KEY']
# DATABASE_URL = os.environ['DATABASE_URL']

metadata=MetaData()
engine=create_engine('postgresql://localhost/postgres')
# engine=create_engine(DATABASE_URL)
conn=engine.connect()
metadata.reflect(bind=engine)
user_table=metadata.tables['user_data']



def clean_string(bla):
    if '(' in bla:
        ind=bla.index('(')
        if bla[ind-1] != ' ':
            return bla[:ind]
        else:
            return bla[:ind-1]
    else:
        return bla



# conn.execute(user_table.insert(),[{'index':22,'cat':2,'dog':3,'chicken':4,'bat':4,'wambat':3}])



# from jinja import Environment, FileSystemLoader
# e = Environment(loader=FileSystemLoader('templates/'),use_memcache=False)
# import itin_gen.api_itin as api_itin
# from api_itin import itin_generator

app=Flask('TravelApp')
app.secret_key='asdfjkl;'


app.config['GOOGLEMAPS_KEY']=GOOGLEMAPS_KEY

GoogleMaps(app)

@app.before_first_request
def initialize():
    session['start']=True


@app.route('/map')
def fun():
    # actual_route = ['The Fairmont San Francisco', 'New United States Mint', 'Vinicola de Coppola',
    # 'Japantown', 'Madame Tussauds San Francisco', 'San Francisco State University',
    # 'Cellarmaker Brewing Company', 'Autodesk Gallery', 'Sea Lion Center', 'Alta Plaza Park',
    # 'San Francisco Opera', 'Golden Gate Bridge', 'San Francisco City Hall', 'Louise M. Davies Symphony Hall',
    # 'Labyrinth of Cultures', 'Sing Fat Co. building', 'Hotel Zelos']


    good_route=session['actual_route']

    source = "https://maps.googleapis.com/maps/api/js?key="+GOOGLEMAPS_KEY+"&callback=initMap"





    new_route=[str(x) for x in good_route]
    dct = get_route_geolocations(good_route)


    return render_template('gmaps2.html', results = dct, map = source)


@app.route('/cookie/')
def cookie():
    if not request.cookies.get('foo'):
        res = make_response("YOLO")
        res.set_cookie('foo', 'barrel', max_age=60*60*24*365*2)
    else:
        res = make_response("Value of cookie foo is {}".format(request.cookies.get('foo')))
    return res

@app.route('/trap', methods=['GET','POST'])
def function():

    if request.method == 'POST':

        session['start']=False


        data=request.form.to_dict()
        d={k:data[k] for k in data.keys()}

        d2={k:d[k] for k in d.keys()}

        try:
            response=make_prediction(d)
        except:
            return render_template('error.html')
        # route=response['actual_route']
        result = response['recommendations']
        session['actual_route']=response['actual_route']
        session['starttime']=d['starttime']
        session['finishtime']=d['finishtime']
        session['budget']=d['budget']
        rec_photo=response['rec_photo']
        recs={clean_string(x[0]):x[1] for x in result}
        uservector={**d2,**recs}
        conn.execute(user_table.insert(),[uservector])
        # print(uservector)
        return render_template('index2.html', result = result[:10],rec_photo=rec_photo[:10])
    return render_template('index2.html')


@app.route('/',methods=['GET','POST'])
def index():

    start=session['start']


    if request.method == 'GET':

        # start=request.cookies.get('start')
        # print(start)
        # data=request.form.to_dict()
        # if request.cookies.get(start):
        #     # nature, history, culture, life= 0,0,0,0
        #     d={'nature':0,'history':0,'culture':0,'life':0}
        #     nature, history, culture, life=0,0,0,0
        #
        # else:

        # set_cookie('start', True, max_age=60*60*24*365*2)

        # res = make_response("Value of cookie foo is {}".format(request.cookies.get('foo')))
        # start = request.cookies.get(start)
        if not start:

            view='SELECT nature, history, culture, life FROM user_data'
            d=dict(pd.read_sql(view,engine).iloc[-1])
            #set start initial time
            nature=d['nature']
            history=d['history']
            culture=d['culture']
            life=d['life']
            starttime=float(session['starttime'])
            finishtime=float(session['finishtime'])
            budget=float(session['budget'])

            d['starttime']=starttime
            d['finishtime']=finishtime
            d['budget']=budget

            response=make_prediction(d)

            route = response['actual_route']
            result={'route':route,'nature':nature,'history':history,\
                    'culture':culture,'life':life,'starttime':starttime,\
                    'finishtime':finishtime,'budget':budget}

            return render_template('index.html',result=result)
        else:
            result={'nature':0,'history':0,'culture':0,'life':0}
            return render_template('index.html',result=result)

    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
