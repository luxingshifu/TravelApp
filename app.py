from flask import Flask, abort, render_template, jsonify, session, request, redirect, url_for, flash, make_response
from api import make_prediction
from flask_googlemaps import GoogleMaps, Map
import pandas as pd
import os
import pickle as pkl
import redis
import json
# from boto.s3.connection import S3Connection
from werkzeug.datastructures import ImmutableMultiDict
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
import itin_gen.get_route_geolocations as get_route_geolocations
from itin_gen.get_route_geolocations import get_route_geolocations

root =os.getcwd()
hotel_index=pkl.load(open(root+'/good_data/San_Francisco/hotel_index.pkl','rb'))


GOOGLEMAPS_KEY = os.environ.get('GOOGLEMAPS_KEY')
# DATABASE_URL = os.environ.get('DATABASE_URL')
# LOCAL_DATABASE=os.environ.get('LOCAL_DATABASE')

metadata=MetaData()
engine=create_engine('postgresql://localhost/postgres')

# engine=create_engine(LOCAL_DATABASE,pool_pre_ping=True)
# engine=create_engine(DATABASE_URL,pool_pre_ping=True)
conn=engine.connect()
metadata.reflect(bind=engine)
user_table=metadata.tables['san_francisco_user_data']

r=redis.Redis(
    host='localhost',
    port=6379)



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


@app.route('/map',methods=['GET','POST'])
def fun():
    # actual_route = ['The Fairmont San Francisco', 'New United States Mint', 'Vinicola de Coppola',
    # 'Japantown', 'Madame Tussauds San Francisco', 'San Francisco State University',
    # 'Cellarmaker Brewing Company', 'Autodesk Gallery', 'Sea Lion Center', 'Alta Plaza Park',
    # 'San Francisco Opera', 'Golden Gate Bridge', 'San Francisco City Hall', 'Louise M. Davies Symphony Hall',
    # 'Labyrinth of Cultures', 'Sing Fat Co. building', 'Hotel Zelos']

    # addons=request.form.to_dict()
    # blah=request.get_json(force=True)
    # addons=request.args.getlist('name')
    addons=request.args.get('name')

    print("#############################################################",flush=True)
    print(type(addons),flush=True)
    print("#############################################################",flush=True)
    good_route=session['actual_route']
    source = "https://maps.googleapis.com/maps/api/js?key="+GOOGLEMAPS_KEY+"&callback=initMap"
    new_route=[str(x) for x in good_route]
    dct = get_route_geolocations(good_route)
    # rec_photo=session['rec_photo']
    rec_photo=pkl.loads(r.get('pkl_rec_photo'))
    # print("Here is the recphoto",flush=True)
    # print(rec_photo,flush=True)
    return render_template('new_map.html', results = dct, map = source,rec_photo=rec_photo)



@app.route('/trap', methods=['GET','POST'])
def function():

    if request.method == 'POST':

        session['start']=False


        data=request.form.to_dict()
        d={str(k):data[k] for k in data.keys()}

        try:
            response=make_prediction(d)
        except:
            return render_template('error.html')
        # route=response['actual_route']
        result = response['recommendations']
        pkl_rec_photo=pkl.dumps(response['rec_photo'])
        r.set('pkl_rec_photo',pkl_rec_photo)

        session['actual_route']=response['actual_route']
        session['starttime']=d['starttime']
        session['name']=d['name']
        session['starthotel']=d['starthotel']
        session['endhotel']=d['endhotel']
        session['finishtime']=d['finishtime']
        session['budget']=d['budget']
        rec_photo=response['rec_photo']
        recs={clean_string(x[0]):x[1] for x in result}
        uservector={**d,**recs}
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

            view='SELECT nature, history, culture, life FROM san_francisco_user_data'
            d=dict(pd.read_sql(view,engine).iloc[-1])
            #set start initial time
            nature=d['nature']
            history=d['history']
            culture=d['culture']
            life=d['life']
            starttime=float(session['starttime'])
            finishtime=float(session['finishtime'])
            budget=float(session['budget'])
            d['name']=session['name']
            d['starthotel']=session['starthotel']
            d['endhotel']=session['endhotel']

            d['starttime']=starttime
            d['finishtime']=finishtime
            d['budget']=budget

            response=make_prediction(d)

            route = response['actual_route']
            result={'route':route,'nature':nature,'history':history,\
                    'culture':culture,'life':life,'starttime':starttime,\
                    'finishtime':finishtime,'budget':budget}

            return render_template('index.html',result=result,hotel_index=hotel_index)
        else:
            result={'nature':0,'history':0,'culture':0,'life':0}
            return render_template('index.html',result=result,hotel_index=hotel_index)

    return render_template('index.html',hotel_index=hotel_index)

if __name__ == '__main__':
  app.run(host="0.0.0.0",port=80,debug=True)
