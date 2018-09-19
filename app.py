from flask import Flask, abort, render_template, jsonify, session, request, redirect, url_for, flash, make_response
from api import make_prediction
import pandas as pd
import os
import pickle as pkl
from flask_googlemaps import GoogleMaps, Map
# from boto.s3.connection import S3Connection
from werkzeug.datastructures import ImmutableMultiDict
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
from get_route_geolocations import get_route_geolocations

dict_directions = {}


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


# with open('good_data/site_index.pkl','rb') as f:
#     fpl=pkl.load(f)
# user_columns=['nature','history','culture','life']+fpl
#

# conn.execute(user_table.insert(),[{'index':22,'cat':2,'dog':3,'chicken':4,'bat':4,'wambat':3}])



# from jinja import Environment, FileSystemLoader
# e = Environment(loader=FileSystemLoader('templates/'),use_memcache=False)
# import itin_gen.api_itin as api_itin
# from api_itin import itin_generator

app=Flask('TravelApp')
app.secret_key='asdfjkl;'
app.config['GOOGLEMAPS_KEY']="AIzaSyB-9ZI7M3DneS6lPAZAItAlnNPZ5TpbgdU"
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
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",flush=True)
    good_route=session['actual_route']
    print(good_route,flush=True)
    # print(type(actual_route),flush=True)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",flush=True)
    new_route=[str(x) for x in good_route]
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",flush=True)
    dct = get_route_geolocations(good_route)
    print(dct,flush=True)
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",flush=True)


    return render_template('gmaps2.html',results = dct)




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

        # res = make_response("Modifying the cookie")
        # res.set_cookie('start', False, max_age=60*60*24*365*2)
        # set_cookie('start', False, max_age=60*60*24*365*2)
        # ct+=1
        session['start']=False
        res2 = make_response("YOLO")
        res2.set_cookie('start', 'False', max_age=60*60*24*365*2)
        print("***************************DDDDADAAAAAATTTTTTAAAAAA****************",flush=True)
        data=request.form.to_dict()
        # print(data,flush=True)

        d={k:data[k] for k in data.keys()}
        print(d,flush=True)
        print("***************************DDDDADAAAAAATTTTTTAAAAAA****************",flush=True)
        d2={k:d[k] for k in d.keys()}
        response=make_prediction(d)
        # route=response['actual_route']
        result = response['recommendations']
        session['actual_route']=response['actual_route']
        recs={clean_string(x[0]):x[1] for x in result}
        uservector={**d2,**recs}
        conn.execute(user_table.insert(),[uservector])

        # print(uservector)

        return render_template('index2.html', result = result[:20])

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
            d['starttime']=9
            d['finishtime']=17
            d['budget']=0



            response=make_prediction(d)
            route = response['actual_route']
            result={'route':route,'nature':nature,'history':history,'culture':culture,'life':life}

            return render_template('index.html', result = result)
        else:
            result={'nature':0,'history':0,'culture':0,'life':0}
            return render_template('index.html',result=result)

    # print(f"The request method is {request.method}",flush=True)

    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
