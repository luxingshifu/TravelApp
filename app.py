from flask import Flask, abort, render_template, jsonify, request, redirect, url_for, flash, make_response
from api import make_prediction
import pandas as pd
import os
import pickle as pkl
# from boto.s3.connection import S3Connection
from werkzeug.datastructures import ImmutableMultiDict
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey

# DATABASE_URL = os.environ['DATABASE_URL']

metadata=MetaData()
engine=create_engine('postgresql://localhost/postgres')
# engine=create_engine(DATABASE_URL)
conn=engine.connect()
metadata.reflect(bind=engine)
user_table=metadata.tables['user_data']


start=True



def clean_string(bla):
    if '(' in bla:
        ind=bla.index('(')
        if bla[ind-1] != ' ':
            return bla[:ind]
        else:
            return bla[:ind-1]
    else:
        return bla


with open('good_data/site_index.pkl','rb') as f:
    fpl=pkl.load(f)
user_columns=['nature','history','culture','life']+fpl


ct=0
# conn.execute(user_table.insert(),[{'index':22,'cat':2,'dog':3,'chicken':4,'bat':4,'wambat':3}])



# from jinja import Environment, FileSystemLoader
# e = Environment(loader=FileSystemLoader('templates/'),use_memcache=False)
# import itin_gen.api_itin as api_itin
# from api_itin import itin_generator

app=Flask('TravelApp')


# def setup_app(app):
#     res = make_response("YOLO")
#     res.set_cookie(start, True, max_age=60*60*24*365*2)

# setup_app(app)


def starting():
    # make the session last indefinitely until it is cleared
    res = make_response("YOLO")
    res.set_cookie('check', 'True', max_age=60*60*24*365*2)
    return 'hello'

app.before_first_request(starting)

# @app.route('/predict',methods=['GET','POST'])
# def do_prediction():
#     if not request.form:
#         abort(400)
#
#     data=request.form
#     print("Here is our lovely data",flush =True)
#     print(data)
#
#     response=make_prediction(data)
#     print("and here is the response.....",flush=True)
#     print(response['actual_route'])
#     print("yep, that was the response", flush=True)
#     return jsonify(response)



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
        data=request.form.to_dict()
        d={k:data[k] for k in data.keys()}
        d2={k:d[k] for k in d.keys()}
        response=make_prediction(d)
        # route=response['actual_route']
        result = response['recommendations']
        recs={clean_string(x[0]):x[1] for x in result}
        uservector={**d2,**recs}
        conn.execute(user_table.insert(),[uservector])

        # print(uservector)

        return render_template('index2.html', result = result[:20])

    return render_template('index2.html')


@app.route('/',methods=['GET','POST'])
def index():

    if request.method == 'GET':
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

        view='SELECT nature, history, culture, life FROM user_data'
        d=dict(pd.read_sql(view,engine).iloc[-1])
        nature=d['nature']
        history=d['history']
        culture=d['culture']
        life=d['life']
        #Let's get the results from the database so we can display the sliders
        #appropriately the next time the user accesses this page.

        # nature=d['nature']
        # history=d['history']
        # culture=d['culture']
        # life=d['life']

        print("hi there, this is what we are feeding in to 'make_prediction'",flush=True)
        print(d)
        print("Hope that was useful",flush=True)
        print(f"The request method is {request.method}",flush=True)
        print(start)
            # d={k:float(data[k]) for k in data.keys()}
        response=make_prediction(d)
        route = response['actual_route']
        # prefs=[1,2,3,4]
        result={'route':route,'nature':nature,'history':history,'culture':culture,'life':life}

        return render_template('index.html', result = result)

    print(f"The request method is {request.method}",flush=True)

    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
