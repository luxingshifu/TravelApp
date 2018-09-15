from flask import Flask, abort, render_template, jsonify, request
from api import make_prediction
import pandas as pd
import os
import pickle as pkl
# from boto.s3.connection import S3Connection
from werkzeug.datastructures import ImmutableMultiDict
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey

DATABASE_URL = os.environ['DATABASE_URL']

metadata=MetaData()
# engine=create_engine('postgresql://localhost/postgres')
engine=create_engine(DATABASE_URL)
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


with open('/Users/williamcottrell72/MyGit/TravelApp/data/site_index.pkl','rb') as f:
    fpl=pkl.load(f)

user_columns=['nature','history','culture','life']+fpl


# conn.execute(user_table.insert(),[{'index':22,'cat':2,'dog':3,'chicken':4,'bat':4,'wambat':3}])



# from jinja import Environment, FileSystemLoader
# e = Environment(loader=FileSystemLoader('templates/'),use_memcache=False)
# import itin_gen.api_itin as api_itin
# from api_itin import itin_generator

app=Flask('TravelApp')


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

@app.route('/trap', methods=['GET','POST'])
def function():

    if request.method == 'POST':
        data=request.form.to_dict()
        d={k:float(data[k]) for k in data.keys()}
        d2={k:d[k] for k in d.keys()}
        response=make_prediction(d)
        # route=response['actual_route']
        result = response['recommendations']
        recs={clean_string(x[0]):x[1] for x in result}
        uservector={**d2,**recs}
        conn.execute(user_table.insert(),[uservector])

        # print(uservector)

        return render_template('index2.html', result = result)

    return render_template('index2.html')


@app.route('/',methods=['GET','POST'])
def index():

    if request.method == 'GET':
        # data=request.form.to_dict()
        view='SELECT nature, history, culture, life FROM user_data'
        d=dict(pd.read_sql(view,engine).iloc[-1])
        print("hi there, this is what we are feeding in to 'make_prediciont'",flush=True)
        print(d)
        print("Hope that was useful",flush=True)
        # d={k:float(data[k]) for k in data.keys()}
        response=make_prediction(d)
        result = response['actual_route']
        return render_template('index.html', result = result)

    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
