from flask import Flask, abort, render_template, jsonify, request
from api import make_prediction
import pandas as pd
from werkzeug.datastructures import ImmutableMultiDict
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
metadata=MetaData()

engine=create_engine('postgresql://localhost/postgres')



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
        response=make_prediction(d)
        # route=response['actual_route']
        result = response['recommendations']
        return render_template('index2.html', result = result)

    return render_template('index2.html')


@app.route('/',methods=['GET','POST'])
def index():

    if request.method == 'POST':
        data=request.form.to_dict()
        d={k:float(data[k]) for k in data.keys()}
        response=make_prediction(d)
        result = response['actual_route']
        return render_template('index.html', result = result)

    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
