from flask import Flask, abort, render_template, jsonify, request
from api import make_prediction
# from jinja import Environment, FileSystemLoader
# e = Environment(loader=FileSystemLoader('templates/'),use_memcache=False)
# import itin_gen.api_itin as api_itin
# from api_itin import itin_generator

app=Flask('TravelApp')


@app.route('/predict',methods=['GET','POST'])
def do_prediction():
    if not request.form:
        abort(400)

    data=request.form
    print("Here is our lovely data",flush =True)
    print(data)

    response=make_prediction(data)
    print("and here is the response.....",flush=True)
    print(response['actual_route'])
    print("yep, that was the response", flush=True)
    return jsonify(response)
    # return jsonify(result)

# @app.route('/newroute',methods=['POST'])
# def test():
#
#     if not request.json:
#         abort(400)
#     print('Here we are!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',flush=True)
#     data=request.json
#     print(data,flush=True)
#     print('Still going!!!!!!!!!!!!!!!!!!!!!')
#     response=make_prediction(data)
#     print('yippEEEEEEEEE',flush=True)
#     print(response['actual_route'],flush=True)
#     print('yes that was the acutal route!!!!!!!!!!!!!!!!!!',flush=True)
#     res=response['actual_route']
#     return render_template('index.html',result = ['damien', 'loves', 'chocolate'])

@app.route('/',methods=['GET','POST'])
def index():

    if request.method == 'POST':
        data=request.form
        print('Here is that data you requested!',flush=True)
        print(data)
        print('Hope you enjoyed your data',flush=True)
        print('And here is data!',flush=True)
        gd=[data['Nature_key'],data['History_key'],data['Culture_key'],data['Life_key']]
        print(gd,flush=True)
        print('that was naaattttuuuuurrreee',flush=True)
        try:
            response=make_prediction(gd)
        except:
            response={'actual_route':'who knows'}
        result = response['actual_route']
        return render_template('index.html', result = result)

    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
