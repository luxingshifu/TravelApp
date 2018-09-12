from flask import Flask, abort, render_template, jsonify, request
from api import make_prediction
# import itin_gen.api_itin as api_itin
# from api_itin import itin_generator

app=Flask('TravelApp')

@app.route('/predict',methods=['POST'])
def do_prediction():
    if not request.json:
        abort(400)

    data=request.json
    print(data)
    print("Zup yo")

    response=make_prediction(data)



    #print("zuppppppp")
    #print(type(data['city']))


    return jsonify(response)


@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
  app.run(debug=True)
