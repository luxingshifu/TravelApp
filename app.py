from flask import Flask, abort, render_template, jsonify, request
from api import make_prediction

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
