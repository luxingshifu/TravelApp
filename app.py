from flask import Flask, abort, render_template, jsonify, request
from api import make_prediction
from api_itin import itin_generator

app=Flask('TravelApp')

@app.route('/predict',methods=['POST'])
def do_prediction():
    if not request.json:
        abort(400)

    data=request.json
    print(data)
    print("Zup yo")

    response1=make_prediction(data)

    progress, routes, best_route, names=itin_generator(response1['recommendations'],alpha=.8,max_iterations=1000)

    actual_route=[names[val] for val in routes[best_route][0]]
    print(f'This is the actual route {actual_route})')


    #print("zuppppppp")
    #print(type(data['city']))


    return jsonify(actual_route)


@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
  app.run(debug=True)
