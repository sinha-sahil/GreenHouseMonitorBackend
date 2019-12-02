#!venv-flask/bin/python
#!venv-flask/bin/flask
import base64
from flask      import Flask, jsonify, make_response, request, abort
from flask_cors import CORS

from machine_model import prediction

app = Flask(__name__)
CORS(app)

error_payload_not_found =   {   "error" : "payload not found"     }

def error_in_result(e):         
    return {"error in evaluate Result" : str(e)}

def prepare_output(a):
    return { "result" : str(a)}

def use_model(image):
    predictor = prediction()
    x = predictor.predict(image)
    return x

@app.route('/')
def index():
    return "Welcome to Green House Monitoring System. Pass data as image to evaluate results"

@app.route('/evaluate', methods=['POST'])
def evaluateResult():
    if request.json and 'payload' in request.json:
        try:
            payload = request.json['payload']
            x = use_model(payload)
            return jsonify(prepare_output(x)), 201
            
        except Exception as e:
            return jsonify(error_in_result(e)), 201

    return jsonify(error_payload_not_found), 201

@app.errorhandler(400)
def payloadNotPassed(error):
    return make_response(jsonify({'error': error}), 400)

@app.errorhandler(405)
def methodNotAllowed(error):
    return make_response(jsonify({'error': 'Please do a post call to this api with image'}), 405)

@app.errorhandler(404)
def notFound(error):
    return make_response(jsonify({'error': 'Please do a POST call with image in order to get evaluation results'}), 404)

@app.errorhandler(500)
def internalServerError(error):
    return make_response(jsonify({'error': ('Something went wrong with server' + error)}), 500)

if __name__ == '__main__':
    app.run(debug=True)