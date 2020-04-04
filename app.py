from flask import Flask, request, jsonify
import numpy as np
import pickle as pickle
import json

tf_idf_load = pickle.load(open("tfidf.pickle", 'rb'))
classifier = pickle.load(open("SVM.pickle", 'rb'))
calibrated = pickle.load(open("calibrated_for_probability.pickle", 'rb'))

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    data = request.args.get("data", None)
    tranformed_data = tf_idf_load.transform([data])
    prediction = np.array2string(classifier.predict(tranformed_data)[0])
    probability = np.array2string(calibrated.predict_proba(tranformed_data)[:,1][0])
    return jsonify({"polarity" : prediction, "probability": probability})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

