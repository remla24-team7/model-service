from flask import Flask, request, jsonify
import keras

# from lib_ml import preprocess

model: keras.Model = None

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    input = request.json["input"]
    return jsonify({"input": input})
    # preprocess()
    # model.predict()


if __name__ == "__main__":
    model = keras.models.load_model("model.h5")
    app.run(host="0.0.0.0", debug=True)
