from lib_ml.model import Model
from flask import Flask, request, jsonify

model = Model(
    model_path="model.h5",
    tokenizer_path="tokenizer.joblib",
    encoder_path="encoder.joblib",
)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    input = request.json["input"]
    [output] = model.predict([input])
    return jsonify(output)


if __name__ == "__main__":
    app.run()
