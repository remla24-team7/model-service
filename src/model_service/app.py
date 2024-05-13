from flask import Flask, request, jsonify
from lib_ml.model import Model

model: Model = None

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    input = request.json["input"]
    return jsonify(model.predict(input))


if __name__ == "__main__":
    model = Model(
        model_path="model.h5",
        tokenizer_path="tokenizer.joblib",
        encoder_path="encoder.joblib",
    )

    app.run(host="0.0.0.0", debug=True)
