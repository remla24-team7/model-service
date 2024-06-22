from lib_ml.model import Model
from flask import Flask, request, jsonify, Response
from flasgger import Swagger
from prometheus_flask_exporter import PrometheusMetrics
import prometheus_client
import time

model = Model(
    model_path="model/model.keras",
    tokenizer_path="model/tokenizer.joblib",
    encoder_path="model/encoder.joblib",
    sequence_length=200,
)

app = Flask(__name__)
app.config["SWAGGER"] = {"title": "Phishing Detection API"}
swagger = Swagger(app)

metrics = PrometheusMetrics(app)
metrics.info('app_backend_info', 'Application backend info', version='1.0.0')

requests_counter = prometheus_client.Counter('requests_counter', 'Number of requests')
agree_counter = prometheus_client.Counter('agree_counter', 'Number of times users agree with the result')
disagree_counter = prometheus_client.Counter('disagree_counter', 'Number of times users disagree with the result')
legitimate_counter = prometheus_client.Counter('legitimate_counter', 'Number of legitimate URLs')
phishing_counter = prometheus_client.Counter('phishing_counter', 'Number of phishing URLs')
predict_time_histogram = prometheus_client.Histogram('predict_time_histogram', 'Time taken for prediction')


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict if the input URL is legitimate or phishing
    ---
    tags:
      - Prediction
    parameters:
      - name: input
        in: body
        type: string
        required: true
        description: The URL to be classified
        schema:
          type: object
          required:
            - input
          properties:
            input:
              type: string
              example: "https://peer.tudelft.nl"
    responses:
      200:
        description: The classification result
        schema:
          type: string
          enum: [legitimate, phishing]
        examples:
          legitimate: "legitimate"
          phishing: "phishing"
    """

    requests_counter.inc()

    input = request.json["input"]

    tic = time.perf_counter()
    [output] = model.predict([input])
    toc = time.perf_counter()

    predict_time_histogram.observe(toc - tic)

    match output:
        case "legitimate":
            legitimate_counter.inc()
        case "phishing":
            phishing_counter.inc()

    return jsonify(output)


@app.route("/agree", methods=["POST"])
def agree_counter_func():
    """
    Indicate user agreement with the classification
    ---
    tags:
      - Feedback
    responses:
      204:
        description: No Content
    """

    agree_counter.inc()
    return Response(status=204)


@app.route("/disagree", methods=["POST"])
def disagree_counter_func():
    """
    Indicate user disagreement with the classification
    ---
    tags:
      - Feedback
    responses:
      204:
        description: No Content
    """

    disagree_counter.inc()
    return Response(status=204)


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return Response(prometheus_client.generate_latest(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
