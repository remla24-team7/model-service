from lib_ml.model import Model
from flask import Flask, request, jsonify, Response
from prometheus_flask_exporter import PrometheusMetrics
import prometheus_client
import time

model = Model(
    model_path="model.h5",
    tokenizer_path="tokenizer.joblib",
    encoder_path="encoder.joblib",
)

app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info('app_backend_info', 'Application backend info', version='1.0.0')

# Define metrics
number_requests = prometheus_client.Counter('number_requests', 'Number of requests')
agree_counter = prometheus_client.Counter('agree_counter', 'Number of times users agree with the '
                                                           'result')
disagree_counter = prometheus_client.Counter('disagree_counter', 'Number of times users disagree '
                                                                 'with the result')
legitimate_counter = prometheus_client.Counter('legitimate_counter', 'Number of legitimate URL\'s')
scams_counter = prometheus_client.Counter('scams_counter', 'Number of scam URL\'s')
predict_time_histogram = prometheus_client.Histogram('predict_time', 'Time taken for prediction')


@app.route("/predict", methods=["POST"])
def predict():
    number_requests.inc()
    start = time.time()
    input = request.json["input"]
    [output] = model.predict([input])
    end = time.time()

    if output == "legitimate":
        legitimate_counter.inc()
    else:
        scams_counter.inc()

    elapsed = end - start
    predict_time_histogram.observe(elapsed)

    return jsonify(output)


@app.route("/agree", methods=["POST"])
def agree_counter_func():
    agree_counter.inc()
    return jsonify("")


@app.route("/disagree", methods=["POST"])
def disagree_counter_func():
    disagree_counter.inc()
    return jsonify("")


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return Response(prometheus_client.generate_latest(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
