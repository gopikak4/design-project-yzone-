import time
from sklearn.metrics import accuracy_score
import numpy as np

def benchmark_model(model, X_test, y_test, runs=5):
    latencies = []
    accuracies = []

    for i in range(runs):
        start = time.time()
        predictions = model.predict(X_test)
        end = time.time()

        latency = end - start
        acc = accuracy_score(y_test, predictions)

        latencies.append(latency)
        accuracies.append(acc)

    results = {
        "avg_latency": np.mean(latencies),
        "avg_accuracy": np.mean(accuracies),
        "reliability_std": np.std(accuracies)  # stability measure
    }

    return results
