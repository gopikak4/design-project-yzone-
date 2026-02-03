from data_loader import load_data
from preprocess import preprocess_data
from model import load_model
from benchmark import benchmark_model

def main():
    print("Phase 1: AI Research & Benchmarking Suite for Industry")

    # Step 1: Data Loading
    df = load_data()
    print("Data Loaded Successfully")

    # Step 2: Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data Preprocessing Completed")

    # Step 3: Model Training
    model = load_model()
    model.fit(X_train, y_train)
    print("Model Training Completed")

    # Step 4: Benchmarking
    results = benchmark_model(model, X_test, y_test)

    print("\n--- Benchmarking Results (Phase 1 Review) ---")
    print(f"Average Latency     : {results['avg_latency']} seconds")
    print(f"Average Accuracy    : {results['avg_accuracy']*100:.2f}%")
    print(f"Reliability Score   : {results['reliability_std']} (lower is better)")

if __name__ == "__main__":
    main()
