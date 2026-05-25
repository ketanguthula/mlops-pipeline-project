import mlflow


EXPERIMENT_NAME = "employee_attrition_experiment"


def compare_runs():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    if runs.empty:
        print("No runs found.")
        return

    sorted_runs = runs.sort_values(
        by="metrics.f1",
        ascending=False
    )

    best_run = sorted_runs.iloc[0]

    print("\nBest Run Based on F1 Score")
    print("-" * 40)

    print(f"Run ID: {best_run['run_id']}")
    print(f"F1 Score: {best_run['metrics.f1']:.4f}")
    print(f"Accuracy: {best_run['metrics.accuracy']:.4f}")
    print(f"Precision: {best_run['metrics.precision']:.4f}")
    print(f"Recall: {best_run['metrics.recall']:.4f}")

    print("\nModel Parameters:")
    print(f"n_estimators: {best_run['params.n_estimators']}")
    print(f"max_depth: {best_run['params.max_depth']}")


if __name__ == "__main__":
    compare_runs()