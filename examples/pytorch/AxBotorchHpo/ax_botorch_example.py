import classifier
from ax.service.ax_client import AxClient
import mlflow
import argparse


def model_training_hyperparameter_tuning(
    max_epochs, experiment_name, total_trials, params, tracking_uri
):

    mlflow.tracking.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    mlflow.start_run(run_name="BaseModel")
    dm = classifier.DataModule()
    model = classifier.LeNet(kwargs=params)
    classifier.train_evaluate(dm=dm, model=model, max_epochs=max_epochs)

    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [0.05, 0.1], "log_scale": True,},
            {"name": "weight_decay", "type": "range", "bounds": [1e-4, 1e-3],},
            {"name": "nesterov", "type": "choice", "values": [True, False],},
            {"name": "momentum", "type": "range", "bounds": [0.7, 1.0],},
        ],
        objective_name="test_accuracy",
    )

    total_trials = total_trials
    # child runs begin here.
    # with mlflow.start_run(run_id=run_id):
    for i in range(total_trials):

        with mlflow.start_run(nested=True, run_name="Trial " + str(i)) as child_run:

            parameters, trial_index = ax_client.get_next_trial()
            dm = classifier.DataModule()

            # evaluate params
            model = classifier.LeNet(kwargs=parameters)

            # calling the model
            test_accuracy = classifier.train_evaluate(
                parameterization=None, dm=dm, model=model, max_epochs=max_epochs
            )

            # completion of trial
            ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())
    best_parameters, metrics = ax_client.get_best_parameters()
    for param_name, value in best_parameters.items():
        mlflow.log_param("optimum " + param_name, value)

    mlflow.end_run()


def model_drift_flow(
    max_epochs=1, mlflow_experiment_name=None, total_trials=0, params=None, tracking_uri=None
):
    model_training_hyperparameter_tuning(
        max_epochs, mlflow_experiment_name, total_trials, params, tracking_uri
    )
    print("Baseline Model and Trial Models have been dumped into Mlflow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_epochs",
        default=2,
        help="Describes the number of times a neural network has to be trained",
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        help="Name of MLFLOW experiment in which results would be dumped",
    )
    parser.add_argument(
        "--total_trials",
        default=3,
        help="It indicated number of trials to be run for the optimization experiment",
    )
    parser.add_argument(
        "--tracking_uri",
        default="http://localhost:5000/",
        help="It indicates the address Mllfow Tracking uri",
    )

    args = parser.parse_args()

    params = {"lr": 0.005, "momentum": 0.9, "weight_decay": 0, "nesterov": False}
    # model_drift_flow(model_data,mlflow_experiment_name,drift_version_count,params)

    model_drift_flow(
        max_epochs=int(args.max_epochs),
        mlflow_experiment_name=args.mlflow_experiment_name,
        total_trials=int(args.total_trials),
        params=params,
        tracking_uri=args.tracking_uri,
    )
