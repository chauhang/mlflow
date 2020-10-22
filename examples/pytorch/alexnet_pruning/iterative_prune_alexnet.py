import copy
import mlflow
import time
import shutil
from ax.service.ax_client import AxClient
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from torch.nn.utils import prune
import alexnet_baseline_model_cifar10 as classifier
import torch
import pytorch_lightning as pl
import pytorch_autolog
import argparse
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from pathlib import Path
import os


global pruning_amount


def load_model(artifact_uri):
    path = Path(_download_artifact_from_uri(artifact_uri))
    model_file_path = os.path.join(path, "models/data/model.pth")
    return torch.load(model_file_path)


def prune_and_save_model(model, model_filename, amount):

    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            # m = prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            m = prune.l1_unstructured(module, name="weight", amount=amount)
            m = prune.remove(module, "weight")
            name = m.weight

        if isinstance(module, torch.nn.Linear):
            # prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            m = prune.l1_unstructured(module, name="weight", amount=amount)
            m = prune.remove(module, "weight")
            name = m.weight

    torch.save(model.state_dict(), model_filename)
    m1 = torch.load(model_filename)
    os.remove(model_filename)
    return m1


def load_pruned_model(filename):
    checkpoint = torch.load(filename)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    return model


def iterative_prune(
    model, model_filename, parametrization, trainer, dm, testloader, iteration_count
):
    global pruning_amount
    if iteration_count == 0:
        pruning_amount = parametrization.get("amount")
    else:
        pruning_amount += 0.15

    mlflow.set_tags({"PRUNING PERCENTAGE": pruning_amount})
    mlflow.log_metric("pruning_percentage", pruning_amount)
    pruned_model = prune_and_save_model(model, model_filename, pruning_amount)
    model.load_state_dict(copy.deepcopy(pruned_model))

    pytorch_autolog.autolog()
    start_time = time.time()
    trainer.fit(model, dm)
    training_time = round((time.time() - start_time) / 60, 2)
    trainer.test(datamodule=testloader)
    metrics = trainer.callback_metrics
    test_accuracy = metrics.get("avg_test_acc")
    return test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_epochs",
        default=3,
        help="Describes the number of times a neural network has to be trained",
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        help="Name of MLFLOW experiment in which results would be dumped",
    )
    parser.add_argument(
        "--total_trials",
        default=3,
        help="It indicated number of AX trials to be run for the optimization experiment",
    )
    parser.add_argument(
        "--total_pruning_iterations",
        default=3,
        help="It indicated number of Iterative Pruning steps to be run on the base model",
    )
    parser.add_argument(
        "--mlflow_run_name",
        help="Name of MLFLOW experiment run with which iterations results have to be attached",
    )
    parser.add_argument(
        "--tracking_uri",
        default="http://localhost:5000/",
        help="Address of the MLFLOW tracking uri",
    )

    args = parser.parse_args()
    tracking_uri = args.tracking_uri
    mlflow_experiment_name = args.mlflow_experiment_name
    run_name = args.mlflow_run_name
    mlflow.tracking.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    identifier = client.get_experiment_by_name(mlflow_experiment_name)
    mlflow.set_experiment(mlflow_experiment_name)
    runs = client.search_runs(
        experiment_ids=identifier.experiment_id, run_view_type=ViewType.ACTIVE_ONLY
    )[0]
    runs_dict = dict(runs)
    run_id = runs_dict.get("info").run_id
    artifact_uri = runs_dict.get("info").artifact_uri

    model = load_model(artifact_uri)
    dm = classifier.DataModule()
    dm.setup("fit")
    testloader = dm.setup("test")

    mlflow.start_run(run_id=run_id, run_name=run_name)

    total_trials = int(args.total_trials)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {"name": "amount", "type": "range", "bounds": [0.05, 0.15], "value_type": "float",},
        ],
        objective_name="test_accuracy",
    )

    for k in range(total_trials):
        print("\n trial", k, "*****************************")

        parameters, trial_index = ax_client.get_next_trial()
        x = parameters.get("amount")
        x = round(x, 3)
        for i in range(int(args.total_pruning_iterations)):
            with mlflow.start_run(nested=True, run_name="Iteration" + str(i)) as child_run:
                mlflow.set_tags({"AX_TRIAL": k})

                trainer = pl.Trainer(max_epochs=int(args.max_epochs))
                model_filename = "alexnet_pruned_version" + str(i) + ".pt"

                # calling the model
                test_accuracy = iterative_prune(
                    model, model_filename, parameters, trainer, dm, testloader, i
                )

                # completion of trial
        ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())

    mlflow.end_run()
