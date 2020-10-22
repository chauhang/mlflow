## Iterative Pruning

In this example, we train a model to classify CIFAR 10 images. Basemodel along with the parameters, metrics and summary are stored in mlflow.
The base model is downloaded and pruned iteratively by using the custom inputs provided from the cli.
AXClient is used to provide the initial pruning percentage as well as decides the number of trails to be run.

## Package Dependencies

Install the package dependencies using requirements.txt.

`pip install -r requirements.txt`

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/MNIST/example1` directory and run the command

1. Base model could be dumped into MLFLOW using the following command:
   `python alexnet_baseline_model_cifar10.py  --max_epochs 15 --mlflow_experiment_name pruning_alexnet --mlflow_run_name base_line --tracking-uri http://localhost:5000`

    expected output: Base Model to be dumped into mlflow experiment -"pruning_alexnet"

2. Downloading the pretrained model, iteratively prune the model and testing it using the following command:
   `python iterative_prune_alexnet.py  --mlflow_experiment_name pruning_alexnet --mlflow_run_name base_line --max_epochs 5 --total_trials 1  --total_pruning_iterations 5`

   expected_output:Baseline model would be pruned and the pruned version of the model would be stored in mlflow with all metrics and models. Summary.txt artifact -contains the pruned number of parameters of each model.
 

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.
