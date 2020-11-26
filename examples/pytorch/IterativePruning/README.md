## Iterative Pruning
Pruning is the process of compressing a neural network that involves removing weights from a trained model.
It could be achieved by either removing the neurons in a specific layer or make the weights of the connections 
that are nearer to zero as zero. The script is an example of later. Pruning a model has an impact on
accuracy of the model, while it makes the model lightwieght. In this example, we train a Alexnet model to classify 
CIFAR 10 dataset. Basemodel along with the parameters, metrics and summary are stored in mlflow.
Subsequently, the trained Alexnet model stored in mlflow is downloaded and pruned iteratively by using the custom 
inputs provided from the cli. AXClient is used to provide the initial pruning percentage as well as decides the number
of trails to be run. The summary of the pruned model is captured in a seperate file and stored as an artifact in mflow.


### Running the code to train the base model

To run the example via MLflow, navigate to the `iterative-pruning/alexnet.py` directory and run the command

```
mlflow run .
```

This will run `alexnet.py` with the default set of parameters such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X -P mlflow_experiment_name =ExperimentName -P mlflow_run_name=RunName
```

where `X` is your desired value for `max_epochs`, "ExperimentName" is the name of the mlflow experiment and "RunName" 
is the name of mlflow run name.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--no-conda`.

```
mlflow run . --no-conda

```


### Passing custom training parameters

The parameters can be overridden via the command line:

1. max_epochs - Number of epochs to train models. Training can be interrupted early via Ctrl+C
2. mlflow_experiment_name -Name of the mlflow experiment
3. mlflow_run_name - Run name of the mlflow experiment


For example:
```
mlflow run . -P max_epochs=5 -P mlflow_experiment_name=Prune -P mlflow_run_name=BaseModel
```

Or to run the training script directly with custom parameters:
```
python alexnet.py \
    --max_epochs 5  \
    --mlflow_experiment_name Prune \
    --mlflow_run_name BaseModel
```


### Running the code to Iteratively Prune the Trained Model

Run the command

 `python iterative_prune_alexnet.py  --mlflow_experiment_name Prune --mlflow_run_name BaseModel --max_epochs 5 --total_trials 1  --total_pruning_iterations 5`
  

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

In the MLflow UI, the Base Model is stored as the Parent Run and the runs for each iterations of the pruing is logged as nested child runs, as shown in the
snippets below:

![Alt text](/c/Users/ANKAN GHOSH/Documents/IPSS.jpg?raw=True "MLflow UI")

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.