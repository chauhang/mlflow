## Iterative Pruning
Pruning is the process of compressing a neural network that involves removing weights from a trained model.
It could be achieved by either removing the neurons in a specific layer or make the weights of the connections 
that are nearer to zero as zero. The script is an example of later. Pruning a model has an impact on
accuracy of the model, while it makes the model lightwieght. In this example, we train a Alexnet model to classify 
CIFAR 10 dataset. Basemodel along with the parameters, metrics and summary are stored in mlflow.
Subsequently, the trained Alexnet model stored in mlflow is downloaded and pruned iteratively by using the custom 
inputs provided from the cli. Ax is a platform for optimizing any kind of experiment, including machine learning experiments,
A/B tests, and simulations. [Ax](https://ax.dev/docs/why-ax.html) can optimize discrete configurations using multi-armed bandit optimization,
and continuous (e.g., integer or floating point)-valued configurations using Bayesian optimization. The objective function of the experiment
trails is "test_accuracy" based on which the model is evaluated at each trial and the best set of parameters are derived.
AXClient is used to provide the initial pruning percentage as well as decides the number
of trails to be run. The summary of the pruned model is captured in a seperate file and stored as an artifact in mflow.

### Setting up Environment variables.

Run the following command on the terminal to set the Experiment name environment variable.

`export MLFLOW_EXPERIMENT_NAME=Prune`

### Running the code to train the base model


To run the training script directly with custom parameters:
```
python alexnet.py \
    --max_epochs 3  \
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

![prune_ankan](https://user-images.githubusercontent.com/51693147/100785435-a66d6e80-3436-11eb-967a-c96b23625d1c.JPG)

We can compare the child runs in the UI, as given below:

![prune_capture](https://user-images.githubusercontent.com/51693147/100785071-2515dc00-3436-11eb-8e3a-de2d569287e6.JPG)

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.