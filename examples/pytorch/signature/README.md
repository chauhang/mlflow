## Iris classification example with MLflow
This example demonstrates training a classification model on the Iris dataset, logging the model signature into mlflow 
and validating the model signature based on the input data. 

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/signature` directory and run the command

```
mlflow run .
```

This will run `iris_classification.py` with the default set of parameters such as  `--max_epochs=30`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P epochs=X
```

where `X` is your desired value for `epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--no-conda`.

```
mlflow run . --no-conda
```

On the training completion, the iris classification model is saved using `mlflow.pytorch.save_model` along with the model signature.

Before predicting the model output, model signature validation is performed. Only if the validation is successful, model prediction results are printed.

## Running against a custom tracking server
To configure MLflow to log to a custom (non-default) tracking location, set the ``MLFLOW_TRACKING_URI`` environment variable, e.g. via  ``export MLFLOW_TRACKING_URI=http://localhost:5000/``.  For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded)
