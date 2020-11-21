# Adding model signature example

The code, adapted from this [repository](http://chappers.github.io/2020/04/19/torch-lightning-using-iris/),
is almost entirely dedicated to extracting model_signature and logging it into mlflow along with the PyTorch model.
The model is trained using all the IRIS dataset features namely sepal-length,sepal-width,petal-length,petal-width. 
Upon deployment, the model would be able to classify the test input into one of the three flower species 
namely namely SETOSA` , `VERSICOLOR`, `VIRGINICA`. The input features to the model and its output  along with their data 
types are constituents of the model signature and is stored in the MLmodel file. For more details on model_signature, see
the [docs](https://mlflow.org/docs/latest/models.html#model-metadata).


### Running the code

To run the example via MLflow, navigate to the `examples/IrisClassification/` directory and run the command

```
mlflow run .

```

This will run `iris.py` with the default set of parameters such as `--max_epochs=10`. You can see the default value in the MLproject file.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument --no-conda.

```
mlflow run . --no-conda
```

### Passing custom training parameters

The parameters can be overridden via the command line:

1. max_epochs - Number of epochs to train models. Training can be interrupted early via Ctrl+C



For example:
```
mlflow run . -P max_epochs=5 
```

Or to run the training script directly with custom parameters:
```
python iris.py \
    --max_epochs 5 \
```
