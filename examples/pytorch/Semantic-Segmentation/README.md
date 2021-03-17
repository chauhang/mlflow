#Semantic Segmentation with Captum and MLflow

In this example, we demonstrate applying Captum to semantic segmentation task, to understand what pixels and regions contribute to the labeling of a particular class. We explore applying GradCAM as well as Feature Ablation to a pretrained Fully-Convolutional Network model with a ResNet-101 backbone. You can find more details [here][https://captum.ai/tutorials/Segmentation_Interpret]


### Running the code

To run the example via MLflow, navigate to the `examples/Semantic-Segmentation/` directory and run the command

```
mlflow run .

```

This will run `segmentation.py` with the default set of parameters such as `--target=6`. You can see the default value in the MLproject file. where target can be any of the trained class out of 20 in the model.

In order to run the file with custom parameters, run the command

```
mlflow run . -P img_url="url" -P target=8
```

where url can be any test image url .

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument --no-conda.

```
mlflow run . --no-conda
```
### Viewing results in the MLflow UI

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more details on MLflow tracking, see [the docs](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking).

## Logging to a custom tracking server
To configure MLflow to log to a custom (non-default) tracking location, set the MLFLOW_TRACKING_URI environment variable, e.g. via export MLFLOW_TRACKING_URI=http://localhost:5000/. For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).
