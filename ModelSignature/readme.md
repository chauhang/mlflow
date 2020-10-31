## Model Signature 

In this classification example which uses IRIS dataset, we have captured the model signature i.e the input data and its type as well as the type of expected output data and have dumped into Mlflow.
The sample input data for signature inference is also stored as son file.

## Package Dependencies

Install the package dependencies using requirements.txt.

`pip install -r requirements.txt`

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/Model_Signature/example1` directory and run the command

1. Model Signature could be dumped into MLFLOW using the following command:
   `python iris.py  --max_epochs 2 --mlflow_experiment_name model_signature  --tracking_uri http://localhost:5000`

    expected output: Model signature will be stored in MLModel file and inputsample.json file would be dumped into Mlflow experiment-"model_signature"


Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.
