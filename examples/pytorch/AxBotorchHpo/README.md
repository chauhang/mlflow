## Ax Botorch HPO example

This example illustrates hyper parameter optimization using AX client. A base model (LeNet), 
along with an optimizer with static parameters (learning rate,momentum,weight decay and nesterov) is trained.

The base model, parameters , training and testing metrics, along with model summary is dumped into mlflow during the parent run.
This is subsequently followed  by child runs intiated by the AX client, with different set of parameters for each trial. 

During each trail run, all the metrics,parameters, model and its summary are pushed into mlflow. 
The best parameters based on the objective function of the trials (in this case it is test accuracy) is stored in the base line model,
which will help in identifying the best set of parameters.

The best model which has these paramters could then be used for deployment in specific cases.

### Running the code

Run the following command

`python ax_botorch_example.py --max_epochs 10  --mlflow_experiment_name demo --total_trials 3 --tracking_uri http://localhost:5000/`

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.
