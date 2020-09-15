# Deploying MNIST Handwritten Recognition using torchserve

## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`

## Installing Deployment plugin

move to `mlflow/pytorch/torchserve` and run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`


## Generating model file (.pt)

Run the `mnist_model.py` script which will perform training on MNIST handwritten dataset. 

By default,  the script exports the model file as `model_cnn.pt`

Command: `python mnist_model.py --epochs 5`

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating and predict deployment

This example uses tensor as input for prediction. In the example 1, mlflow cli is used to creating and predicting
the deployed models. 

In this example, the create and predict deployment can be invoked using python script. 

Run the following command to create and predict the output based on our test data - `test_data/one.png`

`python predict.py`

MNIST model would predict the handwritten digit and the result will be printed in the console. 