# Deploying MNIST Handwritten Recognition using torchserve

## Package Requirement

Following are the list of packages which needs to be installed before running the example

1. torch-model-archiver
2. torchserve
3. mlflow
4. numpy
5. matplotlib
6. mlflow-torchserve deployment plugin


## Generating model file (.pt)

Run the `mnist_model.py` script which will perform training on MNIST handwritten dataset. 

By default,  the script exports the model file as `model_cnn.pt` and generates a sample input file `sample.json'

Command: `python mnist_model.py --epochs 5`

## Setting Environment variables

Following are the mandatory variables to be set before running the torchserve plugin.
1. VERSION - version number needed for generating mar file. `export VERSION=1.0`
2. MODEL_FILE - model file path. `export MODEL_FILE_PATH=mnist_model.py`
3. HANDLER_FILE - handler file path. `export HANDLER_FILE=mnist_handler.py`

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `mnist_test`

`mlflow deployments create -t torchserve -m mnist_cnn.pt --name mnist_test`

## Running prediction based on deployed model

For testing Handwritten dataset, we are going to use a sample image placed in `test_data` directory. 
Run the following command to invoke prediction of our sample input `test_data/one.png`

`mlflow deployments predict --name mnitestst_ --target torchserve --input_path sample.json  --output_path output.json`

MNIST model would predict the handwritten digit and store the output in `output.json`.
