# Deploying Bert - Sentiment Analysis using torchserve

## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`

## Installing Deployment plugin

move to `mlflow/pytorch/torchserve` and run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`

## Generating model file (.pt)

This example uses the pretrained bert model to perform sentiment analysis on plays tore reviews. 

Run the `bert_sentiment_analysis.py` script which will fine tune the model based on play store review comments. 

By default,  the script exports the model file as `bert_pytorch.pt` and generates a sample input file `sample.json`

Command: `python bert_sentiment_analysis.py --epochs 5`

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `sentiment_test`

`mlflow deployments create -t torchserve -m bert_pytorch.pt --name sentiment_test -C "MODEL_FILE=bert_sentiment_analysis.py" -C "HANDLER_FILE=bert_sentiment_analysis_handler.py"`

## Running prediction based on deployed model

For testing the fine tuned model, a sample input text is placed in `input.json`
Run the following command to invoke prediction of our sample input 

`mlflow deployments predict --name sentiment_test --target torchserve --input_path sample.json  --output_path output.json`

Bert model would predict the sentiment of the given text and store the output in `output.json`.
