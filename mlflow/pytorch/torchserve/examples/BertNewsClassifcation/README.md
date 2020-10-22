# Deploying Bert - News Classification using torchserve

## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`

## Installing Deployment plugin

move to `mlflow/pytorch/torchserve` and run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`

## Generating model file (.pt)

This example uses the pretrained bert model to perform text classification on news reports. 
 
Command: `python bert_pytorch.py --epochs 20 --tracking-uri http://localhost:5000/ --mlflow-experiment-name demo --register-model-name bert_model --num-samples 5000`

By default this example logs the requirements(`requirements.txt`) and vocabulary file(`bert_base_cased_vocab.txt`) along with the model into mlflow server.

Once the model is logged into mlflow, the model will get registered as `bert_model`. 

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `news_classification_test`

`mlflow deployments create -t torchserve -m <S3_MODEL_URI> --name news_classification_test -C "MODEL_FILE=news_classifier.py" -C "HANDLER_FILE=news_classifier_handler.py"`

## Running prediction based on deployed model

For testing the fine tuned model, a sample input text is placed in `input.json`
Run the following command to invoke prediction of our sample input 

`mlflow deployments predict --name news_classification_test --target torchserve --input_path sample.json  --output_path output.json`

Bert model would predict the classification of the given news text and stores the output in `output.json`.
