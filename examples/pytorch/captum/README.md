## Imdb Sentiment Analysis with Captum and MLflow

This example uses Imdb dataset for performing sentiment analysis. It uses captum library to find the word importance 
and its corresponding score and position

## Prerequisite

Download the model file from - https://github.com/pytorch/captum/blob/master/tutorials/models/imdb-model-cnn.pt
and place it in the current working directory

Download the data from - https://ai.stanford.edu/~amaas/data/sentiment/ 
and place it in the current working directory


### Running the code
Invoke the python script using following command

```
python bert_imdb_sentiment.py
```

It loads the pretrained model and findss the attribution score for given sentences. The corresponding details
are logged into mlflow as a csv file. 

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.

