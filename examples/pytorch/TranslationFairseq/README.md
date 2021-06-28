## Fairseq en to de translation example with MLflow

This example illustrates the training and testing process of English to German translation model using [fairseq](https://ai.facebook.com/tools/fairseq/) and its integration with MLflow.
During the process, all the parameters, metrics, summary and the model were dumped into mlflow. 

### Package Requirements

Install the required packages mentioned in the following fairseq readme link

https://pytorch.org/hub/pytorch_fairseq_translation/

In addition to these, following pip packages are used

1. prettytable
2. ConfigParser
3. mlflow
4. pyximport

### Clone the repository

Follow the `Requirements and Installation` section from the following link to clone the repository

https://github.com/pytorch/fairseq

### Replace training File

Once repository is cloned, replace `train.py` with `fairseq_cli/train.py` file

### Setting MLflow Configuration

Set the MLflow tracking Uri and experiment name in mflow.cfg file and copy it to the package root directory.

### Building fairseq from source

Run the following command to build the fairseq package

`python setup.py build_ext --inplace`
`python setup.py install`

### Preprocessing

Run the command from section - `WMT'14 English to German (Convolutional)` to preprocess

https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md

### Training and Testing

Run the following command to start the training and testing process. 

`python fairseq_cli/train.py  data-bin/wmt17_en_de  --arch fconv_wmt_en_de --dropout 0.2 --criterion label_smoothed_cross_entropy --label-smoothing 0.1   --optimizer nag --clip-norm 0.1  --lr 0.5 --lr-scheduler fixed --force-anneal 50  --max-tokens 4000 --save-dir checkpoints/fconv_wmt_en_de`

Fairseq by default runs on all available gpus. 

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.
