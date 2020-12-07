import os
from mlflow import pyfunc
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models import Model


_PICKLE_MODULE_INFO_FILE_NAME = "pickle_module_info.txt"
_SERIALIZED_TORCH_MODEL_FILE_NAME = "model.pth"
FLAVOR_NAME = "pytorch_state_dict"


def save_model(state_dict, path, mlflow_model, pickle_module=None, **kwargs):
    pickle_module = pickle_module or mlflow_pytorch_pickle_module

    import torch
    if mlflow_model is None:
        mlflow_model = Model()

    os.makedirs(path)

    model_data_subpath = "data"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)

    pickle_module_path = os.path.join(model_data_path, _PICKLE_MODULE_INFO_FILE_NAME)
    with open(pickle_module_path, "w") as f:
        f.write(pickle_module.__name__)
    # Save pytorch model
    model_path = os.path.join(model_data_path, _SERIALIZED_TORCH_MODEL_FILE_NAME)
    torch.save(state_dict, model_path, pickle_module=pickle_module, **kwargs)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_data=model_data_subpath,
        pytorch_version=torch.__version__,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.pytorch",
        data=model_data_subpath,
        pickle_module_name=pickle_module.__name__,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))