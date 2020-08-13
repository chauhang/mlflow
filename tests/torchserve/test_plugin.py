import json
import os

import pytest

from mlflow import deployments
from mlflow.exceptions import MlflowException

f_target = "torchserve"
f_deployment_id = "test"
f_flavor = None
f_model_uri = "./linear.pt"

env_version = "1.0"
env_model_file = "./linear_model.py"
env_handler_file = "./linear_handler.py"
sample_input_file = "./sample.json"


def test_mandatory_params_missing():
    with pytest.raises(Exception, match=r"Environment Variable VERSION - missing"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(f_deployment_id, f_model_uri, f_flavor, config={})

    os.environ["VERSION"] = env_version

    with pytest.raises(Exception, match=r"Environment Variable MODEL_FILE - missing"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(f_deployment_id, f_model_uri, f_flavor, config={})

    os.environ["MODEL_FILE"] = env_model_file

    with pytest.raises(Exception, match=r"Environment Variable HANDLER_FILE - missing"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(f_deployment_id, f_model_uri, f_flavor, config={})

    os.environ["HANDLER_FILE"] = env_handler_file


def test_create_deployment_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.create_deployment(f_deployment_id, f_model_uri, f_flavor, config={})
    assert isinstance(ret, dict)
    assert ret["name"] == f_deployment_id
    assert ret["flavor"] == f_flavor


def test_list_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.list_deployments()
    data = json.loads(ret[0])
    assert data["models"][0]["modelName"] == f_deployment_id


def test_get_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.get_deployment(f_deployment_id)
    print("Return value is ", json.loads(ret["deploy"]))
    assert json.loads(ret["deploy"])[0]["modelName"] == f_deployment_id


def test_wrong_target_name():
    with pytest.raises(MlflowException):
        deployments.get_deploy_client("wrong_target")


def test_update_deployment_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.update_deployment(f_deployment_id)
    assert ret["flavor"] == None


def test_predict_success():
    client = deployments.get_deploy_client(f_target)
    pred = client.predict(f_deployment_id, input=[sample_input_file])
    assert pred != None


def test_delete_success():
    client = deployments.get_deploy_client(f_target)
    assert client.delete_deployment(f_deployment_id) is None


f_dummy = "dummy"


def test_create_wrong_handler_exception():
    os.environ["HANDLER_FILE"] = f_dummy
    with pytest.raises(Exception, match="Unable to create mar file"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(f_deployment_id, f_model_uri)
    os.environ["HANDLER_FILE"] = env_handler_file


def test_create_wrong_model_exception():
    os.environ["MODEL_FILE"] = f_dummy
    with pytest.raises(Exception, match="Unable to create mar file"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(f_deployment_id, f_model_uri)
    os.environ["MODEL_FILE"] = env_model_file


def test_create_mar_file_exception():
    with pytest.raises(Exception, match="Unable to create mar file"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(f_deployment_id, f_dummy)


def test_update_file_exception():
    with pytest.raises(Exception, match="Unable to list deployments"):
        client = deployments.get_deploy_client(f_target)
        client.update_deployment(f_dummy)


def test_get_file_exception():
    with pytest.raises(Exception, match="Unable to list deployments"):
        client = deployments.get_deploy_client(f_target)
        client.get_deployment("TEST")


def test_delete_file_exception():
    with pytest.raises(Exception, match="Unable to list deployments"):
        client = deployments.get_deploy_client(f_target)
        client.delete_deployment("TEST")


def test_predict_exception():
    with pytest.raises(Exception, match="Input file not found"):
        client = deployments.get_deploy_client(f_target)
        client.predict("TEST", "sample.json")


def test_predict_name_exception():
    with pytest.raises(Exception, match="Unable to infer the results"):
        client = deployments.get_deploy_client(f_target)
        client.predict("TEST", input=[sample_input_file])
