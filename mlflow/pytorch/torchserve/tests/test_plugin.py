import atexit
import json
import os
import pytest
import shutil
import subprocess
import time
from mlflow import deployments
from mlflow.exceptions import MlflowException

f_target = "torchserve"
f_deployment_id = "test"
f_flavor = None
f_model_uri = "mlflow/pytorch/torchserve/tests/resources/linear.pt"

env_version = "1.0"
env_model_file = "mlflow/pytorch/torchserve/tests/resources/linear_model.py"
env_handler_file = "mlflow/pytorch/torchserve/tests/resources/linear_handler.py"
sample_input_file = "mlflow/pytorch/torchserve/tests/resources/sample.json"
sample_output_file = "mlflow/pytorch/torchserve/tests/resources/output.json"


@pytest.fixture(scope="session")
def start_torchserve():
    if not os.path.isdir("model_store"):
        os.makedirs("model_store")
    cmd = "torchserve --start --model-store {}".format("./model_store")
    return_code = subprocess.Popen(cmd, shell=True).wait()

    count = 0
    for _ in range(5):
        value = health_checkup()
        if (
            value is not None
            and value != ""
            and json.loads(value)["status"] == "Healthy"
        ):
            time.sleep(1)
            break
        else:
            count += 1
            time.sleep(5)
    if count >= 5:
        raise Exception("Unable to connect to torchserve")
    return True


def health_checkup():
    curl_cmd = "curl http://localhost:8080/ping"
    (value, err) = subprocess.Popen(
        [curl_cmd], stdout=subprocess.PIPE, shell=True
    ).communicate()
    return value.decode("utf-8")


def stop_torchserve():
    cmd = "torchserve --stop"
    return_code = subprocess.Popen(cmd, shell=True).wait()

    if os.path.isdir("model_store"):
        shutil.rmtree("model_store")


@pytest.fixture
def clear_env_variables():
    if "VERSION" in os.environ:
        os.environ.pop("VERSION")
    if "MODEL_FILE" in os.environ:
        os.environ.pop("MODEL_FILE")
    if "HANDLER_FILE" in os.environ:
        os.environ.pop("HANDLER_FILE")


atexit.register(stop_torchserve)


def test_mandatory_params_missing(start_torchserve, clear_env_variables):
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
    isNamePresent = False
    for i in range(len(ret)):
        if list(ret[i].keys())[0] == f_deployment_id:
            isNamePresent = True
            break
    if isNamePresent:
        assert True
    else:
        assert False


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
    with open(sample_input_file) as fp:
        data = fp.read()
    pred = client.predict(f_deployment_id, data)
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
    with pytest.raises(Exception, match="No such file or directory"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(f_deployment_id, f_dummy)
        os.environ.pop("VERSION")
        os.environ.pop("MODEL_FILE")
        os.environ.pop("HANDLER_FILE")


def test_update_file_exception():
    with pytest.raises(
        Exception, match="Unable to update deployment with name %s" % f_dummy
    ):
        client = deployments.get_deploy_client(f_target)
        client.update_deployment(f_dummy)


def test_get_file_exception():
    with pytest.raises(
        Exception, match="Unable to get deployments with name %s" % f_dummy
    ):
        client = deployments.get_deploy_client(f_target)
        client.get_deployment(f_dummy)


def test_delete_file_exception():
    with pytest.raises(
        Exception, match="Unable to delete deployment for name %s" % f_dummy
    ):
        client = deployments.get_deploy_client(f_target)
        client.delete_deployment(f_dummy)


def test_predict_exception():
    with pytest.raises(
        Exception, match="Input data can either be dataframe or Json string"
    ):
        client = deployments.get_deploy_client(f_target)
        client.predict(f_dummy, "sample")


def test_predict_name_exception():
    with pytest.raises(
        Exception, match="Unable to infer the results for the name %s" % f_dummy
    ):
        client = deployments.get_deploy_client(f_target)
        with open(sample_input_file) as fp:
            data = fp.read()
        client.predict(f_dummy, data)
