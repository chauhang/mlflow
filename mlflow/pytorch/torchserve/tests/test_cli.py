import atexit
import json
import os
import pytest
import shutil
import subprocess
import time
from click.testing import CliRunner
import mock

from mlflow import deployments
from mlflow.deployments import cli

f_target = "torchserve"
f_deployment_id = "test"
f_flavor = None
f_model_uri = os.path.join("mlflow/pytorch/torchserve/tests/resources", "linear.pt")

env_version = "1.0"
env_model_file = os.path.join("mlflow/pytorch/torchserve/tests/resources", "linear_model.py")
env_handler_file = os.path.join("mlflow/pytorch/torchserve/tests/resources", "linear_handler.py")
sample_input_file = os.path.join("mlflow/pytorch/torchserve/tests/resources", "sample.json")


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


atexit.register(stop_torchserve)

@mock.patch.dict(os.environ, {"VERSION": env_version, "MODEL_FILE": env_model_file, "HANDLER_FILE": env_handler_file})
def test_create_cli_success(start_torchserve):
    client = deployments.get_deploy_client(f_target)
    runner = CliRunner()
    res = runner.invoke(
        cli.create_deployment,
        ["-f", f_flavor, "-m", f_model_uri, "-t", f_target, "--name", f_deployment_id],
    )
    assert "{} deployment {} is created".format(f_flavor, f_deployment_id) in res.stdout
    client.delete_deployment(f_deployment_id)
    res = runner.invoke(
        cli.create_deployment,
        [
            "--flavor",
            f_flavor,
            "--model-uri",
            f_model_uri,
            "--target",
            f_target,
            "--name",
            f_deployment_id,
        ],
    )
    assert "{} deployment {} is created".format(f_flavor, f_deployment_id) in res.stdout


def test_update_cli_success():
    runner = CliRunner()
    res = runner.invoke(
        cli.update_deployment,
        [
            "--flavor",
            f_flavor,
            "--model-uri",
            f_model_uri,
            "--target",
            f_target,
            "--name",
            f_deployment_id,
        ],
    )
    assert (
        "Deployment {} is updated (with flavor {})".format(f_deployment_id, f_flavor)
        in res.stdout
    )


def test_list_cli_success():
    runner = CliRunner()
    res = runner.invoke(cli.list_deployment, ["--target", f_target])
    assert "{}".format(f_deployment_id) in res.stdout


def test_get_cli_success():
    runner = CliRunner()
    res = runner.invoke(
        cli.get_deployment, ["--name", f_deployment_id, "--target", f_target]
    )
    assert "{}".format(f_deployment_id) in res.stdout

@mock.patch.dict(os.environ, {"VERSION": env_version})
def test_delete_cli_success():
    runner = CliRunner()
    res = runner.invoke(
        cli.delete_deployment, ["--name", f_deployment_id, "--target", f_target]
    )
    assert "Deployment {} is deleted".format(f_deployment_id) in res.stdout
