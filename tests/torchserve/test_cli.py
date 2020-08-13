import os

from click.testing import CliRunner

from mlflow import deployments
from mlflow.deployments import cli

f_target = "torchserve"
f_deployment_id = "test"
f_flavor = None
f_model_uri = "./linear.pt"

env_version = "1.0"
env_model_file = "./linear_model.py"
env_handler_file = "./linear_handler.py"
sample_input_file = "./sample.json"

os.environ["VERSION"] = env_version
os.environ["MODEL_FILE"] = env_model_file
os.environ["HANDLER_FILE"] = env_handler_file


def test_create_cli_succes():
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
    assert "{}".format(f_deployment_id) in res.stdout.split("\n")[1]


def test_get_cli_success():
    runner = CliRunner()
    res = runner.invoke(
        cli.get_deployment, ["--name", f_deployment_id, "--target", f_target]
    )
    assert "{}".format(f_deployment_id) in res.stdout


def test_delete_cli_success():
    runner = CliRunner()
    res = runner.invoke(
        cli.delete_deployment, ["--name", f_deployment_id, "--target", f_target]
    )
    assert "Deployment {} is deleted".format(f_deployment_id) in res.stdout
