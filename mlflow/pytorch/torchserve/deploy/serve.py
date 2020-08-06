import logging
import os
import requests
from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments import get_deploy_client
from mlflow.pytorch.torchserve.deploy.config import Config
import json

_logger = logging.getLogger(__name__)


class TorchServePlugin(BaseDeploymentClient):
    def __init__(self, uri):
        super(TorchServePlugin, self).__init__(target_uri=uri)
        self.server_config = Config()
        self.inference_api, self.management_api = self.get_torch_serve_port()
        self.validate_mandatory_arguments()

    def get_torch_serve_port(self):
        config_properties = self.server_config["config_properties"]
        inference_port = "http://localhost:8080"
        management_port = "http://localhost:8081"
        address_strings = self.server_config["torchserve_address_names"]
        if config_properties is not None and os.path.exists(config_properties):
            with open(config_properties, "r") as f:
                lines = f.readlines()
                for line in lines:
                    name = line.strip().split("=")
                    if name[0] == address_strings[0] and name[1] is not None:
                        inference_port = name[1]
                    if name[0] == address_strings[1] and name[1] is not None:
                        management_port = name[1]
        return inference_port, management_port

    def validate_mandatory_arguments(self):
        if not self.server_config["version"]:
            raise Exception("Environment Variable VERSION - missing")

        if not self.server_config["model_file"]:
            raise Exception("Environment Variable MODEL_FILE - missing")

        if not self.server_config["handler_file"]:
            raise Exception("Environment Variable HANDLER_FILE - missing")

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        mar_file_path = self.generate_mar_file(
            model_name=name,
            version=self.server_config["version"],
            model_file=self.server_config["model_file"],
            handler_file=self.server_config["handler_file"],
            extra_files=self.server_config["extra_files"],
            model_uri=model_uri,
        )

        self.register_model(
            mar_file_path=mar_file_path, name=name, model_uri=model_uri, flavor=flavor
        )
        return {"name": name, "flavor": flavor}

    def delete_deployment(self, name):
        version = self.server_config["version"]
        url = "{}/{}/{}/{}".format(self.management_api, "models", name, version)
        resp = requests.delete(url)
        if resp.status_code != 200:
            raise Exception("Unable to list deployments")
        return None

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        url = "{}/{}/{}?{}".format(self.management_api, "models", name, "min_worker=3")
        resp = requests.put(url)
        if resp.status_code != 202:
            raise Exception("Unable to list deployments")
        return {"flavor": flavor}

    def list_deployments(self):
        url = "{}/{}".format(self.management_api, "models")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception("Unable to list deployments")
        return [resp.text]

    def get_deployment(self, name):
        url = "{}/{}/{}".format(self.management_api, "models", name)
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception("Unable to list deployments")
        return {"deploy": resp.text}

    def predict(self, deployment_name, data):
        version = self.server_config["version"]
        url = "{}/{}/{}/{}".format(
            self.inference_api, "predictions", deployment_name, version
        )
        data = {"data": data}
        resp = requests.post(url, data)
        if resp.status_code != 200:
            raise Exception("Unable to list deployments")
        return resp.text

    def generate_mar_file(
        self, model_name, version, model_file, handler_file, extra_files, model_uri
    ):
        export_path = self.server_config["export_path"]
        if export_path:
            model_store = export_path
        else:
            model_store = "model_store"
            if not os.path.isdir(model_store):
                os.makedirs(model_store)

        cmd = (
            "torch-model-archiver --force --model-name {} "
            "--version {} --model-file {} --serialized-file {} "
            "--handler {} --export-path {}".format(
                model_name, version, model_file, model_uri, handler_file, model_store
            )
        )
        if extra_files:
            cmd = "{} --extra-files {}".format(cmd, extra_files)

        return_code = os.system(cmd)
        if return_code != 0:
            _logger.error(
                "Error when attempting to load and parse JSON cluster spec from file %s",
                cmd,
            )
            raise Exception("Unable to create mar file")

        if export_path:
            mar_file = "{}/{}.mar".format(export_path, model_name)
        else:
            mar_file = "{}/{}/{}.mar".format(os.getcwd(), model_store, model_name)
            if os.path.isfile(mar_file):
                print("{} file generated successfully".format(mar_file))
        return mar_file

    def register_model(self, mar_file_path, name, model_uri=None, flavor=None):
        url = "{}/{}?url={}".format(self.management_api, "models", mar_file_path)
        resp = requests.post(url=url)
        if resp.status_code != 200:
            raise Exception("Unable to register the model")
        else:
            self.update_deployment(name, model_uri=model_uri, flavor=flavor)
        return True


def run_local(name, model_uri, flavor=None, config=None):
    pass

def target_help():
    pass

def predict_result(name, model_uri, input, output=None):
    client = get_deploy_client('torchserve')
    inp_path = str(input[0])
    if os.path.exists(inp_path):
        with open(inp_path, "r") as f:
            given_input = json.loads(f.read())
        result = client.predict(name, given_input['data'])
        if output is None:
            print("Result is: {}".format(result))
        return True
    else:
        return False