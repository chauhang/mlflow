import logging
import os
import requests
from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments import get_deploy_client
from mlflow.pytorch.torchserve.deploy.config import Config

_logger = logging.getLogger(__name__)


class TorchServePlugin(BaseDeploymentClient):
    def __init__(self, uri):
        super(TorchServePlugin, self).__init__(target_uri=uri)
        self.server_config = Config()
        self.management_api = "http://localhost:8081"
        self.inference_api = "http://localhost:8080"

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        mar_file_path = self.generate_mar_file(
            model_name=name,
            version=self.server_config["version"],
            model_file=self.server_config["model_file"],
            handler_file=self.server_config["handler_file"],
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
        print("\n\n")
        print(resp.status_code)
        print("\n\n")
        if resp.status_code != 202:
            print("Unable to list deployments")
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
        self, model_name, version, model_file, handler_file, model_uri
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


def run_local(name, model_uri=None, flavor=None, config=None):
    client = get_deploy_client("torchserve")
    print(client.predict(name, config["data"]))
