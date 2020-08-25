import json
import logging
import os
from pathlib import Path, PurePath

import requests
from deploy.config import Config

from mlflow.deployments import BaseDeploymentClient
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

_logger = logging.getLogger(__name__)


class TorchServePlugin(BaseDeploymentClient):
    def __init__(self, uri):

        """
        Initializes the deployment plugin and sets the environment variables
        """
        super(TorchServePlugin, self).__init__(target_uri=uri)
        self.server_config = Config()
        self.inference_api, self.management_api = self.__get_torch_serve_port()

    def __get_torch_serve_port(self):
        """
        Reads through the config properties for torchserve inference and management api's
        """
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

    def __validate_mandatory_arguments(self):
        """
        Validate the mandatory arguments is present if not raise exception
        """

        if not self.server_config["version"]:
            raise Exception("Environment Variable VERSION - missing")

        if not self.server_config["model_file"]:
            raise Exception("Environment Variable MODEL_FILE - missing")

        if not self.server_config["handler_file"]:
            raise Exception("Environment Variable HANDLER_FILE - missing")

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        """
        Deploy the model at the model_uri to the specified target
        """

        self.__validate_mandatory_arguments()
        mar_file_path = self.__generate_mar_file(
            model_name=name,
            version=self.server_config["version"],
            model_file=self.server_config["model_file"],
            handler_file=self.server_config["handler_file"],
            extra_files=self.server_config["extra_files"],
            model_uri=model_uri,
        )
        self.__register_model(
            mar_file_path=mar_file_path,
            name=name,
            model_uri=model_uri,
            flavor=flavor,
            config=config,
        )

        return {"name": name, "flavor": flavor}

    def delete_deployment(self, name):
        """
        Delete the deployment with the name given at --name from the specified target
        """

        version = self.server_config["version"]
        url = "{}/{}/{}/{}".format(self.management_api, "models", name, version)
        resp = requests.delete(url)
        if resp.status_code != 200:
            raise Exception(
                "Unable to delete deployment for name %s. Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )
        return None

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        """
        Update the deployment with the name given at --name from the specified target

        Using -C or --config additional parameters shall be updated for the corresponding model
        """

        query_path = ""

        if config is not None:
            for key in config:
                query_path += "&" + key + "=" + str(config[key])

            query_path = query_path[1:]

        url = "{}/{}/{}?{}".format(self.management_api, "models", name, query_path)
        resp = requests.put(url)

        if resp.status_code != 202:
            raise Exception(
                "Unable to update deployment with name %s. Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )
        return {"flavor": flavor}

    def list_deployments(self):
        """
        List the names of all model deployments in the specified target. These names can be used with
        delete , update and get commands
        """
        url = "{}/{}".format(self.management_api, "models")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception(
                "Unable to list deployments. Server returned status code %s and response: %s"
                % (resp.status_code, resp.content)
            )
        return [resp.text]

    def get_deployment(self, name):
        """
        Print the detailed description of the deployment with the name given at --name
        in the specified target
        """

        url = "{}/{}/{}".format(self.management_api, "models", name)
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception(
                "Unable to get deployments with name %s. Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )
        return {"deploy": resp.text}

    def predict(self, deployment_name, input, output=None):
        """
        Predict using the inference api with the input file that is been specified using -I or --input
        and outputs the result to either stdout or specified file
        """

        version = self.server_config["version"]
        inp_path = str(input[0])
        if os.path.exists(inp_path):
            with open(inp_path, "r") as f:
                given_input = json.loads(f.read())
        else:
            raise Exception("Input file not found")
        url = "{}/{}/{}/{}".format(
            self.inference_api, "predictions", deployment_name, version
        )
        resp = requests.post(url, given_input)
        if resp.status_code != 200:
            raise Exception("Unable to infer the results")

        if output is not None and len(output) != 0:
            output_path = str(output[0])
            if os.path.exists(output_path):
                with open(output_path + "/output.json", "w") as fp:
                    json.dump({"result": resp.text}, fp)
            else:
                with open("output.json", "w") as fp:
                    json.dump({"result": resp.text}, fp)
        return resp.text

    def __generate_mar_file(
        self, model_name, version, model_file, handler_file, extra_files, model_uri
    ):

        """
        Generates mar file using the torch archiver in the specified model store path
        """
        valid_file_suffixes = [".pt", ".pth"]
        requirements_file = "requirements.txt"
        req_file_path = None

        if not os.path.isfile(model_uri):
            path = Path(_download_artifact_from_uri(model_uri))
            model_config = path / "MLmodel"
            if not model_config.exists():
                raise Exception(
                    "Failed to find MLmodel configuration within the specified model's root directory."
                )
            else:
                model_path = None
                if path.suffix in valid_file_suffixes:
                    model_uri = path
                else:
                    for file in path.iterdir():
                        if file.is_dir():
                            for sub_files in file.iterdir():
                                if sub_files.suffix in valid_file_suffixes:
                                    model_path = sub_files
                                if PurePath(sub_files).name == requirements_file:
                                    req_file_path = sub_files
                        else:
                            if file.suffix in valid_file_suffixes:
                                model_path = file
                            if PurePath(file).name == requirements_file:
                                req_file_path = file
                    if model_path is None:
                        raise RuntimeError(
                            "Model file does not have a valid suffix. Expected to be one of "
                            + ", ".join(valid_file_suffixes)
                        )
                    model_uri = model_path

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
        if req_file_path:
            cmd = "{} -r {}".format(cmd, req_file_path)

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

    def __register_model(
        self, mar_file_path, name, model_uri=None, flavor=None, config=None
    ):
        """
        Register the model using the mar file that has been generated by the archiver
        """
        query_path = mar_file_path
        if config is not None:
            for key in config:
                query_path += "&" + key + "=" + str(config[key])
        else:
            query_path += "&initial_workers=" + str(1)

        url = "{}/{}?url={}".format(self.management_api, "models", query_path)
        resp = requests.post(url=url)

        if resp.status_code != 200:
            raise Exception("Unable to register the model")
        return True


def run_local(name, model_uri, flavor=None, config=None):
    raise Exception("Yet to be implemented!")


def target_help():
    help_string = (
        "\nmlflow-torchserve plugin integrates torchserve to mlflow deployment pipeline. "
        "For detailed explanation and to see multiple examples, checkout the Readme at "
        "README https://github.com/chauhang/mlflow/blob/master/README.rst \n\n"
        "Following are the various options available using the existing mlflow deployments functions\n\n"
        "CREATE: \n"
        "Deploy the model at 'model_uri' to the specified target.\n"
        "Additional plugin-specific arguments may also be passed to this command, via -C key=value\n\n"
        "UPDATE: \n"
        "Update the deployment with ID 'deployment_id' in the specified target.\n"
        "You can update the URI of the model and/or the flavor of the deployed model (in which case the model URI must also be specified).\n"
        "Additional plugin-specific arguments may also be passed to this command, via '-C key=value'.\n\n"
        "DELETE: \n"
        "Delete the deployment with name given at '--name' from the specified target.\n\n"
        "LIST: \n"
        "List the names of all model deployments in the specified target. These names can be used with the 'delete', 'update', and 'get' commands.\n\n"
        "GET: \n"
        "Print a detailed description of the deployment with name given at '--name' in the specified target.\n\n"
        "HELP: \n"
        "Display additional help for a specific deployment target, e.g. info on target-specific config options and the target's URI format.\n\n"
        "RUN-LOCAL: \n"
        "Deploy the model locally. This has very similar signature to 'create' API\n\n"
        "PREDICT: \n"
        "Predict the results for the deployed model for the given input(s)\n\n"
    )

    return help_string
