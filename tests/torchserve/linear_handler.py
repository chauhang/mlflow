import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
from linear_model import LinearRegression

logger = logging.getLogger(__name__)


class LinearRegressionHandler(object):
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "linear.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "linear_model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = LinearRegression(1, 1)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file {0} loaded successfully".format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        data = data[0]
        number = float(data["data"])
        np_data = np.array(number, dtype=np.float32)
        np_data = np_data.reshape(-1, 1)
        data_tensor = torch.from_numpy(np_data)
        return data_tensor

    def inference(self, num):

        self.model.eval()
        inputs = Variable(num).to(self.device)
        outputs = self.model.forward(inputs)
        return [outputs.detach().item()]

    def postprocess(self, inference_output):
        return inference_output


_service = LinearRegressionHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)
    print("Data: {}".format(data))
    return data
