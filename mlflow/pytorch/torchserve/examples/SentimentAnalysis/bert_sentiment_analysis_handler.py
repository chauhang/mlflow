import logging
import os

import numpy as np
import torch
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class BERTSentimentHandler(object):
    """
    BERTSentimentHandler class. This handler takes a review / sentence
    and returns the sentiment either positive / neutral / negative
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "bert_pytorch.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "bert_sentiment_analysis.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        from bert_sentiment_analysis import SentimentClassifier

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = SentimentClassifier()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)
        self.initialized = True

    def preprocess(self, data):
        """
        Receives text in form of json and converts it into an encoding for the inference stage
        """

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        text = text.decode("utf-8")

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        encoding = tokenizer.encode_plus(
            text,
            max_length=32,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",  # Return PyTorch tensors
            truncation=True,
        )

        return encoding

    def inference(self, encoding):
        """ Predict the class whether it is Positive / Neutral / Negative
        """

        self.model.eval()
        inputs = encoding.to(self.device)
        outputs = self.model.forward(**inputs)

        out = np.argmax(outputs.cpu().detach())
        return [out.item()]

    def postprocess(self, inference_output):
        return inference_output


_service = BERTSentimentHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
