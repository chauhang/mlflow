import os


class Config(dict):
    def __init__(self):
        super().__init__()
        self["version"] = os.environ.get("VERSION")
        self["model_file"] = os.environ.get("MODEL_FILE")
        self["handler_file"] = os.environ.get("HANDLER_FILE")
        self["export_path"] = os.environ.get("EXPORT_PATH")
        self["extra_files"] = os.environ.get("EXTRA_FILES")
