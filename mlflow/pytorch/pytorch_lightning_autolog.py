"""
Enable automatic logging from Pytorch-Lightning to MLflow.
Logs loss and any other metrics specified in the fit
function, and optimizer data as parameters. Model checkpoints
are logged as artifacts to a 'models' directory.
MLflow will also log the parameters of the EarlyStopping callback
"""
import logging
import os
import pytorch_lightning as pl
import shutil
import tempfile
from pytorch_lightning.core.memory import ModelSummary

logging.basicConfig(level=logging.ERROR)


class __MLflowPLCallback(pl.Callback):
    """
    Callback for auto-logging metrics and parameters.
    """

    def __init__(self, log_every_n_iter=1):
        super().__init__()
        if log_every_n_iter:
            self.every_n_iter = log_every_n_iter

    def on_epoch_end(self, trainer, pl_module):
        """
        Log loss and other metrics values after each epoch
        """
        if (pl_module.current_epoch - 1) % self.every_n_iter == 0:
            metrics = trainer.callback_metrics

            for key, value in metrics.items():
                trainer.logger.experiment.log_metric(trainer.logger.run_id, key, float(value),
                                                     step=pl_module.current_epoch)

    def on_train_start(self, trainer, pl_module):
        """
        Logs Optimizer related metrics when the train begins
        """
        if trainer.early_stop_callback:
            self._log_early_stop_params(trainer, trainer.early_stop_callback)

        if hasattr(trainer, "optimizers"):
            for optimizer in trainer.optimizers:
                trainer.logger.experiment.log_param(
                    trainer.logger.run_id, "optimizer_name", type(optimizer).__name__
                )
                if hasattr(optimizer, "defaults"):
                    defaults_dict = optimizer.defaults
                    if "lr" in defaults_dict:
                        trainer.logger.experiment.log_param(
                            trainer.logger.run_id, "learning_rate", defaults_dict["lr"]
                        )

                    if "eps" in defaults_dict:
                        trainer.logger.experiment.log_param(
                            trainer.logger.run_id, "epsilon", defaults_dict["eps"]
                        )

        summary = str(ModelSummary(pl_module, mode="full"))
        tempdir = tempfile.mkdtemp()
        try:
            summary_file = os.path.join(tempdir, "model_summary.txt")
            with open(summary_file, "w") as f:
                f.write(summary)

            trainer.logger.experiment.log_artifact(trainer.logger.run_id, local_path=summary_file)
        finally:
            shutil.rmtree(tempdir)

    def on_train_end(self, trainer, pl_module):
        """
        Logs the model checkpoint into mlflow - models folder on the training end
        """

        model_file_name = "model.ckpt"
        trainer.save_checkpoint(model_file_name)
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, local_path=model_file_name, artifact_path="models")

    def on_test_end(self, trainer, pl_module):
        """
        Logs accuracy and other relevant metrics on the testing end
        """
        metrics = trainer.callback_metrics

        for key, value in metrics.items():
            trainer.logger.experiment.log_metric(trainer.logger.run_id, key, float(value))

    def _log_early_stop_params(self, trainer, early_stop_obj):
        """
        Logs Early Stop parameters into mlflow
        """
        if hasattr(early_stop_obj, "monitor"):
            trainer.logger.experiment.log_param(trainer.logger.run_id, "monitor", early_stop_obj.monitor)
        if hasattr(early_stop_obj, "mode"):
            trainer.logger.experiment.log_param(trainer.logger.run_id, "mode", early_stop_obj.mode)
        if hasattr(early_stop_obj, "patience"):
            trainer.logger.experiment.log_param(trainer.logger.run_id, "patience", float(early_stop_obj.patience))
        if hasattr(early_stop_obj, "min_delta"):
            trainer.logger.experiment.log_param(
                trainer.logger.run_id, "min_delta", float(early_stop_obj.min_delta)
            )
        if hasattr(early_stop_obj, "stopped_epoch"):
            trainer.logger.experiment.log_param(
                trainer.logger.run_id, "stopped_epoch", float(early_stop_obj.stopped_epoch)
            )