import re
import os
import sys
import torch

from ultralytics.utils import TESTS_RUNNING, LOGGER, RUNS_DIR, colorstr
PREFIX = colorstr("MLflow: ")

try:
    import mlflow
    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(mlflow, '__version__')  # verify package is not directory
except (ImportError, AssertionError):
    mlflow = None


def on_train_start(trainer):
    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    full_path_uri = os.path.abspath(uri)
    # Get the current run ID
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "No active run"
    LOGGER.info(f"{PREFIX} Run ID: {run_id}")
    LOGGER.info(f"{PREFIX} tracking uri saved at: {full_path_uri}")


def on_train_epoch_end(trainer):
    trainer.stop_training = True
    torch.cuda.empty_cache()
    sys.exit("Training stopped voluntarily.")


def on_fit_epoch_end(trainer):
    """Logs training metrics to Mlflow."""
    if mlflow:
        metrics_dict = {f"{re.sub('[()]', '', k)}": float(v)
                        for k, v in trainer.metrics.items()}
        mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)


callbacks = {
    'on_train_start': on_train_start,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_epoch_end': on_train_epoch_end} if mlflow else {}
