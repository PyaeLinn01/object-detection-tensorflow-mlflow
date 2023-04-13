import base64
import io
import os
from sys import version_info

import cloudpickle
import mlflow
import tensorflow as tf
from get_or_create_mlflow_experiment import get_experiment_id
from tensorflow import keras

MODELS_DIR = "models"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
