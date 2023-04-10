import mlflow
import tensorflow as tf
import tensorflow_datasets as tfds

user = "root"
pwd = "Yonyamin@10"
hostname = "127.0.0.1"
port = 3306
database = "mlflow"
uri = "mysql://{user}:{password}@{hostname}:{port}/{databse}"
mlflow.set_tracking_uri(uri)
