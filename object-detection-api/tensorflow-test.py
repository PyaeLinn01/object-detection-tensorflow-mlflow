import mlflow
import tensorflow as tf

uri = "http://0.0.0.0:5001"
experiment_name = "mnist_detection"

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        cifar = tf.keras.datasets.cifar100
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        model = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_shape=(32, 32, 3),
            classes=100,
        )
    mlflow.tensorflow.autolog(every_n_iter=1)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=2, batch_size=64)
