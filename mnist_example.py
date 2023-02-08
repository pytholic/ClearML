import tensorflow as tf
from clearml import Task

# Task.add_requirements("./requirements.txt")

task = Task.init(project_name="mnist_example", task_name="local_training")

parameters = {"lr": 0.001}

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

task.connect(parameters)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=parameters["lr"]),
    loss=loss_fn,
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
