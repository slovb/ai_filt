import tensorflow as tf


def create_binary_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
    return model


def compile_binary_model(binary_model):
    binary_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )
