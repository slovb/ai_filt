import tensorflow as tf


EXPORT_MODEL_PATH = './saved_models/export'


def save_export_model(export_model):
    tf.saved_model.save(export_model, EXPORT_MODEL_PATH)


def load_export_model():
    return tf.saved_model.load(EXPORT_MODEL_PATH)


def create_export_model(vectorize_layer, model):
    export_model = tf.keras.Sequential(
        [vectorize_layer, model, tf.keras.layers.Activation('sigmoid')]
    )
    return export_model


def compile_export_model(export_model):
    export_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy']
    )
