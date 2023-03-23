import tensorflow as tf


def create_model(vocab_size, embedding_dim, num_labels):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(num_labels),
        ]
    )
    print(model.summary())
    return model


def compile_model(int_model):
    int_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )
