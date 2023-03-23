import tensorflow as tf


def create_model(vocab_size, embedding_dim, num_labels):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
            tf.keras.layers.Conv1D(
                64, 5, padding="valid", activation="relu", strides=2
            ),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(num_labels),
        ]
    )
    return model


def compile_model(int_model):
    int_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )
