import tensorflow as tf


def create_model(vocab_size, embedding_dim, num_labels):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
            tf.keras.layers.Conv1D(
                16, 3, padding="valid", activation="relu", strides=2
            ),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Conv1D(
                64, 5, padding="valid", activation="relu", strides=2
            ),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Conv1D(
                64, 5, padding="valid", activation="relu", strides=2
            ),
            # tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.GlobalMaxPooling1D(),
            # tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(num_labels),
        ]
    )
    print(model.summary())
    return model


def compile_model(int_model):
    int_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )
