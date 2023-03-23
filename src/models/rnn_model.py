import tensorflow as tf


def create_model(vocab_size, embedding_dim, num_labels, rnn_units):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.LSTM(rnn_units, return_sequences=False),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(num_labels),
        ]
    )
    print(model.summary())
    return model


def compile_model(int_model):
    int_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Nadam(),
        metrics=["accuracy"],
    )
