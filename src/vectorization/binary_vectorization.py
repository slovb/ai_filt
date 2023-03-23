import tensorflow as tf


def binary_vectorization(train_text, vocab_size):
    binary_vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="binary",
    )
    binary_vectorize_layer.adapt(train_text)

    def binary_vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return binary_vectorize_layer(text), label

    return (binary_vectorize_layer, binary_vectorize_text)
