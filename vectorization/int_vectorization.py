import tensorflow as tf


def int_vectorization(train_text, vocab_size, max_sequence_length):
    int_vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_sequence_length,
    )
    int_vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return int_vectorize_layer(text), label

    return (int_vectorize_layer, vectorize_text)
