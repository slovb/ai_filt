import tensorflow as tf

from dataset import finalize_datasets, raw_datasets
from export_model import compile_export_model, create_export_model, save_export_model
from vectorization.int_vectorization import int_vectorization
from visualize import visualize_history


def create_model(vocab_size, embedding_dim, num_labels, rnn_units):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.LSTM(rnn_units, return_sequences=False),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(16, activation='relu'),
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


if __name__ == "__main__":
    epochs = 50

    # raw datasets
    train_dir = "data/train"
    test_dir = "data/test"
    raw_train_ds, raw_val_ds, raw_test_ds = raw_datasets(
        train_dir=train_dir, test_dir=test_dir
    )
    train_text = raw_train_ds.map(lambda text, _: text)

    # vectorization layers
    vocab_size = 10000
    max_sequence_length = 50
    vectorize_layer, vectorize_text = int_vectorization(
        train_text=train_text,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
    )

    # finalize the datasets
    train_ds, val_ds, test_ds = finalize_datasets(
        vectorize_text=vectorize_text,
        raw_train_ds=raw_train_ds,
        raw_val_ds=raw_val_ds,
        raw_test_ds=raw_test_ds,
    )

    # model
    model = create_model(
        vocab_size=vocab_size + 1, embedding_dim=64, num_labels=2, rnn_units=32
    )
    compile_model(model)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # evaluate
    loss, accuracy = model.evaluate(test_ds)
    print("Model accuracy: {:.2%}".format(accuracy))

    # export model
    export_model = create_export_model(vectorize_layer=vectorize_layer, model=model)
    compile_export_model(export_model=export_model)

    loss, accuracy = export_model.evaluate(raw_test_ds)
    print("Accuracy: {:.2%}", format(accuracy))

    save_export_model(export_model=export_model)

    visualize_history(history=history, epochs=epochs)
