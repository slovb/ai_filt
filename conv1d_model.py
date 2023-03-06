import tensorflow as tf

from dataset import raw_datasets
from export_model import (compile_export_model, create_export_model,
                          save_export_model)
from vectorization.int_vectorization import int_vectorization
from visualize import visualize_history


def create_model(vocab_size, num_labels):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
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
        optimizer="adam",
        metrics=["accuracy"],
    )


if __name__ == "__main__":
    epochs = 10

    # raw datasets
    train_dir = "data/train"
    test_dir = "data/test"
    raw_train_ds, raw_val_ds, raw_test_ds = raw_datasets(
        train_dir=train_dir, test_dir=test_dir
    )
    train_text = raw_train_ds.map(lambda text, _: text)

    # vectorization layers
    vocab_size = 10000
    max_sequence_length = 250
    vectorize_layer, vectorize_text = int_vectorization(
        train_text=train_text,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
    )

    # text_batch, label_batch = next(iter(raw_train_ds))
    # first_mail, first_label = text_batch[0], label_batch[0]
    # print('mail', first_mail)
    # print('label', first_label)

    # print(int_vectorize_text(first_mail, first_label)[0])
    # print(int_vectorize_layer.get_vocabulary()[51])

    # finalize the datasets
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    autotune = tf.data.AUTOTUNE

    def configure_dataset(dataset):
        return dataset.cache().prefetch(buffer_size=autotune)

    int_train_ds = configure_dataset(train_ds)
    int_val_ds = configure_dataset(val_ds)
    int_test_ds = configure_dataset(test_ds)

    # model
    model = create_model(vocab_size=vocab_size + 1, num_labels=2)
    compile_model(model)
    history = model.fit(int_train_ds, validation_data=int_val_ds, epochs=epochs)

    # evaluate
    loss, accuracy = model.evaluate(int_test_ds)
    print("Model accuracy: {:2.2%}".format(accuracy))

    # export model
    export_model = create_export_model(
        vectorize_layer=vectorize_layer, model=model
    )
    compile_export_model(export_model=export_model)

    loss, accuracy = export_model.evaluate(raw_test_ds)
    print("Accuracy: {:2.2%}", format(accuracy))

    save_export_model(export_model=export_model)

    visualize_history(history=history, epochs=epochs)