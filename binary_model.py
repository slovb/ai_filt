import tensorflow as tf

from dataset import raw_datasets
from export_model import create_export_model, compile_export_model, save_export_model
from visualize import visualize_history
from vectorization.binary_vectorization import binary_vectorization


def create_binary_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
    return model


def compile_binary_model(binary_model):
    binary_model.compile(
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
    vectorize_layer, vectorize_text = binary_vectorization(
        train_text=train_text, vocab_size=vocab_size
    )

    # finalize the datasets
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    autotune = tf.data.AUTOTUNE

    def configure_dataset(dataset):
        return dataset.cache().prefetch(buffer_size=autotune)

    train_ds = configure_dataset(train_ds)
    val_ds = configure_dataset(val_ds)
    test_ds = configure_dataset(test_ds)

    # binary model
    model = create_binary_model()
    compile_binary_model(binary_model=model)
    binary_history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs
    )

    # evaluate
    loss, accuracy = model.evaluate(test_ds)

    print("Binary model accuracy: {:2.2%}".format(accuracy))

    # export model
    export_model = create_export_model(
        vectorize_layer=vectorize_layer, model=model
    )
    compile_export_model(export_model)
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print("Accuracy: {:2.2%}", format(accuracy))

    save_export_model(export_model=export_model)

    visualize_history(history=binary_history, epochs=epochs)
