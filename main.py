import tensorflow as tf
import matplotlib.pyplot as plt

EXPORT_MODEL_PATH = './saved_models/export'


def save_export_model(model):
    tf.saved_model.save(model, EXPORT_MODEL_PATH)


def load_export_model():
    return tf.saved_model.load(EXPORT_MODEL_PATH)


def visualize_history(history, epochs) -> None:
    '''Visualize the accuracy and loss plots'''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def raw_datasets(train_dir, test_dir, seed=42):
    batch_size = 32
    validation_split = 0.2
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        subset='training',
        seed=seed
    )
    # for i, label in enumerate(raw_train_ds.class_names):
    #     print("Label", i, "corresponds to", label)
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        subset='validation',
        seed=seed
    )
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size
    )
    return (raw_train_ds, raw_val_ds, raw_test_ds)


def binary_vectorization(train_text, vocab_size):
    binary_vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='binary',
    )
    binary_vectorize_layer.adapt(train_text)

    def binary_vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return binary_vectorize_layer(text), label
    return (binary_vectorize_layer, binary_vectorize_text)


def int_verctorization(train_text, vocab_size, max_sequence_length):
    int_vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )
    int_vectorize_layer.adapt(train_text)

    def int_vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return int_vectorize_layer(text), label
    return (int_vectorize_layer, int_vectorize_text)


if __name__ == '__main__':
    epochs = 10
    
    # raw datasets
    train_dir = 'data/train'
    test_dir = 'data/test'
    raw_train_ds, raw_val_ds, raw_test_ds = raw_datasets(train_dir=train_dir, test_dir=test_dir)
    train_text = raw_train_ds.map(lambda text, _: text)

    # vectorization layers
    vocab_size = 10000
    max_sequence_length = 250
    binary_vectorize_layer, binary_vectorize_text = binary_vectorization(train_text=train_text, vocab_size=vocab_size)
    int_vectorize_layer, int_vectorize_text = int_verctorization(train_text=train_text, vocab_size=vocab_size, max_sequence_length=max_sequence_length)

    # text_batch, label_batch = next(iter(raw_train_ds))
    # first_mail, first_label = text_batch[0], label_batch[0]
    # print('mail', first_mail)
    # print('label', first_label)

    # print(int_vectorize_text(first_mail, first_label)[0])
    # print(int_vectorize_layer.get_vocabulary()[51])

    # finalize the datasets
    binary_train_ds = raw_train_ds.map(binary_vectorize_text)
    binary_val_ds = raw_val_ds.map(binary_vectorize_text)
    binary_test_ds = raw_test_ds.map(binary_vectorize_text)

    int_train_ds = raw_train_ds.map(int_vectorize_text)
    int_val_ds = raw_val_ds.map(int_vectorize_text)
    int_test_ds = raw_test_ds.map(int_vectorize_text)

    autotune = tf.data.AUTOTUNE

    def configure_dataset(dataset):
        return dataset.cache().prefetch(buffer_size=autotune)

    binary_train_ds = configure_dataset(binary_train_ds)
    binary_val_ds = configure_dataset(binary_val_ds)
    binary_test_ds = configure_dataset(binary_test_ds)

    int_train_ds = configure_dataset(int_train_ds)
    int_val_ds = configure_dataset(int_val_ds)
    int_test_ds = configure_dataset(int_test_ds)

    # binary model
    binary_model = tf.keras.Sequential([tf.keras.layers.Dense(4)])
    binary_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    binary_history = binary_model.fit(
        binary_train_ds, validation_data=binary_val_ds, epochs=epochs
    )

    # int model
    def create_int_model(vocab_size, num_labels):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
            tf.keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=2),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(num_labels)
        ])
        return model
    int_model = create_int_model(vocab_size=vocab_size + 1, num_labels=2)
    int_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    int_history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=epochs)

    # evaluate
    binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)
    int_loss, int_accuracy = int_model.evaluate(int_test_ds)

    print("Binary model accuracy: {:2.2%}".format(binary_accuracy))
    print("Int model accuracy: {:2.2%}".format(int_accuracy))

    # export model
    export_int_model = tf.keras.Sequential(
        [binary_vectorize_layer, binary_model, tf.keras.layers.Activation('sigmoid')]
    )
    export_int_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy']
    )
    loss, accuracy = export_int_model.evaluate(raw_test_ds)
    print("Accuracy: {:2.2%}", format(accuracy))

    save_export_model(export_int_model)

    visualize_history(history=binary_history, epochs=epochs)
    visualize_history(history=int_history, epochs=epochs)
