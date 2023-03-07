import tensorflow as tf


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


def finalize_datasets(vectorize_text, raw_train_ds, raw_val_ds, raw_test_ds):
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    autotune = tf.data.AUTOTUNE

    def configure_dataset(dataset):
        return dataset.cache().prefetch(buffer_size=autotune)

    train_ds = configure_dataset(train_ds)
    val_ds = configure_dataset(val_ds)
    test_ds = configure_dataset(test_ds)
    return (train_ds, val_ds, test_ds)
