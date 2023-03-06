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
