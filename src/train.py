import dataset.dataset
import models.export_model
import vectorization.int_vectorization
import visualize.visualize


if __name__ == '__main__':
    epochs = 50

    # raw datasets
    train_dir = "data/train"
    test_dir = "data/test"
    raw_train_ds, raw_val_ds, raw_test_ds = dataset.dataset.raw_datasets(
        train_dir=train_dir, test_dir=test_dir
    )
    train_text = raw_train_ds.map(lambda text, _: text)

    # vectorization layers
    vocab_size = 10000
    max_sequence_length = 50
    vectorize_layer, vectorize_text = vectorization.int_vectorization.int_vectorization(
        train_text=train_text,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
    )

    # finalize the datasets
    train_ds, val_ds, test_ds = dataset.dataset.finalize_datasets(
        vectorize_text=vectorize_text,
        raw_train_ds=raw_train_ds,
        raw_val_ds=raw_val_ds,
        raw_test_ds=raw_test_ds,
    )

    # model
    import models.rnn_model as m
    model = m.create_model(
        vocab_size=vocab_size + 1, embedding_dim=256, num_labels=2, rnn_units=128
    )
    m.compile_model(model)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # evaluate
    loss, accuracy = model.evaluate(test_ds)
    print("Model accuracy: {:2.2%}".format(accuracy))

    # export model
    export_model = models.export_model.create_export_model(vectorize_layer=vectorize_layer, model=model)
    models.export_model.compile_export_model(export_model=export_model)

    loss, accuracy = export_model.evaluate(raw_test_ds)
    print("Accuracy: {:2.2%}".format(accuracy))

    models.export_model.save_export_model(export_model=export_model)

    visualize.visualize.visualize_history(history=history, epochs=epochs)
