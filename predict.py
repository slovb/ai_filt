from glob import glob
import tensorflow as tf
from dataset import raw_datasets
from export_model import load_export_model
from decode import Logger, read

if __name__ == '__main__':
    train_dir = 'data/train'
    test_dir = 'data/test'
    raw_train_ds, _, _ = raw_datasets(train_dir=train_dir, test_dir=test_dir)

    def get_string_labels(predicted_scores_batch):
        predicted_int_labels = tf.math.argmax(predicted_scores_batch, axis=1)
        predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
        return predicted_labels

    # read testdata
    pathname = 'test/*.eml'
    names = []
    inputs = []
    files = glob(pathname)
    logger = Logger()
    for filename in files:
        names.append(filename)
        inputs.append(read(filename=filename, logger=logger))

    #
    export_model = load_export_model()
    predicted_scores = export_model(inputs)
    predicted_labels = get_string_labels(predicted_scores)
    for filename, label in zip(names, predicted_labels):
        print(filename, 'är', label.numpy())
