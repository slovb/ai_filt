from glob import glob
import matplotlib.pyplot as plt
from dataset import raw_datasets
from export_model import load_export_model
from decode import Logger, read
import numpy as np


def read_data(pathname):
    names = []
    inputs = []
    files = glob(pathname)
    logger = Logger()
    for filename in files:
        names.append(filename)
        inputs.append(read(filename=filename, logger=logger))
    return names, inputs


def normalize(vec):
    return vec / vec.numpy().sum(axis=1)[:, np.newaxis]


if __name__ == '__main__':
    train_dir = 'data/train'
    test_dir = 'data/test'
    raw_train_ds, _, _ = raw_datasets(train_dir=train_dir, test_dir=test_dir)
    
    # read intressant data
    namn_intressanta, intressanta = read_data('data/raw/intressanta/*.eml')
    namn_ej_intressanta, ej_intressanta = read_data('data/raw/ej_intressanta/*.eml')

    export_model = load_export_model()
    predicted_intressanta = export_model(intressanta)
    # predicted_intressanta = normalize(predicted_intressanta)
    predicted_ej_intressanta = export_model(ej_intressanta)
    # predicted_ej_intressanta = normalize(predicted_ej_intressanta)
    
    # print out errors
    for row in zip(namn_ej_intressanta, predicted_ej_intressanta):
        namn, val = row
        if val[1] > val[0]:
            print(namn, val)

    for row in zip(namn_intressanta, predicted_intressanta):
        namn, val = row
        if val[0] > val[1]:
            print(namn, val)
    
    # plot
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(predicted_intressanta[:, 0], predicted_intressanta[:, 1], c='green', marker='1', label='Intressanta')
    plt.scatter(predicted_ej_intressanta[:, 0], predicted_ej_intressanta[:, 1], c='red', marker='2', label='Ej intressanta')
    
    # lower_bound = min([min(predicted_intressanta[:, 0]), min(predicted_intressanta[:, 1]), min(predicted_ej_intressanta[:, 0]), min(predicted_ej_intressanta[:, 1])])
    lower_bound = min(np.amin(predicted_intressanta), np.amin(predicted_ej_intressanta))
    # upper_bound = max([max(predicted_intressanta[:, 0]), max(predicted_intressanta[:, 1]), max(predicted_ej_intressanta[:, 0]), max(predicted_ej_intressanta[:, 1])])
    upper_bound = max(np.amax(predicted_intressanta), np.amax(predicted_ej_intressanta))
    plt.plot([lower_bound, upper_bound], [lower_bound, upper_bound], c='blue', linestyle='dotted')
    
    plt.legend(loc='upper right')
    plt.title('Resultat')
    
    plt.subplot(1, 2, 2)
    intresse_intressanta = sorted(predicted_intressanta[:, 1] - predicted_intressanta[:, 0])
    intresse_ej_intressanta = sorted(predicted_ej_intressanta[:, 1] - predicted_ej_intressanta[:, 0])
    index_intressanta = []
    index_ej_intressanta = []
    for i in range(len(intresse_intressanta) + len(intresse_ej_intressanta)):
        if len(index_intressanta) >= len(intresse_intressanta):
            index_ej_intressanta.append(i)
        elif len(index_ej_intressanta) >= len(intresse_ej_intressanta):
            index_intressanta.append(i)
        elif intresse_intressanta[len(index_intressanta)] < intresse_ej_intressanta[len(index_ej_intressanta)]:
            index_intressanta.append(i)
        else:
            index_ej_intressanta.append(i)
    
    plt.stem(index_intressanta, intresse_intressanta, linefmt='green')
    plt.stem(index_ej_intressanta, intresse_ej_intressanta, markerfmt='D', linefmt='red')
    
    plt.title('Intresse')
    
    plt.show()
