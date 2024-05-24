import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from time import time
import json


def save_results(data, append=False):
    mode = 'at' if append else 'wt'
    with open('model_scores.txt', mode) as f:
        json.dump(data, f, indent=4)


def scale(data):
    return data / data.max()


def fit_predict_score(model, x_train, y_train, x_test, y_test, verbose=False, showtime=False):
    start = time()
    if type(model).__name__.startswith('Sequential'):
        x_train = x_train.values.reshape(x_train.shape[0], 28, 28)
        x_test = x_test.values.reshape(x_test.shape[0], 28, 28)
        y_train, y_test = y_train.values, y_test.values
        model.fit(x_train, y_train, epochs=8)
        acc_train = model.evaluate(x_train, y_train)[1]
        acc_test = model.evaluate(x_test, y_test)[1]
        print(model.get_config())
    else:
        model.fit(x_train, y_train)
        acc_train = accuracy_score(y_train, model.predict(x_train))
        acc_test = accuracy_score(y_test, model.predict(x_test))
    run_time = time() - start
    if verbose:
        print(f'{model} train score: {acc_train}, test score: {acc_test}')
        if showtime:
            print(f'{model} train time: {run_time:.3f}')
    return {'train_score': acc_train, 'test_score': acc_test, 'train_time': run_time}


def main():
    # prepare datasets
    train_set = pd.read_csv('data/fashion-mnist_train.csv')
    test_set = pd.read_csv('data/fashion-mnist_test.csv')
    x_train = scale(train_set.drop('label', axis=1))
    y_train = train_set['label']
    x_test = scale(test_set.drop('label', axis=1))
    y_test = test_set['label']

    # configure convolutional neural network with keras
    cnn = Sequential(layers=[layers.Input((28, 28, 1)),
                             layers.Conv2D(32, (3, 3), activation='relu'),
                             layers.MaxPooling2D((2, 2)),
                             layers.Flatten(),
                             layers.Dense(64, activation='relu'),
                             layers.Dense(10, activation='softmax')], name='cnn_1c-3d_e8')
    cnn.compile(optimizer='Adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # models to compare
    models = [MLPClassifier(hidden_layer_sizes=(60, 80, 60), solver='adam', alpha=0.01, max_iter=10),
              KNeighborsClassifier(n_neighbors=3),
              DecisionTreeClassifier(criterion='gini', max_depth=80),
              cnn]

    # train models, check accuracy and training time
    scores = {}
    for model in models:
        name = model.name if type(model).__name__ == 'Sequential' else str(model)
        scores[name] = fit_predict_score(model, x_train, y_train, x_test, y_test,
                                         verbose=True, showtime=True)

    # print and save results to file
    print(scores)
    save_results(scores, append=True)


if __name__ == '__main__':
    main()
