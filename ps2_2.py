from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import mlrose
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np



clf_rhc = mlrose.NeuralNetwork(hidden_nodes=[100, 100, 100, 100, 100], activation='relu', algorithm='random_hill_climb',
                               max_iters=10000, bias=True, is_classifier=True, learning_rate=10,
                               early_stopping=False, clip_max=5, max_attempts=100, random_state=3, curve=True)
clf_sa = mlrose.NeuralNetwork(hidden_nodes=[100, 100, 100, 100, 100], activation='relu', algorithm='simulated_annealing',
                              schedule=mlrose.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001),
                              max_iters=10000, bias=True, is_classifier=True, learning_rate=10,
                              early_stopping=False, clip_max=5, max_attempts=100, random_state=3, curve=True)
clf_ga = mlrose.NeuralNetwork(hidden_nodes=[100, 100, 100, 100, 100], activation='relu', algorithm='genetic_alg',
                              pop_size=20, mutation_prob=0.1,
                              max_iters=10000, bias=True, is_classifier=True, learning_rate=10,
                              early_stopping=False, clip_max=5, max_attempts=100, random_state=3, curve=True)
clf_gd = mlrose.NeuralNetwork(hidden_nodes=[100, 100, 100, 100, 100], activation='relu', algorithm='gradient_descent',
                              max_iters=10000, bias=True, is_classifier=True, learning_rate=1e-5,
                              early_stopping=False, clip_max=5, max_attempts=100, random_state=3, curve=True)
clfs = [clf_rhc, clf_sa, clf_ga, clf_gd]
algos = ['RHC', 'SA', 'GA', 'GD']
data = load_wine(return_X_y=True)


def evaluation(clf, X_train, X_test, y_train, y_test):
    # evaluate model by cross validation
    res_train = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    res_test = cross_val_score(clf, X_test, y_test, cv=2, scoring='accuracy')

    return sum(res_train)/len(res_train), sum(res_test)/len(res_test)


def nn_backprop():
    # train with various training size
    train_size = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    train_scores, test_scores = [], []
    for ts in train_size:
        tmp1, tmp2 = [], []
        for state in [0,1,2,3]:
            # split train and test set
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=1-ts, random_state=state)

            # train and evaluate model
            clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,
                                hidden_layer_sizes=(100, 5), random_state=1, max_iter=10000)
            res_train, res_test = evaluation(clf, X_train, X_test, y_train, y_test)
            tmp1.append(res_train)
            tmp2.append(res_test)

        train_scores.append(sum(tmp1)/len(tmp1))
        test_scores.append(sum(tmp2)/len(tmp2))

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Training size %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Training size")
    ax.plot(train_size, train_scores, marker='o', label="train")
    ax.plot(train_size, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    print(max(test_scores))


def nn_opt():
    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.3, random_state=1)

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    df = pd.DataFrame(data=np.zeros((5, len(algos))), columns=algos)
    for i in range(len(clfs)):
        print(i)
        clf = clfs[i]

        tic = time.process_time()
        clf.fit(X_train_scaled, y_train_hot)
        toc = time.process_time()
        process_time = toc - tic

        y_train_pred = clf.predict(X_train_scaled)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

        y_test_pred = clf.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        df.iloc[:, i] = [y_train_accuracy, y_test_accuracy, process_time, len(clf.fitness_curve), -clf.loss]

        # plot
        plt.title("Training Curve - " + algos[i])
        plt.xlabel("Iteration No.")
        plt.ylabel("Loss")
        plt.plot(range(len(clf.fitness_curve)), clf.fitness_curve)
        plt.show()

    df.set_index(pd.Index(['Train Accuracy', 'Test Accuracy', 'Process Time', '# of Iterations', 'Final Loss']), inplace=True)
    df.to_csv('NN Result.csv')
    print(df)


def main():
    # nn_backprop()
    nn_opt()

if __name__ == '__main__':
    main()