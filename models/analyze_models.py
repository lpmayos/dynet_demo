import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_loss_and_accuracy(file_path):
    losses = []
    accuracies = []
    with open(file_path) as f:
        for line in f.readlines():
            split = line.split()
            if 'Epoch' in split:
                losses.append(float(split[-3]))
                accuracies.append(float(split[-1]))
    return losses, accuracies


def plot_training_data(df, plot_name):
    df = df.fillna(0)
    ax = sns.lineplot(data=df)
    # ax = sns.lineplot(x='training_point', y=plot_name, data=df)
    ax.figure.savefig('%s.png' % plot_name)
    plt.clf()


def analyze_models_learning(path):
    file_paths = []
    accuracies = []
    losses = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file == 'log.log':
                file_losses, file_accuracies = get_loss_and_accuracy(file_path)
                file_paths.append(file_path)
                losses.append(file_losses)
                accuracies.append(file_accuracies)

    df1 = pd.DataFrame(accuracies, dtype=float)
    df2 = pd.DataFrame(losses, dtype=float)

    plot_training_data(df1.transpose(), "accuracy")
    plot_training_data(df2.transpose(), "loss")


if __name__ == '__main__':
    analyze_models_learning('.')