'''
 File Created: Fri Jun 26 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from det3.ops import read_pkl
from time import time
from matplotlib import offsetbox
from sklearn import manifold

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                color=plt.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def main(X, y, output_path):
    n_samples, n_features = X.shape
    n_neighbors = 30
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne,
                "t-SNE embedding of the digits (time %.2fs)" %
                (time() - t0))
    plt.savefig(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visulize T-SNE')
    parser.add_argument('--pkl-path',
                        type=str, metavar='PKL PATH',
                        help='pkl path {"x": [#samples, #feat] (np.ndarray),' +
                                       '"y": [#samples, 1](np.ndarray)}')
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()
    data = read_pkl(args.pkl_path)
    X = data["x"]
    y = data["y"]
    main(X, y, args.output_path)