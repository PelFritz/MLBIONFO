"""Perform Principal Component Analysis (PCA)

In this script, we will generate an artificial dataset for a biological organism (e.g Zea mays). However,
our Zea mays plants will only have 15 genes. Our hypothetical scenario is that half of these plants are tolerant to
heat stress while the other half can't tolerate high heat. We will the use principal component analysis to have an
overview of our dataset, to see if truly in our created dataset we have some sort of separation between our
heat tolerant and heat susceptible plants.

This is one of the helpful steps when we perform RNA-seq analysis. PCA always helps us to see at first glance if the
factor under study contributes majority variance in our data.

This script has the following functions:
    * create_dataset: this creates our artificial dataset
    * do_pca: performs the pca for us with the help of scikit-learn
    * plot_pc: helps us to visualize our data
"""

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def create_dataset(size=500, num_genes=15):
    """

    :param num_genes: number of genes per organism
    :param size: number of organisms to have in our artificial dataset
    :return: x:array where columns represent genes and rows organisms, y: the expected class
    """
    x, y = make_classification(n_samples=size, n_features=num_genes, n_informative=3, random_state=42)
    return x, y


def do_pca(x):
    """

    :param x: dataset created by create_dataset function
    :return: principal components, explained variance ratio
    """
    scaler = StandardScaler()
    scaler.fit(x)
    x_std = scaler.transform(x)
    pca = PCA(n_components=4)
    pca.fit(x_std)
    pcs = pca.transform(x_std)
    return pcs, np.round(pca.explained_variance_ratio_, 2)


def plot_pc(x, y, perc_var):
    """

    :param x: principal components
    :param y: classes for every plant in dataset
    :param perc_var: explained variance per principal components
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(x=x[:, 0], y=x[:, 1], c=y)
    ax[0].set_xlabel(f'PC1 ({perc_var[0]*100}) %')
    ax[0].set_ylabel(f'PC2 ({perc_var[1]*100}) %')

    ax[1].scatter(x=x[:, 2], y=x[:, 3], c=y)
    ax[1].set_xlabel(f'PC3 ({perc_var[2]*100}) %')
    ax[1].set_ylabel(f'PC4 ({perc_var[3]*100}) %')
    fig.suptitle('Principal Component Analysis')
    plt.savefig("pca.png", bbox_inches='tight', dpi=300)
    plt.show()


x, y = create_dataset(size=700)
x_trans, explained_var = do_pca(x)
plot_pc(x_trans, y, explained_var)
