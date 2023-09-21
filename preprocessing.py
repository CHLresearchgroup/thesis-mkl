import matplotlib.pyplot as plt
import pandas as pd
import pybaselines.polynomial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from BaselineRemoval import BaselineRemoval
import matplotlib.pyplot as plt
import scipy
import numpy
import sklearn
from pybaselines import polynomial

def standardize_byscalar(spectra, scalar):
    # multiply all values by a scalar
    # used to standardize spectra

    spectra_scaled = spectra * scalar
    # uncover to visualize scaling
    # plt.plot(np.linspace(0,len(spectra),len(spectra)),spectra)
    # plt.plot(np.linspace(0, len(spectra), len(spectra)), spectra_scaled)
    # plt.show()
    return spectra_scaled

def normalize(spectra):
    names = spectra['names']
    concentrations = spectra['conc_GSSG']

    spectra = spectra.drop(columns=['names','conc_GSSG','index'])

    normalized = []

    for index, rowy in spectra.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = StandardScaler().fit_transform(row)
        row = row.flatten()
        row = row.reshape(1, -1)
        normalized_df = pd.DataFrame(row, columns=rowy.index)
        normalized.append(normalized_df)
    normalized_spectra = pd.concat(normalized, axis=0, ignore_index=True)

    #normalized_spectra['names'] = names
    normalized_spectra['conc_GSSG'] = concentrations

    return normalized_spectra


def PCA1(data, known_components):
    pca = PCA(n_components=2)

    # Fit the PCA model with your known components
    pca.components_ = known_components

    # Transform your data using the PCA model (X should be your data)
    X_transformed = pca.transform(data)

    return X_transformed

def PCA2(data):
    pca = PCA(n_components=2)


def remove_baseline(spectra):
    # polynomial fitting for baseline removal

    concentrations = spectra['conc_GSSG']

    spectra = spectra.drop(columns=['conc_GSSG'])

    baseline_removed = []
    for index, rowy in spectra.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = row.flatten()
        row_polyfit = pybaselines.polynomial.modpoly(row)[0]
        # plt.plot(row)
        # plt.plot(row_polyfit)
        # plt.show()
        row = row - row_polyfit
        row = row.flatten()
        row = row.reshape(1, -1)
        normalized_df = pd.DataFrame(row, columns=rowy.index)
        baseline_removed.append(normalized_df)

    baselined_spectra = pd.concat(baseline_removed, axis=0, ignore_index=True)
    baselined_spectra['conc_GSSG'] = concentrations

    return baselined_spectra

if __name__ == '__main__':
    data = pd.read_csv('data/data_610.csv')

    print(remove_baseline(data))
    plt.plot(remove_baseline(data).iloc[4])
    plt.show()

    # cleaned up 580 data by removing all of the messed up spectra manually
    # need to do this for the 610 spectra
    # then need to remove the baseline in all of the spectra
    # then save
    # then normalize
    # then u can do PCA



