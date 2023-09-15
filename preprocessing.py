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

def PCA(x):
    scalar = StandardScaler()
    scaled_data = pd.DataFrame(scalar.fit_transform(x))
    print(scaled_data)

def remove_baseline(x):
    # polynomial fitting for baseline removal

    polyfit = pybaselines.polynomial.modpoly(x)[0]
    # uncover to visualize polyfit
    # plt.plot(data['x'], data['y'])
    # plt.plot(data['x'], polyfit)
    # plt.show()
    return x - polyfit

def standardize_byscalar(spectra, scalar):
    # multiply all values by a scalar
    # used to standardize spectra

    spectra_scaled = spectra * scalar
    # uncover to visualize scaling
    # plt.plot(np.linspace(0,len(spectra),len(spectra)),spectra)
    # plt.plot(np.linspace(0, len(spectra), len(spectra)), spectra_scaled)
    # plt.show()
    return spectra_scaled

if __name__ == '__main__':
    data = pd.read_csv('data/GSSG_20mM_1min_80mW_580.txt', names=['x', 'col2', 'y'])
    data = data.drop(columns=['col2'])

    print(standardize_byscalar(data['y'],2))