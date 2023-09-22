import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from tqdm import tqdm
import math

def create_trainingandtest(x, y):
    indices = np.where(y.isna())[0]
    x = np.array(x)
    x = np.delete(x, indices, axis=0)
    y = y.dropna()
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)
    return x_train, x_test, y_train, y_test

def evaluate_withmodels(xset, yset, models, iter=10):
    """
    :param xset: xvalues, for example, absemiswork
    :param yset: yvalues, for validation
    :param models: list of models to loop through and evaluate
    :param iter: number of times model is evaluated
    :return: prints stats on each model and returns list of stats
    """
    columns = xset.columns

    output = np.empty([0,0])

    for model in tqdm(models, desc='model'):
        maes = []
        rmses = []
        r2s = []

        for i in tqdm(range(iter), leave=False, desc='training_iter'):
            # train test split
            x_train, x_test, y_train, y_test = create_trainingandtest(xset, yset)
            regressor = model
            #print(cross_val_score(regressor, xset, yset, cv=10, scoring='mean_absolute_error'))
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            maes.append(mae)
            r2s.append(r2)
            rmses.append(rmse)


        #print(f'MAE for {type(model).__name__}: {np.average(maes)}')
        #print(f'RMSE for {type(model).__name__}: {np.average(rmses)}')
        #print(f'R2 for {type(model).__name__}: {np.average(r2s)}')
        #, np.average(rmses), np.average(r2s)
        output = np.append(output, np.average(maes))
    return output

if __name__ == '__main__':
    pca = pd.read_csv('data/pca_data/allsol_580_BR_NM_5com.csv')
    yvals = pd.read_csv('data/data_580_concentrations.csv')

    x_train, x_test, y_train, y_test = create_trainingandtest(pca, yvals)