import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_collection import is_nan_string


# def separate_by_sol_andplot(data, indices):
#         output = pd.DataFrame(columns=['0', '1'])
#         for i in indices:
#             if is_nan_string(i) == True:
#                 return output
#             i = int(i)
#             rowy = data.iloc[i]
#             rowy = rowy.T
#             output = pd.concat([output, rowy])


def separate_by_sol_andplot(data, indices):
    transposed_rows = []  # Collect transposed rows in a list

    for i in indices:
        if is_nan_string(i):
            continue  # Skip NaN strings

        i = int(i)
        rowy = data.iloc[i]
        rowy = rowy.T
        transposed_rows.append(rowy)

    # Concatenate all transposed rows into the output DataFrame
    if transposed_rows:
        output = pd.concat(transposed_rows, axis=1)
    else:
        output = pd.DataFrame()  # Empty DataFrame if no valid rows were found

    return output.T


if __name__ == '__main__':
    pcadata = pd.read_csv('data/pca_data/allsol_610_BR_NM_3com.csv')
    soldata = pd.read_csv('data/separate_by_sol_610.csv')

    bsadf = separate_by_sol_andplot(pcadata, soldata['BSA'])
    pegdf = separate_by_sol_andplot(pcadata, soldata['PEG'])
    phosdf = separate_by_sol_andplot(pcadata, soldata['phos'])

    #plt.scatter(bsadf['0'], bsadf['1'])
    plt.scatter(pegdf['0'], pegdf['1'])
    plt.scatter(phosdf['0'], phosdf['1'])
    plt.show()
