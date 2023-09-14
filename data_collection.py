import os

import pandas as pd

if __name__ == '__main__':
    folder_path = 'data/raman_data'  # filepath to folder of data
    n = 1340 # length of spectra
    columns = [i for i in range(1, n)] # create list of same length
    data = pd.DataFrame(columns=columns) # empty dataframe with columns for each value
    names = [] # store names of files
    for file in os.listdir(folder_path):
        # loop through files in folder
        filepath = os.path.join(folder_path, file)
        text = pd.read_csv(filepath, names=['x','col2','y']) # import text file
        names.append(file)
        data.loc[len(data.index)] = text['y'].transpose() # add text file data to the dataframe
    data.insert(0, 'names', names, True) # add name column

    # adding feature columns and labels
    data['conc_GSSG'] = data['names'].str.extract(r'(?:.*GSH.*)?(\d+\s*mM)')[0] # these lines from chatGPT
    data['conc_GSSG'] = data['conc_GSSG'].str.replace('mM', '').str.strip()

    contains_580 = data[data['names'].str.contains('580')]
    contains_610 = data[data['names'].str.contains('610')]

    print(data)
    print(contains_610)
    print(contains_580)
