import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas


def correct_baseline(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def PCA(x):
    scalar = StandardScaler()
    scaled_data = pd.DataFrame(scalar.fit_transform(x))
    print(scaled_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #f = open('data/GSSG_20mM_1min_80mW_580.txt', 'r')
    data = pd.read_csv('data/GSSG_20mM_1min_80mW_580.txt', names=['x', 'col2', 'y'])
    data = data.drop(columns=['col2'])
    print(data)
    #PCA(data)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
