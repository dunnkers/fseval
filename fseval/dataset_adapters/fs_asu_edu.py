import os
from sklearn.utils import Bunch
import numpy as np
import scipy.io as sio 
from sklearn.preprocessing import scale, LabelEncoder

def load_dataset(filename):
    le = LabelEncoder()
    dataset = sio.loadmat(filename)
    X = dataset['X']
    X = scale(X)
    y = dataset['Y']
    y = np.ravel(y)
    y = le.fit_transform(y)

    return Bunch(
        data=X,
        target=y,
        filename=os.path.abspath(filename)
    )
