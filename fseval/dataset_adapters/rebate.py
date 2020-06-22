import pandas as pd
import os
from sklearn.utils import Bunch
from tqdm import tqdm
import csv
import gzip
import numpy as np
import time
import sys

def load_dataset(filename):
    print('Reading dataset `{}`...'.format(filename))
    columns = None
    rows = []
    _, file_extension = os.path.splitext(filename)
    opener = gzip.open if file_extension == '.gz' else open
    with opener(filename, mode='rt') as csvfile:
        readCSV = csv.reader(csvfile, delimiter='\t')
        for row in tqdm(readCSV):
            if columns == None:
                columns = row
            else:
                # cast all values to decimals. problematic because some should
                # actually be (ints)? probably does not influence computations.
                rows.append(np.array(row, dtype='d'))
    print('Constructing pandas DataFrame...')
    s = time.time()
    dataset = pd.DataFrame(rows, columns=columns)
    print('Done. Took {} seconds.'.format(time.time() - s))
    print('Dataset shape = {}'.format(dataset.shape))
    print('In-memory size = {} MB'.format(sys.getsizeof(dataset) / (1024 ** 2)))

    return Bunch(
        data=dataset.drop('Class', axis=1).values,
        target= dataset['Class'].values,
        feature_names= dataset.drop('Class', axis=1).columns.values,
        filename= os.path.abspath(filename)
    )
