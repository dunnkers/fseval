import pandas as pd

if __name__ == '__main__':
    print('initialized main thread.')
else:
    print('initialized worker.')

from sklearn.feature_selection import chi2
import numpy as np
from dask import delayed

def chi2_ranking(X, y):
    scores, _ = chi2(X, y)
    ranking = np.argsort(-scores)
    return ranking

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def KNN_5Fold(X, y):
    estimator = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    scores = cross_val_score(estimator, X, y, cv=5, verbose=1)
    output = pd.DataFrame(data={
        'fold_no': range(1, len(scores) + 1),
        'score': scores
    })
    return output

from pymongo import MongoClient
import os
import time
def run_fs(ranking_func, datacol, batch_id):
    # Initialize database
    password = os.environ['MONGODB_ROOT_PASSWORD']
    ip = os.environ['MONGODB_IP']
    connstr = 'mongodb://root:{}@{}'.format(password, ip)
    dbclient = MongoClient(connstr)
    db = dbclient['restheart']
    rankingcol = db['ranking']

    # Perform feature selection
    dataset = pd.read_csv(datacol.path, sep='\t')
    X = dataset.drop('Class', axis=1).values
    y = dataset['Class'].values
    feature_names = dataset.drop('Class', axis=1).columns.values


    # Execute feature ranking
    start = time.time()
    ranking_all = ranking_func(X, y)
    time_elapsed = time.time() - start
    print('{} feature selection took {} seconds.'.format(\
        ranking_func.__name__, time_elapsed))

    # Write results to db
    ranking = pd.Series(ranking_all).dropna()
    subsets = ranking.map(lambda rank: \
        ranking[ranking <= rank].index.tolist())
    output = {
        'feature_index': ranking.index.tolist(),
        'feature_rank': ranking.values.tolist(),
        'subset': subsets.values.tolist(),
        'cpu_time': time_elapsed,
        'ranking_method': ranking_func.__name__,

        # dataset
        'dataset_id': datacol['id'],
        'dataset_name': datacol['name'],
        'replica_no': int(datacol['replica_no']),
        'replicas': int(datacol['replicas']),
        'adapter': datacol['adapter'],
        'p': int(datacol['p']),
        'n': int(datacol['n']),
        'p_informative': list(map(\
            lambda s: int(s), datacol['p_informative'].split(','))),
        'timestamp': time.time(),
        'batch_id': batch_id
    }

    result_id = rankingcol.insert_one(output).inserted_id
    print('result_id =', result_id)
    return result_id

def validate_subset(validation_func):
    print('Validating {} rank {}'.format(ranking_method, feature_rank))
    start = time.time()
    scores = validation_func(X, y)
    time_elapsed = time.time() - start


def run_validation(validation_func, datacol, batch_id):
    # Initialize database
    password = os.environ['MONGODB_ROOT_PASSWORD']
    ip = os.environ['MONGODB_IP']
    connstr = 'mongodb://root:{}@{}'.format(password, ip)
    dbclient = MongoClient(connstr)
    db = dbclient['restheart']
    rankingcol = db['ranking']
    validationcol = db['validation']


    # Perform feature selection
    data = rankingcol.find_one({
        'batch_id': batch_id,
        'dataset_id': datacol['id'],
        'replica_no': datacol['replica_no']
    })
    assert(len(data['feature_rank']) == \
           len(data['feature_index']) == \
           len(data['subset']))
    dataset = pd.read_csv(datacol.path, sep='\t')
    X = dataset.drop('Class', axis=1).values
    y = dataset['Class'].values
    feature_names = dataset.drop('Class', axis=1).columns.values
    # tasks = []
    results = []
    for subset in data['subset']:
        X_subset = np.take(X, subset, axis=1)
        start = time.time()
        scores = validation_func(X_subset, y)
        time_elapsed = time.time() - start
        n_features = np.size(X_subset, axis=1)
        print('k-NN validation took {:.4f} sec'.format(time_elapsed), 
            ' n_features =', n_features)
            
        scores['n_features'] = n_features
        scores['ranking_method'] = data['ranking_method']
        scores['cpu_time'] = time_elapsed
        scores['validation_method'] = validation_func.__name__

            # dataset
        scores['dataset_id'] = datacol['id']
        # scores['dataset_name'] = datacol['name']
        scores['replica_no'] = int(datacol['replica_no'])
        # scores['replicas'] = int(datacol['replicas'])
        # scores['adapter'] = datacol['adapter']
        # scores['p'] = int(datacol['p'])
        # scores['n'] = int(datacol['n'])
        # scores['p_informative'] = list(map(
        #     lambda s: int(s), datacol['p_informative'].split(',')))
        scores['timestamp'] = time.time()
        scores['batch_id'] = batch_id

        outputs = [row.to_dict() for _, row in scores.iterrows()]
        result = validationcol.insert_many(outputs)
        results.append(result.inserted_ids)
        # task = delayed(validate_subset)(validation_func)
        # tasks.append(task)

    return results

if __name__ == '__main__':

    # Initialize Dask cluster
    from dask.distributed import Client, LocalCluster
    # from dask_jobqueue import SLURMCluster
    # cluster = SLURMCluster(cores=2,
    #                        processes=1,
    #                        memory="4GB",
    #                        project="woodshole2",
    #                        walltime="00:05:00",
    #                        queue="regular",
    #                        interface="ib0")
    # print(cluster.job_script())
    client = Client()

    # Read data collection
    data_collection = pd.read_csv('dask/descriptor.csv')
    batch_id = time.time()
    print('batch_id =', batch_id)
    tasks = []
    for _, datacol in data_collection.iterrows():
        task = delayed(run_fs)(chi2_ranking, datacol, batch_id)
        tasks.append(task)
        
    results = client.compute(tasks)
    print(results)
    for res in results:
        print('result=',res.result())
    print('end. batch_id =',batch_id)

    # k-NN validation
    # batch_id = 1599644276.1216607
    print('batch_id =', batch_id)
    tasks = []
    for _, datacol in data_collection.iterrows():
        task = delayed(run_validation)(KNN_5Fold, datacol, batch_id)
        tasks.append(task)
        break
        
    results = client.compute(tasks)
    print(results)
    for res in results:
        print('result=',res.result())
    print('end. batch_id =',batch_id)