import os
import time

import numpy as np
import pandas as pd
from dask import delayed
from pymongo import MongoClient
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

DB_NAME = 'restheart'

if __name__ == '__main__':
    print('initialized main thread.')
else:
    print('initialized worker.')

def chi2_ranking(X, y):
    scores, _ = chi2(X, y)
    ranking = np.argsort(-scores)
    return ranking


def KNN_5Fold(X, y):
    estimator = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    scores = cross_val_score(estimator, X, y, cv=5, verbose=1)
    output = pd.DataFrame(data={
        'fold_no': range(1, len(scores) + 1),
        'score': scores
    })
    return output

def get_dbclient():
    connstr = 'mongodb://root:{}@{}'.format(
        os.environ['MONGODB_ROOT_PASSWORD'], os.environ['MONGODB_IP']
    )
    return MongoClient(connstr)

def run_fs(ranking_func, replica_id, batch_id):
    # Initialize database
    dbclient = get_dbclient()
    rankingcol = dbclient[DB_NAME].ranking
    datasetscol = dbclient[DB_NAME].datasets

    # Perform feature selection
    dataset = datasetscol.find_one({ '_id': replica_id })

    # Execute feature ranking
    start = time.time()
    ranking_all = ranking_func(dataset['X'], dataset['y'])
    time_elapsed = time.time() - start
    print('{} feature selection took {} seconds.'.format(\
        ranking_func.__name__, time_elapsed))

    # Write results to db
    ranking_res = pd.Series(ranking_all).dropna()
    subsets = ranking_res.map(lambda rank: \
        ranking_res[ranking_res <= rank].index.tolist())
    ranking = {
        'feature_index': ranking_res.index.tolist(),
        'feature_rank': ranking_res.values.tolist(),
        'subset': subsets.values.tolist(),
        'cpu_time': time_elapsed,
        'ranking_method': ranking_func.__name__,

        # dataset
        'replica_id': replica_id,
        'dataset_id': dataset['id'],
        'timestamp': time.time(),
        'batch_id': batch_id
    }

    ranking_id = rankingcol.insert_one(ranking).inserted_id
    print('ranking_id =', ranking_id)
    return ranking_id


def run_validation(validation_func, ranking_id):
    # Initialize database
    dbclient = get_dbclient()
    rankingcol = dbclient[DB_NAME].ranking
    validationcol = dbclient[DB_NAME].validation
    datasetscol = dbclient[DB_NAME].datasets

    # Perform feature selection
    ranking = rankingcol.find_one({ '_id': ranking_id })
    dataset = datasetscol.find_one({ '_id': replica_id })
    assert(len(ranking['feature_rank']) == \
           len(ranking['feature_index']) == \
           len(ranking['subset']))
    results = []
    for subset in ranking['subset']:
        X_subset = np.take(dataset['X'], subset, axis=1)
        start = time.time()
        scores = validation_func(X_subset, dataset['y'])
        time_elapsed = time.time() - start
        n_features = np.size(X_subset, axis=1)
        print('k-NN validation took {:.4f} sec'.format(time_elapsed), 
            ' n_features =', n_features)
            
        scores['n_features'] = n_features
        scores['ranking_method'] = ranking['ranking_method']
        scores['cpu_time'] = time_elapsed
        scores['validation_method'] = validation_func.__name__

        # dataset
        scores['replica_id'] = replica_id
        scores['dataset_id'] = dataset['id']
        scores['timestamp'] = time.time()
        scores['batch_id'] = batch_id

        outputs = [row.to_dict() for _, row in scores.iterrows()]
        result = validationcol.insert_many(outputs)
        results.append(result.inserted_ids)

    return results

def run_dataset_importer(datacol):
    # Initialize database
    dbclient = get_dbclient()
    datasetscol = dbclient[DB_NAME].datasets

    # Attach more info to dataset
    dataset = pd.read_csv(datacol.path, sep='\t')
    X = dataset.drop('Class', axis=1).values.tolist()
    y = dataset['Class'].values.tolist()
    feature_names = dataset.drop('Class', axis=1).columns.values.tolist()
    output = datacol.to_dict()
    output['X'] = X
    output['y'] = y
    output['feature_names'] = feature_names
    output['p_informative'] = list(map(\
        lambda s: int(s), datacol['p_informative'].split(',')))

    # Insert
    result = datasetscol.insert_one(output)
    return result.inserted_id

if __name__ == '__main__':
    is_local = os.getenv('FSEVAL_LOCAL', 'False') == 'True' # assume hpc

    # Initialize Dask cluster
    from dask.distributed import Client, LocalCluster
    from dask_jobqueue import SLURMCluster
    if is_local:
        print('Running on LocalCluster')
        cluster = cluster = LocalCluster()
    else:
        print('Running on SLURMCluster')
        cluster = SLURMCluster(cores=2,
                            processes=1,
                            memory="4GB",
                            project="woodshole2",
                            walltime="00:15:00",
                            queue="regular",
                            interface="ib0")
        print(cluster.job_script())
    client = Client(cluster)


    # Upload datasets to MongoDB
    # data_collection = pd.read_csv('dask/descriptor.csv')
    # for _, datacol in data_collection.iterrows():
    #     print('Importing {} [{}/{}] to MongoDB...'.format(
    #         datacol.description, datacol.replica_no, datacol.replicas))
    #     run_dataset_importer(datacol)
    #     print('Done.')

    # Feature selection
    dbclient = get_dbclient()
    datasetscol = dbclient[DB_NAME].datasets
    batch_id = time.time()
    print('batch_id =', batch_id)
    tasks = []
    datasets = list(datasetscol.find({}, { '_id':1 }))
    for dataset in datasets:
        replica_id = dataset['_id']
        ranking_id = delayed(run_fs)(chi2_ranking, replica_id, batch_id)
        val_ids = delayed(run_validation)(KNN_5Fold, ranking_id)
        tasks.append(val_ids)
        
    tasks.visualize()
    results = client.compute(tasks)
    print(results)
    for res in results:
        print('result=',res.result())
    print('end. batch_id =', batch_id)
