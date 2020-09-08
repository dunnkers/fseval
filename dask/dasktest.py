import pandas as pd

if __name__ == '__main__':
    print('initialized main thread.')
else:
    print('initialized worker.')

from sklearn.feature_selection import chi2
import numpy as np

def chi2_ranking(X, y):
    scores, _ = chi2(X, y)
    ranking = np.argsort(-scores)
    return ranking

from pymongo import MongoClient
import os
def run_fs(row, ranking_func):
    # Initialize database
    password = os.environ['MONGODB_ROOT_PASSWORD']
    ip = os.environ['MONGODB_IP']
    connstr = 'mongodb://root:{}@{}'.format(password, ip)
    dbclient = MongoClient(connstr)
    db = dbclient['results']
    fstest = db['fstest']

    # Perform feature selection
    dataset = pd.read_csv(row.path, sep='\t')
    X = dataset.drop('Class', axis=1).values
    y = dataset['Class'].values
    feature_names = dataset.drop('Class', axis=1).columns.values


    # Execute feature ranking
    import time
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
        'dataset_name': row['name'],
        'replica_no': int(row['replica_no']),
        'replicas': int(row['replicas']),
        'adapter': row['adapter'],
        'p': int(row['p']),
        'n': int(row['n']),
        'p_informative': list(map(\
            lambda s: int(s), row['p_informative'].split(',')))
    }

    result_id = fstest.insert_one(output).inserted_id
    print('result_id =', result_id)
    return result_id

if __name__ == '__main__':

    # Initialize Dask cluster
    from dask.distributed import Client, LocalCluster
    from dask import delayed
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
    tasks = []
    for idx, row in data_collection.iterrows():
        task = delayed(run_fs)(row, chi2_ranking)
        tasks.append(task)
        
    results = client.compute(tasks)
    print(results)
    for res in results:
        print('result=',res.result())

    # Dask delayed test
    print('running client.compute')
    def step_1_w_single_GPU(data):
        return "Step 1 done for: %s" % data
    def step_2_w_local_IO(data):
        return "Step 2 done for: %s" % data
    stage_1 = [delayed(step_1_w_single_GPU)(i) for i in range(3)]
    stage_2 = [delayed(step_2_w_local_IO)(s2) for s2 in stage_1]
    result_stage_2 = client.compute(stage_2)
    print(result_stage_2)
    for res in result_stage_2:
        print('result = ',res.result())
    print(client.submit(lambda x: x + 1, 10).result())
    print('end of script.')