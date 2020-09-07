import pandas as pd

print('initialized worker.')

from sklearn.feature_selection import chi2
import numpy as np

def chi2_ranking(X, y):
    scores, _ = chi2(X, y)
    ranking = np.argsort(-scores)
    return ranking

if __name__ == '__main__':
    from distributed import Client, LocalCluster
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

    data_collection = pd.read_csv('dask/descriptor.csv')
    row = data_collection.iloc[0]
    dataset = pd.read_csv(row.path, sep='\t')
    X = dataset.drop('Class', axis=1).values
    y = dataset['Class'].values
    feature_names = dataset.drop('Class', axis=1).columns.values

    from pymongo import MongoClient
    import urllib.parse
    username = urllib.parse.quote_plus('user')
    password = urllib.parse.quote_plus('pass/word')
    # helm install my-mongodb --set architecture=standalone,useStatefulSet=true,externalAccess.enabled=true bitnami/mongodb
    

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