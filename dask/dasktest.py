from dask_jobqueue import SLURMCluster
from dask import delayed
from distributed import Client

print('initializing slurm cluster.')
cluster = SLURMCluster(cores=2,
                       processes=1,
                       memory="4GB",
                       project="woodshole2",
                       walltime="00:05:00",
                       queue="regular",
                       interface="ib0")
print('initialized.')
cluster.scale(jobs=2)
print(cluster.job_script())
client = Client(cluster)

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