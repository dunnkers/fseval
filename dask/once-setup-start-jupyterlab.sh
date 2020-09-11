# on cluster
conda activate dask-env
cd dasktest
module load git
jupyter lab --no-browser --ip="*" --port 8888

# locally
ssh -L 8888:peregrine.hpc.rug.nl:8888 s2995697@peregrine.hpc.rug.nl