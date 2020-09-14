# on cluster
ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl
conda activate dask-env
cd dasktest
module load git
jupyter lab --no-browser --ip="*" --port 8888

# locally
ssh -L 8888:peregrine.hpc.rug.nl:8888 $PEREGRINE_USERNAME@peregrine.hpc.rug.nl