# execute on cluster
module load Python/3.6.4-foss-2018a
pip install --user jupyterlab
jupyter notebook password [somesuperstrongpassword]
module load nodejs
jupyter lab --no-browser --ip="*" --port 8888




# execute in another terminal
# ssh -L 8787:peregrine.hpc.rug.nl:8787 $PEREGRINE_USERNAME@peregrine.hpc.rug.nl
ssh -L 8888:peregrine.hpc.rug.nl:8888 $PEREGRINE_USERNAME@peregrine.hpc.rug.nl





# Enabling lab extension
pip install --user dask_labextension
# -> we need config Conda first.
conda create --name dask-env
conda env list
conda init bash
# re-open shell
conda activate dask-env
# now that Conda is configured..
jupyter labextension install dask-labextension
jupyter serverextension enable dask_labextension