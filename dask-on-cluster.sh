module load Python/3.6.4-foss-2018a
pip install --user jupyterlab
pip install --user dask_labextension
jupyter notebook password [somesuperstrongpassword]
module load nodejs
jupyter lab --no-browser --ip="*" --port 8888
ssh -L 8787:peregrine.hpc.rug.nl:8787 s2995697@peregrine.hpc.rug.nl