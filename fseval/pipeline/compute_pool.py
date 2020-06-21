#!/usr/bin/env python
import multiprocessing 
import time
import sys
import math
import pandas as pd
import numpy as np
import os
import importlib
from datetime import datetime
import argparse

def load_dataset(datacol):
    path = '...dataset_adapters.{}'.format(datacol.adapter)
    adapter = importlib.import_module(path, __name__)
    dataset = adapter.load_dataset(datacol.path)
    return dataset

def save_output(output, outpath, append=True):
    outdir = os.path.dirname(outpath)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if append and os.path.exists(outpath): # merge with existing results
        # TODO perhaps results should be immutable.
        output = pd.concat([ pd.read_csv(outpath), output ])
    output.to_csv(outpath, index=False)

def ranking_pool(ranking_func, ranking_method, datacol, batch_id):
    filename = datacol.path
    dataset = load_dataset(datacol)

    # Execute feature ranking
    start = time.time()
    ranking_all = ranking_func(dataset.data, dataset.target)
    time_elapsed = time.time() - start
    print('{} feature selection took {} seconds.'.format(\
        ranking_method, time_elapsed))

    # Write results to .csv
    ranking = pd.Series(ranking_all).dropna()
    subsets = ranking.map(lambda rank: ','.join(\
            ranking[ranking <= rank].index.astype(str)))
    output = pd.DataFrame(data={
        'feature_index': ranking.index,
        'feature_rank': ranking.values,
        'subset': subsets.values,
        'cpu_time': time_elapsed,
        'ranking_method': ranking_method
    })

    # Save to .csv
    outpath = '{}.csv'.format(\
        os.path.join('results', batch_id, 'ranking', filename))
    save_output(output, outpath)

def validation_pool(validation_func, validation_method, datacol, batch_id):
    filename = datacol.path
    dataset = load_dataset(datacol)

    # Load feature ranking
    inpath = '{}.csv'.format(os.path.join('results', batch_id,
                                            'ranking', filename))
    data = pd.read_csv(inpath)

    # Split by each ranking method
    results = pd.DataFrame()
    for key, rankdata in data.groupby([ 'ranking_method', 'feature_rank',\
                                        'cpu_time' ]):
        # assert that `rankdata` really does come from the same experiment
        subset = rankdata['subset'].values[0]
        assert((subset == rankdata['subset']).all())

        # extract data subset from indices
        subset_indices = np.array(subset.split(',')).astype(int)
        X_subset = np.take(dataset.data, subset_indices, axis=1)

        # run validation
        ranking_method, feature_rank, _ = key
        print('Validating {} rank {}'.format(ranking_method, feature_rank))
        start = time.time()
        scores = validation_func(X_subset, dataset.target)
        time_elapsed = time.time() - start

        # store results
        scores['n_features'] = np.size(X_subset, axis=1)
        scores['ranking_method'] = ranking_method
        scores['cpu_time'] = time_elapsed
        results = results.append(scores)
            
    results['validation_method'] = validation_method
    outpath = '{}.csv'.format(\
        os.path.join('results', batch_id, 'validation', filename))
    save_output(results, outpath)

def analysis_pool(analysis_func, folder, datacol, batch_id):
    filename = datacol.path
    print('Analyzing {}'.format(datacol.get('description', filename)))

    # Load feature ranking
    inpath = '{}.csv'.format(os.path.join('results', batch_id,
                                            'ranking', filename))
    if not os.path.exists(inpath):
        return
    data = pd.read_csv(inpath)

    # Add more specific info to the ranking rows; e.g. replica_no, etc.
    dataset = datacol.drop(labels=[ # `labels` cause `dataset` is Series
        'directory', 'path', 'adapter']) # drop some cols to save space
    rankdata = data.assign(**dataset)
    # Run analysis function
    statdata = analysis_func(rankdata)

    outpath = '{}.csv'.format(\
        os.path.join('results', batch_id, folder, filename))
    save_output(statdata, outpath, append=False)


def update_slurm_jobname(descr):
    job_id = os.environ.get('SLURM_JOB_ID', None)
    if not job_id == None:
        os.system('scontrol update JobID={} name="{}"'.format(job_id, descr))

def run_pool(func, *map_args):
    # Parse arguments
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    parser = argparse.ArgumentParser(description='Run feature selection pool.')
    parser.add_argument('data_col_path')
    parser.add_argument('--batch_id')
    parser.add_argument('--cpus')
    args = parser.parse_args()

    # Read SLURM config
    batch_id =     (args.batch_id or # CLI args have precedence over env vars
                    os.environ.get('SLURM_ARRAY_JOB_ID', timestamp))
    task_id = int(  os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    cpus = int(     args.cpus or 
                    os.environ.get('SLURM_JOB_CPUS_PER_NODE', \
                                        multiprocessing.cpu_count()))
    n_jobs = int(   os.environ.get('SLURM_ARRAY_TASK_COUNT', '1'))

    # Read in data collection
    print('... reading data collection from: {}'.format(args.data_col_path))
    data_collection = pd.read_csv(args.data_col_path)

    # Prepare inputs
    datasets = np.array_split(data_collection, n_jobs)[task_id]
    # set more meaningful job description
    descrs = np.unique(datasets.get('description', datasets.get('path', [])))
    descr = descrs[0] if len(descrs) == 1 else 'task {}/{} w/ {} datasets'\
        .format(task_id, n_jobs, len(descrs))
    update_slurm_jobname(descr)
    # map arguments to inputs
    inputs = map(lambda row: [*map_args, row[1], batch_id], datasets.iterrows())

    # Run pool
    print('Running pool batch `{}` task {}/{}...'.format(\
        batch_id, task_id, n_jobs - 1))
    print('\tcpus = {}'.format(cpus))
    print('\tmap_args = {}'.format(map_args))
    print('\tn_datasets = {}'.format(len(datasets)))
    print('\tunique datasets ({}):'.format(len(descrs)))
    print('\t\t{}'.format('\n\t\t'.join(descrs)))
    begin = time.time()
    pool = multiprocessing.Pool(processes=cpus,)
    pool.starmap(func, inputs)
    pool.close()
    pool.join()
    elapsedTime = time.time() - begin
    print('Time elapsed for ' , cpus, ' workers: ', elapsedTime, ' seconds')