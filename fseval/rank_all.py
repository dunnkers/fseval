#!/usr/bin/env python
import os
import sys

def exec_ranking(module_name):
    args = " ".join(str(item) for item in sys.argv[1:])
    os.system('python -m fseval.ranking.{} {}'.format(module_name, args))

if __name__ == '__main__':
    exec_ranking('rfe')
    exec_ranking('multisurf')
    exec_ranking('featboost')
    exec_ranking('featboost_knn')
    exec_ranking('featboost_deep')
    exec_ranking('stability_selection')
    exec_ranking('extratrees')
