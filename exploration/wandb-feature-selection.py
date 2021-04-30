#%%
from itertools import product
import numpy as np
import seaborn as sns
import pandas as pd
import wandb
from sklearn import datasets as sklearn_datasets
from sklearn.feature_selection import SelectKBest, chi2, f_classif,\
    mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from openml.datasets import get_dataset as get_openml_ds
from dataclasses import dataclass

@dataclass
class Dataset:
    name: str
    X: list
    y: list
    feature_names: list[str]
    target_names: list[str]

#%%
iris = sklearn_datasets.load_iris()
# new Dataset
iris_ds = Dataset('iris-flowers', iris['data'], iris['target'],
    iris['feature_names'], iris['target_names'])

#%%
credit = get_openml_ds(31)
X, y, cat, feature_names = credit.get_data(target='class')
# drop qualitative columns
to_drop = X.columns[np.array(cat)]
X = X.drop(columns=to_drop).values
y, target_names = pd.factorize(y)
target_names = target_names.to_numpy()
# new Dataset
credit_ds = Dataset('german-credit-risk', X, y,
    feature_names, target_names)


#%%
from pmlb import fetch_data
adult_data = fetch_data('adult')
X, y = fetch_data('adult', return_X_y=True)
y, target_names = pd.factorize(y)
feature_names = adult_data.drop(columns='target').columns
feature_names = feature_names.to_numpy()
# ds
adult_ds = Dataset('adult-income', X, y,
    feature_names, target_names)

#%%
datasets = dict()
for ds in [iris_ds]:
    datasets[ds.name] = ds
datasets.keys()

#%%
selectors = dict(
    chi2=chi2,
    # fscore=f_classif,
    # mutinf=mutual_info_classif
)
selectors

#%%
validators = dict(decision_tree=DecisionTreeClassifier)
validators

#%%
experiments = product(  datasets.keys(),
                        selectors.keys(),
                        validators.keys())

for dataset, selector, validator in experiments:
    ds = datasets[dataset]
    run = wandb.init(project='toy-wandb-fs', config=dict(
            dataset=ds.name,
            validator=validator,
            selector=selector
    ))
    X_train, X_test, y_train, y_test = train_test_split(
            ds.X, ds.y, stratify=ds.y, random_state=0)
    n, p_ds = X_train.shape

    validate = False
    if not validate:
        fimps, pvals = selectors[selector](X_train, y_train)
        fimps = fimps / fimps.sum()
        artifact = wandb.Artifact(ds.name, type='feature-ranking')
        columns = [f'X{p}' for p in range(1, p_ds + 1)]
        table = wandb.Table(columns=columns,
            data=[fimps])
        artifact.add(table, "chi2")
        run.log_artifact(artifact)
        # run.upsert_artifact(artifact)

    else:
        for p in range(1, p_ds):
            fs = SelectKBest(selectors[selector], k=p)
            clf = validators[validator]()
            pipeline = make_pipeline(fs, clf)
            pipeline.fit(X_train, y_train)
            
            score = pipeline.score(X_test, y_test)
            wandb.log(dict(score=score, p=p))
    wandb.finish()
