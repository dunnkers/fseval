from fseval.config import EstimatorConfig

knn_validator = EstimatorConfig(
    name="k-NN",
    estimator=dict(
        _target_="sklearn.neighbors.KNeighborsClassifier",
    ),
    _estimator_type="classifier",
    multioutput=False,
    estimates_target=True,
)
