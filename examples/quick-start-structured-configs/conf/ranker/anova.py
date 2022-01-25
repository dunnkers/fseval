from fseval.config import EstimatorConfig

anova_ranker = EstimatorConfig(
    name="ANOVA F-value",
    estimator=dict(
        _target_="benchmark.ANOVAFValueClassifier",
    ),
    _estimator_type="classifier",
    estimates_feature_importances=True,
)
