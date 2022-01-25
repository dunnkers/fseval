from fseval.config import EstimatorConfig

mutual_info_ranker = EstimatorConfig(
  name= "Mutual Info",
  estimator=dict(
    _target_="benchmark.MutualInfoClassifier",
  ),
  _estimator_type="classifier",
  multioutput=False,
  estimates_feature_importances=True,
)
